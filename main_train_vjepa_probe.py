#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Clean MViT Training Script using LaViLa Infrastructure
Based on feature extraction script but focused on training
"""

import argparse
from collections import OrderedDict
import csv
import json
import numpy as np
import os
import sys
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
import torch.nn.parallel
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
from sklearn.metrics import confusion_matrix
import wandb

# For loading .safetensors files
try:
    from safetensors.torch import load_file as load_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    print("Warning: safetensors not available. Install with 'pip install safetensors'")

from lavila.data import datasets
from lavila.data import datasets_flow
from lavila.data.datasets import VideoCaptionDatasetBase, get_frame_ids, video_loader_by_frames
from lavila.data.video_transforms import Permute, SpatialCrop, TemporalCrop
from lavila.models.tokenizer import SimpleTokenizer
from lavila.utils import distributed as dist_utils
from lavila.utils.evaluation import accuracy
from lavila.utils.meter import AverageMeter, ProgressMeter
from lavila.utils.preprocess import generate_label_map
from lavila.utils.random import random_seed
from lavila.utils.scheduler import cosine_scheduler

from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights


VJEPA2_MODEL_SPECS = {
    'vjepa2_large': {'hub_name': 'vjepa2_vit_large', 'crop_size': 256},
    'vjepa2_huge': {'hub_name': 'vjepa2_vit_huge', 'crop_size': 256},
    'vjepa2_giant': {'hub_name': 'vjepa2_vit_giant', 'crop_size': 256},
    'vjepa2_giant_384': {'hub_name': 'vjepa2_vit_giant_384', 'crop_size': 384},
}


def _get_vjepa2_attentive_pooler():
    vjepa2_root = os.path.join(os.path.dirname(__file__), "thirdparty", "vjepa2")
    if os.path.isdir(vjepa2_root) and vjepa2_root not in sys.path:
        sys.path.append(vjepa2_root)
    try:
        from src.models.attentive_pooler import AttentivePooler
    except Exception as exc:
        raise RuntimeError(
            "Failed to import V-JEPA2 attentive pooler. "
            "Ensure thirdparty/vjepa2 is present and on the Python path."
        ) from exc
    return AttentivePooler


def _load_vjepa2_encoder(variant_key):
    if variant_key not in VJEPA2_MODEL_SPECS:
        raise ValueError(f'Unsupported V-JEPA2 variant: {variant_key}')
    hub_name = VJEPA2_MODEL_SPECS[variant_key]['hub_name']
    encoder, _ = torch.hub.load('facebookresearch/vjepa2', hub_name)
    return encoder


class VJEPA2MeanPoolClassifier(nn.Module):
    """Mean-pool classifier on top of a V-JEPA2 encoder."""

    def __init__(self, variant_key, num_classes, dropout=0.5):
        super().__init__()
        self.encoder = _load_vjepa2_encoder(variant_key)
        self.num_features = self.encoder.embed_dim
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.num_features, num_classes),
        )

    def forward(self, x, use_checkpoint=False):
        features = self.encoder(x)  # [B, N, C]
        pooled = features.mean(dim=1)
        return self.classifier(pooled)


class VJEPA2ProbeHead(nn.Module):
    """Attentive probe head matching the V-JEPA2 frozen evaluation setup."""

    def __init__(
        self,
        embed_dim,
        num_classes,
        num_heads,
        depth,
        mlp_ratio=4.0,
        dropout=0.0,
        use_activation_checkpointing=True,
    ):
        super().__init__()
        AttentivePooler = _get_vjepa2_attentive_pooler()
        self.pooler = AttentivePooler(
            num_queries=1,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
            use_activation_checkpointing=use_activation_checkpointing,
        )
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.pooler(x).squeeze(1)
        x = self.dropout(x)
        return self.classifier(x)


class VJEPA2MultiTaskProbeHead(nn.Module):
    """Attentive probe head with verb/noun/action classifiers."""

    def __init__(
        self,
        embed_dim,
        num_verb_classes,
        num_noun_classes,
        num_action_classes,
        num_heads,
        depth,
        mlp_ratio=4.0,
        dropout=0.0,
        use_activation_checkpointing=True,
    ):
        super().__init__()
        AttentivePooler = _get_vjepa2_attentive_pooler()
        self.pooler = AttentivePooler(
            num_queries=3,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
            use_activation_checkpointing=use_activation_checkpointing,
        )
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.verb_classifier = nn.Linear(embed_dim, num_verb_classes, bias=True)
        self.noun_classifier = nn.Linear(embed_dim, num_noun_classes, bias=True)
        self.action_classifier = nn.Linear(embed_dim, num_action_classes, bias=True)

    def forward(self, x):
        x = self.pooler(x)
        x_verb, x_noun, x_action = x[:, 0, :], x[:, 1, :], x[:, 2, :]
        x_verb = self.verb_classifier(self.dropout(x_verb))
        x_noun = self.noun_classifier(self.dropout(x_noun))
        x_action = self.action_classifier(self.dropout(x_action))
        return dict(verb=x_verb, noun=x_noun, action=x_action)


class VJEPA2ProbeClassifier(nn.Module):
    """Frozen V-JEPA2 encoder with an attentive probe head."""

    def __init__(
        self,
        variant_key,
        num_classes,
        probe_num_heads=16,
        probe_num_blocks=4,
        probe_mlp_ratio=4.0,
        probe_dropout=0.0,
        use_activation_checkpointing=True,
        freeze_encoder=True,
    ):
        super().__init__()
        self.encoder = _load_vjepa2_encoder(variant_key)
        self.num_features = self.encoder.embed_dim
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

        self.probe = VJEPA2ProbeHead(
            embed_dim=self.num_features,
            num_classes=num_classes,
            num_heads=probe_num_heads,
            depth=probe_num_blocks,
            mlp_ratio=probe_mlp_ratio,
            dropout=probe_dropout,
            use_activation_checkpointing=use_activation_checkpointing,
        )

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, x, use_checkpoint=False):
        if self.freeze_encoder:
            with torch.no_grad():
                features = self.encoder(x)
        else:
            features = self.encoder(x)
        return self.probe(features)


class VJEPA2MultiTaskProbeClassifier(nn.Module):
    """Frozen V-JEPA2 encoder with verb/noun/action attentive probes."""

    def __init__(
        self,
        variant_key,
        num_verb_classes,
        num_noun_classes,
        num_action_classes,
        probe_num_heads=16,
        probe_num_blocks=4,
        probe_mlp_ratio=4.0,
        probe_dropout=0.0,
        use_activation_checkpointing=True,
        freeze_encoder=True,
    ):
        super().__init__()
        self.encoder = _load_vjepa2_encoder(variant_key)
        self.num_features = self.encoder.embed_dim
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

        self.probe = VJEPA2MultiTaskProbeHead(
            embed_dim=self.num_features,
            num_verb_classes=num_verb_classes,
            num_noun_classes=num_noun_classes,
            num_action_classes=num_action_classes,
            num_heads=probe_num_heads,
            depth=probe_num_blocks,
            mlp_ratio=probe_mlp_ratio,
            dropout=probe_dropout,
            use_activation_checkpointing=use_activation_checkpointing,
        )

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, x, use_checkpoint=False):
        if self.freeze_encoder:
            with torch.no_grad():
                features = self.encoder(x)
        else:
            features = self.encoder(x)
        return self.probe(features)


class MViT_Spatial(nn.Module):
    """MViT model for spatial (RGB) input"""
    def __init__(self, num_classes, dropout=0.5):
        super().__init__()
        # Load pretrained MViT model
        weights = MViT_V2_S_Weights.DEFAULT
        self.mvit = mvit_v2_s(weights=weights)
        
        # Get the feature dimension
        feature_dim = self.mvit.head[1].in_features
        self.num_features = feature_dim
        
        # Replace classification head
        self.mvit.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes)
        )
        
    def forward(self, x):
        # x shape: [batch_size, 3, num_frames, height, width]
        return self.mvit(x)


class MViT_Temporal(nn.Module):
    """MViT model for temporal (optical flow) input"""
    def __init__(self, num_classes, dropout=0.5):
        super().__init__()
        weights = MViT_V2_S_Weights.DEFAULT
        mvit_model = mvit_v2_s(weights=weights)
        
        # Get the feature dimension
        feature_dim = mvit_model.head[1].in_features
        self.num_features = feature_dim
        
        # Modify first conv layer for 2-channel optical flow input
        original_conv = mvit_model.conv_proj
        new_conv = nn.Conv3d(
            in_channels=2,  # Optical flow has 2 channels
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )
        
        # Initialize the new conv layer
        nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        if new_conv.bias is not None:
            nn.init.constant_(new_conv.bias, 0)
        
        mvit_model.conv_proj = new_conv
        
        # Replace classification head
        mvit_model.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes)
        )
        
        self.mvit = mvit_model
    
    def forward(self, x):
        # x shape: [batch_size, 2, num_frames, height, width]
        return self.mvit(x)


def get_args_parser():
    parser = argparse.ArgumentParser(description='Clean MViT training with LaViLa', add_help=False)
    
    # Data
    parser.add_argument('--dataset', default='egtea', type=str,
                        choices=['ek100_cls', 'egtea'])
    parser.add_argument('--ek100-train-csv',
                        default='/home/dz/Projects/multi-modal_AR/data/EK/data/EPIC_100_train.csv',
                        type=str, help='path to EK100 train csv for label map')
    parser.add_argument('--ek100-val-csv',
                        default='/home/dz/Projects/multi-modal_AR/data/EK/data/EPIC_100_validation.csv',
                        type=str, help='path to EK100 val csv for label map')
    parser.add_argument('--egtea-idx-root',
                        default='../data/EGTEA/raw/annotation/idx',
                        type=str, help='root dir containing egtea idx txt files')
    parser.add_argument('--charades-classlist',
                        default='datasets/CharadesEgo/CharadesEgo/Charades_v1_classes.txt',
                        type=str, help='class list txt for CharadesEgo')
    parser.add_argument('--root',
                        default='/mnt/j/video_clips/cropped_clips/',
                        type=str, help='path to dataset root')
    parser.add_argument('--metadata-train',
                        default='../data/EGTEA/raw/annotation/split/train_split1.txt',
                        type=str, help='path to metadata file (train set)')
    parser.add_argument('--metadata-val',
                        default='../data/EGTEA/raw/annotation/split/test_split1.txt',
                        type=str, help='path to metadata file (val set)')
    parser.add_argument('--output-dir', default='./checkpoints', type=str, help='output dir')
    parser.add_argument('--num-crops', default=1, type=int, help='number of crops for val')
    parser.add_argument('--num-clips', default=1, type=int, help='number of clips for val')
    parser.add_argument('--clip-length', default=16, type=int, help='clip length')
    parser.add_argument('--clip-stride', default=2, type=int, help='clip stride')
    parser.add_argument('--sparse-sample', action='store_true', help='switch to sparse sampling')
    
    # Model
    parser.add_argument('--model-type', default='mvit_temporal', type=str,
                        choices=['mvit_spatial', 'mvit_temporal'] + list(VJEPA2_MODEL_SPECS.keys()),
                        help='type of MViT model to use')
    parser.add_argument('--pretrain-model', default='', type=str, help='path to pretrain model')
    parser.add_argument('--resume', default='', type=str, help='path to resume from')
    parser.add_argument('--dropout-ratio', default=0.5, type=float, help='dropout ratio')
    parser.add_argument('--vjepa2-head', default='attentive', type=str,
                        choices=['attentive', 'meanpool'],
                        help='classification head for V-JEPA2 models')
    parser.add_argument('--probe-num-heads', default=16, type=int,
                        help='attention heads in attentive probe (V-JEPA2)')
    parser.add_argument('--probe-num-blocks', default=4, type=int,
                        help='attentive probe depth (V-JEPA2)')
    parser.add_argument('--probe-mlp-ratio', default=4.0, type=float,
                        help='attentive probe MLP ratio (V-JEPA2)')
    parser.add_argument('--probe-dropout', default=0.0, type=float,
                        help='dropout before classifier in probe head (V-JEPA2)')
    parser.add_argument('--probe-use-activation-checkpointing', action='store_true',
                        help='enable activation checkpointing in attentive probe')
    parser.add_argument('--unfreeze-encoder', action='store_true',
                        help='train V-JEPA2 encoder weights instead of freezing')
    parser.add_argument('--multi-task', action='store_true',
                        help='train verb/noun/action jointly (EK100 only)')
    parser.add_argument('--num-classes', default=106, type=int, help='number of classes')
    parser.add_argument('--task-type', default='action', type=str,
                        choices=['action', 'verb', 'noun'],
                        help='classification task type')
    
    # Training
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--warmup-epochs', default=5, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=8, type=int,
                        help='number of samples per-device/per-gpu')
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--lr-start', default=1e-6, type=float, help='initial warmup lr')
    parser.add_argument('--lr-end', default=1e-5, type=float, help='minimum final lr')
    parser.add_argument('--wd', default=0.01, type=float, help='weight decay')
    parser.add_argument('--betas', default=(0.9, 0.999), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--label-smoothing', default=0.1, type=float)
    parser.add_argument('--class-weight', default='none', type=str,
                        choices=['none', 'balanced'],
                        help='use class-balanced weights for cross-entropy')
    parser.add_argument('--verb-loss-weight', default=1.0, type=float,
                        help='loss weight for verb head (multi-task)')
    parser.add_argument('--noun-loss-weight', default=1.0, type=float,
                        help='loss weight for noun head (multi-task)')
    parser.add_argument('--action-loss-weight', default=1.0, type=float,
                        help='loss weight for action head (multi-task)')
    parser.add_argument('--clip-grad-value', default=1.0, type=float, help='gradient clipping')
    parser.add_argument('--update-freq', default=1, type=int,
                        help='gradient accumulation steps')
    parser.add_argument('--use-sgd', action='store_true', help='use SGD instead of AdamW')
    parser.add_argument('--disable-amp', action='store_true',
                        help='disable mixed-precision training')
    
    # Evaluation
    parser.add_argument('--eval-freq', default=5, type=int)
    parser.add_argument('--save-freq', default=5, type=int)
    
    # System
    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
    parser.add_argument('--workers', default=4, type=int, help='data loading workers')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')
    parser.add_argument('--find-unused-parameters', action='store_true')
    
    # Data loading options
    parser.add_argument('--use-timestamps', action='store_true',
                        help='Use timestamps instead of frame numbers for EK100 (original LaViLa approach)')
    
    # Distributed
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    
    return parser


def load_checkpoint(checkpoint_path):
    """Load checkpoint from either .pt/.pth or .safetensors file"""
    if checkpoint_path.endswith('.safetensors'):
        if not HAS_SAFETENSORS:
            raise ImportError("safetensors library required. Install with 'pip install safetensors'")
        print(f"Loading .safetensors file: {checkpoint_path}")
        state_dict = load_safetensors(checkpoint_path)
        return {'state_dict': state_dict, 'epoch': 0, 'best_acc1': 0.0}
    else:
        print(f"Loading PyTorch checkpoint: {checkpoint_path}")
        return torch.load(checkpoint_path, map_location='cpu', weights_only=False)


def save_checkpoint(state, is_best, output_dir):
    """Save training checkpoint"""
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, 'checkpoint.pt')
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(output_dir, 'checkpoint_best.pt')
        torch.save(state, best_path)


def compute_class_weights(args, train_dataset, label_mapping):
    counts = np.zeros(args.num_classes, dtype=np.int64)
    if args.dataset == 'ek100_cls':
        for sample in train_dataset.samples:
            verb = sample[4]
            noun = sample[5]
            if args.task_type == 'verb':
                label_key = str(verb)
            elif args.task_type == 'noun':
                label_key = str(noun)
            else:
                label_key = f"{verb}:{noun}"
            idx = label_mapping.get(label_key)
            if idx is not None:
                counts[idx] += 1
    elif args.dataset == 'egtea':
        for sample in train_dataset.samples:
            label_key = sample[3]
            idx = label_mapping.get(label_key)
            if idx is not None:
                counts[idx] += 1
    else:
        raise ValueError(f'Unsupported dataset for class weights: {args.dataset}')

    counts = np.maximum(counts, 1)
    weights = counts.sum() / (len(counts) * counts.astype(np.float32))
    return torch.tensor(weights, dtype=torch.float32)


def build_ek100_multitask_label_maps(train_csv):
    verb_set = set()
    noun_set = set()
    action_set = set()

    with open(train_csv) as f:
        csv_reader = csv.reader(f)
        _ = next(csv_reader)
        for row in csv_reader:
            verb = int(row[10])
            noun = int(row[12])
            verb_set.add(str(verb))
            noun_set.add(str(noun))
            action_set.add(f"{verb}:{noun}")

    verb_list = sorted(verb_set, key=int)
    noun_list = sorted(noun_set, key=int)
    action_list = sorted(action_set, key=lambda vn: (int(vn.split(":")[0]), int(vn.split(":")[1])))

    verb_map = {v: i for i, v in enumerate(verb_list)}
    noun_map = {n: i for i, n in enumerate(noun_list)}
    action_map = {a: i for i, a in enumerate(action_list)}

    return dict(verb=verb_map, noun=noun_map, action=action_map)


def compute_multitask_class_weights(train_dataset, label_maps):
    counts = {
        "verb": np.zeros(len(label_maps["verb"]), dtype=np.int64),
        "noun": np.zeros(len(label_maps["noun"]), dtype=np.int64),
        "action": np.zeros(len(label_maps["action"]), dtype=np.int64),
    }

    for sample in train_dataset.samples:
        if len(sample) < 6:
            raise ValueError("Expected EK100 samples to include verb and noun labels.")
        verb = str(sample[4])
        noun = str(sample[5])
        action = f"{verb}:{noun}"
        counts["verb"][label_maps["verb"][verb]] += 1
        counts["noun"][label_maps["noun"][noun]] += 1
        counts["action"][label_maps["action"][action]] += 1

    weights = {}
    for key, count in counts.items():
        count = np.maximum(count, 1)
        weight = count.sum() / (len(count) * count.astype(np.float32))
        weights[key] = torch.tensor(weight, dtype=torch.float32)

    return weights


class EK100MultiTaskDataset(VideoCaptionDatasetBase):
    def __init__(
        self,
        args,
        root,
        metadata,
        transform=None,
        is_training=True,
        label_maps=None,
        filter_actions=False,
        clip_length=32,
        clip_stride=2,
        sparse_sample=False,
    ):
        super().__init__(args, "ek100_cls", root, metadata)
        self.transform = transform
        self.is_training = is_training
        self.label_maps = label_maps
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.sparse_sample = sparse_sample

        if filter_actions and self.label_maps is not None:
            before = len(self.samples)
            filtered = []
            for sample in self.samples:
                verb = str(sample[4])
                noun = str(sample[5])
                if f"{verb}:{noun}" in self.label_maps["action"]:
                    filtered.append(sample)
            self.samples = filtered
            dropped = before - len(self.samples)
            if dropped > 0:
                print(f"=> Filtered {dropped} validation samples with unseen actions")

    def __getitem__(self, i):
        vid_path, start_frame, end_frame, _, verb, noun = self.samples[i]
        frame_ids = get_frame_ids(start_frame, end_frame, num_segments=self.clip_length, jitter=self.is_training)
        frames = video_loader_by_frames(self.root, vid_path, frame_ids)

        if self.transform is not None:
            frames = self.transform(frames)

        verb_key = str(verb)
        noun_key = str(noun)
        action_key = f"{verb_key}:{noun_key}"
        verb_label = self.label_maps["verb"][verb_key]
        noun_label = self.label_maps["noun"][noun_key]
        action_label = self.label_maps["action"][action_key]

        return frames, verb_label, noun_label, action_label


def _average_multitask_logits(logits_list):
    if not logits_list:
        raise ValueError("Expected non-empty logits list for multi-task averaging.")
    keys = logits_list[0].keys()
    return {k: torch.mean(torch.stack([o[k] for o in logits_list]), dim=0) for k in keys}


def train(train_loader, model, flow_model, criterion, optimizer, scaler, epoch, lr_schedule, args):
    """Training function"""
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')

    if args.multi_task:
        losses = AverageMeter('Loss', ':.4e')
        top1_action = AverageMeter('Acc@1-A', ':6.2f')
        top1_verb = AverageMeter('Acc@1-V', ':6.2f')
        top1_noun = AverageMeter('Acc@1-N', ':6.2f')
        top5_action = AverageMeter('Acc@5-A', ':6.2f')
        top5_verb = AverageMeter('Acc@5-V', ':6.2f')
        top5_noun = AverageMeter('Acc@5-N', ':6.2f')
        meters = [batch_time, data_time, losses, top1_action, top1_verb, top1_noun, top5_action, top5_verb, top5_noun, mem]
    else:
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        meters = [batch_time, data_time, losses, top1, top5, mem]

    iters_per_epoch = len(train_loader)
    progress = ProgressMeter(iters_per_epoch, meters, prefix="Epoch: [{}]".format(epoch))
    
    # Switch to train mode
    model.train()
    
    end = time.time()
    for data_iter, batch_data in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        
        # Handle different model types
        if args.model_type == 'mvit_temporal':
            if args.multi_task:
                raise ValueError("Multi-task training is not supported for temporal flow models.")
            images, images_flow, target = batch_data
            batch_size = images.shape[0]
            num_frames = args.clip_length
            
            images = images.cuda(args.gpu, non_blocking=True)
            images_flow = images_flow.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            
            # Compute optical flow
            flow_list = []
            with torch.no_grad():
                with torch.backends.cudnn.flags(enabled=False):
                    for b in range(batch_size):
                        batch_flows = []
                        for t in range(num_frames):
                            frame1 = images[b:b+1, :, t, :, :].contiguous()
                            frame2 = images_flow[b:b+1, :, t, :, :].contiguous()
                            flow_out = flow_model(frame1, frame2)
                            flow_frame = flow_out[-1]
                            batch_flows.append(flow_frame)
                        batch_flows = torch.cat(batch_flows, dim=0)
                        flow_list.append(batch_flows.unsqueeze(0))
            
            flow_sequence = torch.cat(flow_list, dim=0)
            
            # Normalize flow
            flow_normalize = transforms.Normalize(mean=[0, 0], std=[20, 20])
            flow_normalized = flow_normalize(flow_sequence.view(-1, 2, 224, 224)).view(
                batch_size, num_frames, 2, 224, 224)
            
            # Permute to [B, 2, T, H, W]
            model_input = flow_normalized.permute(0, 2, 1, 3, 4)
            
        else:  # RGB models (MViT spatial or V-JEPA2)
            if args.multi_task:
                images, verb_target, noun_target, action_target = batch_data
                images = images.cuda(args.gpu, non_blocking=True)
                verb_target = verb_target.cuda(args.gpu, non_blocking=True)
                noun_target = noun_target.cuda(args.gpu, non_blocking=True)
                action_target = action_target.cuda(args.gpu, non_blocking=True)
            else:
                images, target = batch_data
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
            model_input = images
        
        # Forward pass
        with amp.autocast(enabled=not args.disable_amp):
            output = model(model_input)
            if args.multi_task:
                loss_verb = criterion["verb"](output["verb"], verb_target)
                loss_noun = criterion["noun"](output["noun"], noun_target)
                loss_action = criterion["action"](output["action"], action_target)
                loss = (
                    args.verb_loss_weight * loss_verb
                    + args.noun_loss_weight * loss_noun
                    + args.action_loss_weight * loss_action
                )
            else:
                loss = criterion(output, target)
        
        # Measure accuracy and record loss
        if args.multi_task:
            acc1_action, acc5_action = accuracy(output["action"], action_target, topk=(1, 5))
            acc1_verb, acc5_verb = accuracy(output["verb"], verb_target, topk=(1, 5))
            acc1_noun, acc5_noun = accuracy(output["noun"], noun_target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1_action.update(acc1_action.item(), images.size(0))
            top1_verb.update(acc1_verb.item(), images.size(0))
            top1_noun.update(acc1_noun.item(), images.size(0))
            top5_action.update(acc5_action.item(), images.size(0))
            top5_verb.update(acc5_verb.item(), images.size(0))
            top5_noun.update(acc5_noun.item(), images.size(0))
        else:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
        
        # Compute gradient and do optimizer step
        loss = loss / args.update_freq
        scaler.scale(loss).backward()
        
        if (data_iter + 1) % args.update_freq == 0:
            # Gradient clipping
            if args.clip_grad_value is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_value)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Update learning rate
            if lr_schedule is not None:
                step = epoch * (iters_per_epoch // args.update_freq) + data_iter // args.update_freq
                if step < len(lr_schedule):
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_schedule[step]
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        mem.update(torch.cuda.max_memory_allocated() // 1e9)
        end = time.time()
        
        if data_iter % args.print_freq == 0:
            progress.display(data_iter)
    
    if args.multi_task:
        return {
            'loss': losses.avg,
            'acc1_action': top1_action.avg,
            'acc1_verb': top1_verb.avg,
            'acc1_noun': top1_noun.avg,
            'acc5_action': top5_action.avg,
            'acc5_verb': top5_verb.avg,
            'acc5_noun': top5_noun.avg,
            'lr': optimizer.param_groups[0]['lr'],
        }
    return {'loss': losses.avg, 'acc1': top1.avg, 'acc5': top5.avg,
            'lr': optimizer.param_groups[0]['lr']}


def validate(val_loader, model, flow_model, args):
    """Validation function"""
    batch_time = AverageMeter('Time', ':6.2f')
    if args.multi_task and args.model_type == 'mvit_temporal':
        raise ValueError("Multi-task validation is not supported for temporal flow models.")
    if args.multi_task:
        top1_action = AverageMeter('Acc@1-A', ':6.2f')
        top1_verb = AverageMeter('Acc@1-V', ':6.2f')
        top1_noun = AverageMeter('Acc@1-N', ':6.2f')
        top5_action = AverageMeter('Acc@5-A', ':6.2f')
        top5_verb = AverageMeter('Acc@5-V', ':6.2f')
        top5_noun = AverageMeter('Acc@5-N', ':6.2f')
        meters = [batch_time, top1_action, top1_verb, top1_noun, top5_action, top5_verb, top5_noun]
    else:
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        meters = [batch_time, top1, top5]
    progress = ProgressMeter(len(val_loader), meters, prefix='Test: ')
    
    # Switch to evaluate mode
    model.eval()
    
    all_outputs = []
    all_targets = []
    all_outputs_verb = []
    all_outputs_noun = []
    all_targets_verb = []
    all_targets_noun = []
    
    with torch.no_grad():
        end = time.time()
        for i, batch_data in enumerate(val_loader):
            # Handle different model types
            if args.model_type == 'mvit_temporal':
                images, images_flow, target = batch_data
                
                # Handle list of crops
                if isinstance(images, list):
                    logit_allcrops = []
                    for crop, crop_flow in zip(images, images_flow):
                        crop = crop.cuda(args.gpu, non_blocking=True)
                        crop_flow = crop_flow.cuda(args.gpu, non_blocking=True)
                        
                        batch_size = crop.shape[0]
                        num_frames = args.clip_length
                        
                        # Compute optical flow
                        flow_list = []
                        with torch.backends.cudnn.flags(enabled=False):
                            for b in range(batch_size):
                                batch_flows = []
                                for t in range(num_frames):
                                    frame1 = crop[b:b+1, :, t, :, :].contiguous()
                                    frame2 = crop_flow[b:b+1, :, t, :, :].contiguous()
                                    flow_out = flow_model(frame1, frame2)
                                    flow_frame = flow_out[-1]
                                    batch_flows.append(flow_frame)
                                batch_flows = torch.cat(batch_flows, dim=0)
                                flow_list.append(batch_flows.unsqueeze(0))
                        
                        flow_sequence = torch.cat(flow_list, dim=0)
                        flow_normalize = transforms.Normalize(mean=[0, 0], std=[20, 20])
                        flow_normalized = flow_normalize(flow_sequence.view(-1, 2, 224, 224)).view(
                            batch_size, num_frames, 2, 224, 224)
                        flow_input = flow_normalized.permute(0, 2, 1, 3, 4)
                        
                        logit = model(flow_input)
                        logit_allcrops.append(logit)
                    
                    # Average predictions across crops
                    output = torch.mean(torch.stack(logit_allcrops), dim=0)
                else:
                    # Single crop
                    images = images.cuda(args.gpu, non_blocking=True)
                    images_flow = images_flow.cuda(args.gpu, non_blocking=True)
                    
                    batch_size = images.shape[0]
                    num_frames = args.clip_length
                    
                    # Compute optical flow
                    flow_list = []
                    with torch.backends.cudnn.flags(enabled=False):
                        for b in range(batch_size):
                            batch_flows = []
                            for t in range(num_frames):
                                frame1 = images[b:b+1, :, t, :, :].contiguous()
                                frame2 = images_flow[b:b+1, :, t, :, :].contiguous()
                                flow_out = flow_model(frame1, frame2)
                                flow_frame = flow_out[-1]
                                batch_flows.append(flow_frame)
                            batch_flows = torch.cat(batch_flows, dim=0)
                            flow_list.append(batch_flows.unsqueeze(0))
                    
                    flow_sequence = torch.cat(flow_list, dim=0)
                    flow_normalize = transforms.Normalize(mean=[0, 0], std=[20, 20])
                    flow_normalized = flow_normalize(flow_sequence.view(-1, 2, 224, 224)).view(
                        batch_size, num_frames, 2, 224, 224)
                    flow_input = flow_normalized.permute(0, 2, 1, 3, 4)
                    
                    output = model(flow_input)
                
            else:  # RGB models (MViT spatial or V-JEPA2)
                if args.multi_task:
                    images, verb_target, noun_target, action_target = batch_data
                else:
                    images, target = batch_data

                if isinstance(images, list):
                    # Multiple crops
                    logit_allcrops = []
                    for crop in images:
                        crop = crop.cuda(args.gpu, non_blocking=True)
                        logit = model(crop)
                        logit_allcrops.append(logit)
                    if args.multi_task:
                        output = _average_multitask_logits(logit_allcrops)
                    else:
                        output = torch.mean(torch.stack(logit_allcrops), dim=0)
                else:
                    # Single crop
                    images = images.cuda(args.gpu, non_blocking=True)
                    output = model(images)

            if args.multi_task:
                verb_target = verb_target.cuda(args.gpu, non_blocking=True)
                noun_target = noun_target.cuda(args.gpu, non_blocking=True)
                action_target = action_target.cuda(args.gpu, non_blocking=True)

                all_outputs.append(output["action"].cpu())
                all_targets.append(action_target.cpu())
                all_outputs_verb.append(output["verb"].cpu())
                all_targets_verb.append(verb_target.cpu())
                all_outputs_noun.append(output["noun"].cpu())
                all_targets_noun.append(noun_target.cpu())

                acc1_action, acc5_action = accuracy(output["action"], action_target, topk=(1, 5))
                acc1_verb, acc5_verb = accuracy(output["verb"], verb_target, topk=(1, 5))
                acc1_noun, acc5_noun = accuracy(output["noun"], noun_target, topk=(1, 5))
                batch_size = action_target.size(0)
                top1_action.update(acc1_action.item(), batch_size)
                top1_verb.update(acc1_verb.item(), batch_size)
                top1_noun.update(acc1_noun.item(), batch_size)
                top5_action.update(acc5_action.item(), batch_size)
                top5_verb.update(acc5_verb.item(), batch_size)
                top5_noun.update(acc5_noun.item(), batch_size)
            else:
                target = target.cuda(args.gpu, non_blocking=True)

                # Store for confusion matrix
                all_outputs.append(output.cpu())
                all_targets.append(target.cpu())

                # Measure accuracy
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                batch_size = target.size(0)
                top1.update(acc1.item(), batch_size)
                top5.update(acc5.item(), batch_size)
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % args.print_freq == 0:
                progress.display(i)
    
    # Compute per-class accuracy
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    predictions = all_outputs.argmax(dim=1)
    cm = confusion_matrix(all_targets.numpy(), predictions.numpy())
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    mean_class_acc = np.mean(per_class_acc)

    if args.multi_task:
        all_outputs_verb = torch.cat(all_outputs_verb)
        all_targets_verb = torch.cat(all_targets_verb)
        pred_verb = all_outputs_verb.argmax(dim=1)
        cm_verb = confusion_matrix(all_targets_verb.numpy(), pred_verb.numpy())
        mean_class_acc_verb = np.mean(cm_verb.diagonal() / cm_verb.sum(axis=1))

        all_outputs_noun = torch.cat(all_outputs_noun)
        all_targets_noun = torch.cat(all_targets_noun)
        pred_noun = all_outputs_noun.argmax(dim=1)
        cm_noun = confusion_matrix(all_targets_noun.numpy(), pred_noun.numpy())
        mean_class_acc_noun = np.mean(cm_noun.diagonal() / cm_noun.sum(axis=1))

        print(
            " * Acc@1 (A/V/N) "
            f"{top1_action.avg:.3f} {top1_verb.avg:.3f} {top1_noun.avg:.3f} "
            "Acc@5 (A/V/N) "
            f"{top5_action.avg:.3f} {top5_verb.avg:.3f} {top5_noun.avg:.3f} "
            "Mean Class Acc (A/V/N) "
            f"{mean_class_acc:.3f} {mean_class_acc_verb:.3f} {mean_class_acc_noun:.3f}"
        )

        return {
            'acc1_action': top1_action.avg,
            'acc1_verb': top1_verb.avg,
            'acc1_noun': top1_noun.avg,
            'acc5_action': top5_action.avg,
            'acc5_verb': top5_verb.avg,
            'acc5_noun': top5_noun.avg,
            'mean_class_acc_action': mean_class_acc,
            'mean_class_acc_verb': mean_class_acc_verb,
            'mean_class_acc_noun': mean_class_acc_noun,
        }

    print(f' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Mean Class Acc {mean_class_acc:.3f}')

    return top1.avg


def main(args):
    # Initialize distributed training
    dist_utils.init_distributed_mode(args)
    
    # Set random seed
    random_seed(args.seed, dist_utils.get_rank())
    print(f'Random seed: {args.seed}')
    
    # Create output directory
    if dist_utils.is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
    
    label_maps = None
    if args.multi_task:
        if args.dataset != 'ek100_cls':
            raise ValueError("Multi-task training is only supported for EK100.")
        if args.model_type not in VJEPA2_MODEL_SPECS:
            raise ValueError("Multi-task training is only supported for V-JEPA2 models.")
        if args.vjepa2_head != 'attentive':
            raise ValueError("Multi-task training requires the attentive V-JEPA2 head.")
        label_maps = build_ek100_multitask_label_maps(args.ek100_train_csv)
        args.num_classes_action = len(label_maps["action"])
        args.num_classes_verb = len(label_maps["verb"])
        args.num_classes_noun = len(label_maps["noun"])
        args.num_classes = args.num_classes_action
    else:
        # Set number of classes based on dataset and task type
        if args.dataset == 'egtea':
            if args.task_type == 'action':
                args.num_classes = 106
            elif args.task_type == 'verb':
                args.num_classes = 19
            elif args.task_type == 'noun':
                args.num_classes = 53
        elif args.dataset == 'ek100_cls':
            if args.task_type == 'action':
                args.num_classes = 3806
            elif args.task_type == 'verb':
                args.num_classes = 97
            elif args.task_type == 'noun':
                args.num_classes = 300
    
    # Set task-specific arguments for compatibility with LaViLa datasets
    # The egtea_finetune_type is used by the dataset to determine which labels to return
    if args.dataset == 'egtea':
        args.egtea_finetune_type = args.task_type
    elif args.dataset == 'ek100_cls':
        # For EK100, we also use egtea_finetune_type for consistency with the dataset code
        args.egtea_finetune_type = 'action' if args.multi_task else args.task_type
    
    print(f"=> Creating model: {args.model_type}")
    if args.multi_task:
        print(
            "=> Multi-task classes (action/verb/noun): "
            f"{args.num_classes_action}/{args.num_classes_verb}/{args.num_classes_noun}"
        )
    else:
        print(f"=> Number of classes: {args.num_classes}")
    
    # Create model
    if args.model_type == 'mvit_spatial':
        model = MViT_Spatial(args.num_classes, dropout=args.dropout_ratio)
    elif args.model_type == 'mvit_temporal':
        model = MViT_Temporal(args.num_classes, dropout=args.dropout_ratio)
    elif args.model_type in VJEPA2_MODEL_SPECS:
        if args.multi_task:
            model = VJEPA2MultiTaskProbeClassifier(
                args.model_type,
                num_verb_classes=args.num_classes_verb,
                num_noun_classes=args.num_classes_noun,
                num_action_classes=args.num_classes_action,
                probe_num_heads=args.probe_num_heads,
                probe_num_blocks=args.probe_num_blocks,
                probe_mlp_ratio=args.probe_mlp_ratio,
                probe_dropout=args.probe_dropout,
                use_activation_checkpointing=args.probe_use_activation_checkpointing,
                freeze_encoder=not args.unfreeze_encoder,
            )
        elif args.vjepa2_head == 'meanpool':
            model = VJEPA2MeanPoolClassifier(args.model_type, args.num_classes, dropout=args.dropout_ratio)
        else:
            model = VJEPA2ProbeClassifier(
                args.model_type,
                args.num_classes,
                probe_num_heads=args.probe_num_heads,
                probe_num_blocks=args.probe_num_blocks,
                probe_mlp_ratio=args.probe_mlp_ratio,
                probe_dropout=args.probe_dropout,
                use_activation_checkpointing=args.probe_use_activation_checkpointing,
                freeze_encoder=not args.unfreeze_encoder,
            )
    else:
        raise ValueError(f'Unknown model type: {args.model_type}')
    
    # Load pretrained weights if provided
    if args.pretrain_model:
        print(f"=> Loading pretrained model: {args.pretrain_model}")
        checkpoint = load_checkpoint(args.pretrain_model)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        
        # Clean up state dict keys
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            k = k.replace('module.', '')
            new_state_dict[k] = v
        
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
    
    model.cuda(args.gpu)
    
    # Initialize RAFT model for temporal mode
    flow_model = None
    if args.model_type == 'mvit_temporal':
        flow_model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).cuda(args.gpu)
        flow_model.eval()
        print("=> Initialized RAFT model for optical flow computation")
    
    # Setup distributed model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], bucket_cap_mb=200,
            find_unused_parameters=args.find_unused_parameters
        )
    
    # Create optimizer
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise RuntimeError("No trainable parameters found. Check encoder freeze settings.")
    if args.use_sgd:
        optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=args.betas[0], weight_decay=args.wd)
    else:
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, betas=args.betas, eps=args.eps, weight_decay=args.wd)
    
    # Create gradient scaler for mixed precision
    scaler = amp.GradScaler(enabled=not args.disable_amp)
    
    # Resume from checkpoint
    best_acc1 = 0.
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> Resuming from checkpoint: {args.resume}")
            checkpoint = load_checkpoint(args.resume)
            args.start_epoch = checkpoint['epoch']
            state_dict = checkpoint['state_dict']
            
            # Handle DDP state dict
            if not args.distributed:
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    k = k.replace('module.', '')
                    new_state_dict[k] = v
                model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(state_dict)
            
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            best_acc1 = checkpoint.get('best_acc1', 0.)
            print(f"=> Loaded checkpoint (epoch {checkpoint['epoch']})")
        else:
            print(f"=> No checkpoint found at '{args.resume}'")
    
    # Data loading
    cudnn.benchmark = True
    
    # Data transforms
    default_crop_size = 224
    crop_size = VJEPA2_MODEL_SPECS.get(args.model_type, {}).get('crop_size', default_crop_size)
    if args.model_type == 'mvit_temporal':
        # For flow, don't normalize RGB - RAFT needs raw pixel values
        train_transform = transforms.Compose([
            Permute([3, 0, 1, 2]),  # T H W C -> C T H W
            transforms.RandomResizedCrop(crop_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
        val_transform = transforms.Compose([
            Permute([3, 0, 1, 2]),  # T H W C -> C T H W
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            TemporalCrop(frames_per_clip=args.clip_length, stride=args.clip_length),
            SpatialCrop(crop_size=crop_size, num_crops=args.num_crops),
        ])
    else:
        # RGB transforms with normalization (used by both MViT spatial and V-JEPA2 backbones)
        train_transform = transforms.Compose([
            Permute([3, 0, 1, 2]),  # T H W C -> C T H W
            transforms.RandomResizedCrop(crop_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms_video.NormalizeVideo(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
        ])
        val_transform = transforms.Compose([
            Permute([3, 0, 1, 2]),  # T H W C -> C T H W
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms_video.NormalizeVideo(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
            TemporalCrop(frames_per_clip=args.clip_length, stride=args.clip_length),
            SpatialCrop(crop_size=crop_size, num_crops=args.num_crops),
        ])
    
    # Build datasets
    tokenizer = SimpleTokenizer()  # Required by dataset but not used for classification
    mapping_vn2act = None
    if not args.multi_task:
        _, mapping_vn2act = generate_label_map(args.dataset, args)

        # Debug label mapping for EK100
        if args.dataset == 'ek100_cls':
            print(f"Label mapping for {args.task_type}: {len(mapping_vn2act)} classes")
            if args.task_type == 'verb':
                # Show some sample mappings to verify
                sample_items = list(mapping_vn2act.items())[:10]
                print(f"First 10 verb mappings: {sample_items}")
                # Verify mapping values are 0-based
                mapping_values = list(mapping_vn2act.values())
                print(f"Mapping value range: {min(mapping_values)} to {max(mapping_values)}")
    
    # Store original num_clips for validation
    num_clips_at_val = args.num_clips
    args.num_clips = 1  # Single clip for training
    
    if args.multi_task:
        if args.model_type == 'mvit_temporal':
            raise ValueError("Multi-task training is not supported for temporal flow datasets.")
        train_dataset = EK100MultiTaskDataset(
            args,
            args.root,
            args.metadata_train,
            transform=train_transform,
            is_training=True,
            label_maps=label_maps,
            filter_actions=False,
            clip_length=args.clip_length,
            clip_stride=args.clip_stride,
            sparse_sample=args.sparse_sample,
        )
    elif args.model_type == 'mvit_temporal':
        # Use flow dataset that returns (images, images_flow, target)
        train_dataset = datasets_flow.get_downstream_dataset(
            train_transform, tokenizer, args, subset='train', label_mapping=mapping_vn2act,
        )
    else:
        # Use standard dataset for RGB
        train_dataset = datasets.get_downstream_dataset(
            train_transform, tokenizer, args, subset='train', label_mapping=mapping_vn2act,
        )
    
    # Restore num_clips for validation
    args.num_clips = num_clips_at_val
    
    if args.multi_task:
        val_dataset = EK100MultiTaskDataset(
            args,
            args.root,
            args.metadata_val,
            transform=val_transform,
            is_training=False,
            label_maps=label_maps,
            filter_actions=True,
            clip_length=args.clip_length,
            clip_stride=args.clip_stride,
            sparse_sample=args.sparse_sample,
        )
    elif args.model_type == 'mvit_temporal':
        val_dataset = datasets_flow.get_downstream_dataset(
            val_transform, tokenizer, args, subset='val', label_mapping=mapping_vn2act,
        )
    else:
        val_dataset = datasets.get_downstream_dataset(
            val_transform, tokenizer, args, subset='val', label_mapping=mapping_vn2act,
        )
    
    # Data loaders
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)  # Disable distributed for val
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False
    )
    
    print(f'Training samples: {len(train_dataset)}')
    print(f'Training batches: {len(train_loader)}')
    print(f'Validation samples: {len(val_dataset)}')
    print(f'Validation batches: {len(val_loader)}')

    # Create loss function (optionally class-balanced)
    if args.multi_task:
        if args.class_weight == 'balanced':
            class_weights = compute_multitask_class_weights(train_dataset, label_maps)
            if dist_utils.is_main_process():
                print('=> Using balanced class weights for multi-task cross-entropy')
            criterion = {
                "verb": nn.CrossEntropyLoss(
                    weight=class_weights["verb"].cuda(args.gpu),
                    label_smoothing=args.label_smoothing,
                ).cuda(args.gpu),
                "noun": nn.CrossEntropyLoss(
                    weight=class_weights["noun"].cuda(args.gpu),
                    label_smoothing=args.label_smoothing,
                ).cuda(args.gpu),
                "action": nn.CrossEntropyLoss(
                    weight=class_weights["action"].cuda(args.gpu),
                    label_smoothing=args.label_smoothing,
                ).cuda(args.gpu),
            }
        else:
            criterion = {
                "verb": nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).cuda(args.gpu),
                "noun": nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).cuda(args.gpu),
                "action": nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).cuda(args.gpu),
            }
    else:
        if args.class_weight == 'balanced':
            class_weights = compute_class_weights(args, train_dataset, mapping_vn2act)
            if dist_utils.is_main_process():
                print('=> Using balanced class weights for cross-entropy '
                      f'(min={class_weights.min().item():.4f}, max={class_weights.max().item():.4f})')
            class_weights = class_weights.cuda(args.gpu)
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing).cuda(args.gpu)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).cuda(args.gpu)
    
    # Learning rate schedule
    lr_schedule = cosine_scheduler(
        args.lr, args.lr_end, args.epochs, len(train_loader) // args.update_freq,
        warmup_epochs=args.warmup_epochs, start_warmup_value=args.lr_start,
    )
    
    # Initialize wandb
    if dist_utils.is_main_process() and args.wandb:
        wandb_id = os.path.split(args.output_dir)[-1]
        wandb.init(project='MViT-LaViLa', id=wandb_id, config=args, resume='allow')
    
    print("=> Starting training")
    print(args)
    
    # Training loop
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        # Train for one epoch
        train_stats = train(train_loader, model, flow_model, criterion, optimizer, scaler, 
                          epoch, lr_schedule, args)
        
        # Evaluate
        if (epoch + 1) % args.eval_freq == 0:
            if args.multi_task:
                val_stats = validate(val_loader, model, flow_model, args)
                val_acc1 = val_stats["acc1_action"]
            else:
                val_acc1 = validate(val_loader, model, flow_model, args)
                val_stats = {"acc1": val_acc1}
            
            # Remember best acc@1 and save checkpoint
            is_best = val_acc1 > best_acc1
            best_acc1 = max(val_acc1, best_acc1)
            
            if dist_utils.is_main_process():
                # Save checkpoint
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'best_acc1': best_acc1,
                    'args': args,
                }, is_best, args.output_dir)
                
                # Log to wandb
                if args.wandb:
                    wandb_payload = {
                        'epoch': epoch,
                        'train_loss': train_stats['loss'],
                        'best_acc1': best_acc1,
                        'lr': train_stats['lr'],
                    }
                    if args.multi_task:
                        wandb_payload.update({
                            'train_acc1_action': train_stats['acc1_action'],
                            'train_acc1_verb': train_stats['acc1_verb'],
                            'train_acc1_noun': train_stats['acc1_noun'],
                            'train_acc5_action': train_stats['acc5_action'],
                            'train_acc5_verb': train_stats['acc5_verb'],
                            'train_acc5_noun': train_stats['acc5_noun'],
                            'val_acc1_action': val_stats['acc1_action'],
                            'val_acc1_verb': val_stats['acc1_verb'],
                            'val_acc1_noun': val_stats['acc1_noun'],
                            'val_acc5_action': val_stats['acc5_action'],
                            'val_acc5_verb': val_stats['acc5_verb'],
                            'val_acc5_noun': val_stats['acc5_noun'],
                            'val_mean_class_acc_action': val_stats['mean_class_acc_action'],
                            'val_mean_class_acc_verb': val_stats['mean_class_acc_verb'],
                            'val_mean_class_acc_noun': val_stats['mean_class_acc_noun'],
                        })
                    else:
                        wandb_payload.update({
                            'train_acc1': train_stats['acc1'],
                            'train_acc5': train_stats['acc5'],
                            'val_acc1': val_acc1,
                        })
                    wandb.log(wandb_payload)
                
                # Log to file
                log_stats = {
                    'epoch': epoch,
                    **{f'train_{k}': v for k, v in train_stats.items()},
                    'best_acc1': best_acc1,
                    **{f'val_{k}': v for k, v in val_stats.items()},
                }
                with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                    f.write(json.dumps(log_stats) + '\n')
        
        # Save checkpoint every save_freq epochs
        elif (epoch + 1) % args.save_freq == 0:
            if dist_utils.is_main_process():
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'best_acc1': best_acc1,
                    'args': args,
                }, False, args.output_dir)
    
    print("=> Training completed")
    if args.multi_task:
        print(f"Best Acc@1 (action): {best_acc1:.3f}")
    else:
        print(f"Best Acc@1: {best_acc1:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Clean MViT training with LaViLa', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
