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
from lavila.data.video_transforms import Permute, SpatialCrop, TemporalCrop
from lavila.models.tokenizer import SimpleTokenizer
from lavila.models.model_flow import adapt_vjepa_for_flow
from lavila.utils import distributed as dist_utils
from lavila.utils.evaluation import accuracy
from lavila.utils.meter import AverageMeter, ProgressMeter
from lavila.utils.preprocess import generate_label_map
from lavila.utils.random import random_seed
from lavila.utils.scheduler import cosine_scheduler

from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights


VJEPA2_MODEL_SPECS = {
    'vjepa2_large': {'hub_name': 'vjepa2_vit_large', 'crop_size': 256},
    'vjepa2_huge': {'hub_name': 'vjepa2_vit_huge', 'crop_size': 256},
    'vjepa2_giant': {'hub_name': 'vjepa2_vit_giant', 'crop_size': 256},
    'vjepa2_giant_384': {'hub_name': 'vjepa2_vit_giant_384', 'crop_size': 384},
}


class VJEPA2Classifier(nn.Module):
    """Thin wrapper that turns a V-JEPA2 encoder into a classifier."""

    def __init__(self, variant_key, num_classes, dropout=0.5):
        super().__init__()
        if variant_key not in VJEPA2_MODEL_SPECS:
            raise ValueError(f'Unsupported V-JEPA2 variant: {variant_key}')

        hub_name = VJEPA2_MODEL_SPECS[variant_key]['hub_name']
        encoder, _ = torch.hub.load('facebookresearch/vjepa2', hub_name)

        self.encoder = encoder
        self.num_features = encoder.embed_dim
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.num_features, num_classes),
        )

    def forward(self, x, use_checkpoint=False):
        features = self.encoder(x)  # [B, N, C]
        pooled = features.mean(dim=1)
        return self.classifier(pooled)


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


def prepare_flow_input(flow_tensor):
    if flow_tensor.ndim == 4:
        flow_tensor = flow_tensor.unsqueeze(0)
    if flow_tensor.ndim == 5 and flow_tensor.shape[1] != 2 and flow_tensor.shape[2] == 2:
        flow_tensor = flow_tensor.permute(0, 2, 1, 3, 4)
    return flow_tensor


def train(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args):
    """Training function"""
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    use_flow = args.model_type == 'mvit_temporal' or args.model_type in VJEPA2_MODEL_SPECS
    
    iters_per_epoch = len(train_loader)
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, losses, top1, top5, mem],
        prefix="Epoch: [{}]".format(epoch))
    
    # Switch to train mode
    model.train()
    
    end = time.time()
    for data_iter, batch_data in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        
        # Handle different model types
        if use_flow:
            flow, target = batch_data
            target = target.cuda(args.gpu, non_blocking=True)
            if isinstance(flow, list):
                flow_inputs = []
                for crop in flow:
                    crop = crop.cuda(args.gpu, non_blocking=True)
                    flow_inputs.append(prepare_flow_input(crop))
            else:
                flow = flow.cuda(args.gpu, non_blocking=True)
                model_input = prepare_flow_input(flow)
        else:  # RGB models
            images, target = batch_data
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # Forward pass
        with amp.autocast(enabled=not args.disable_amp):
            if use_flow:
                if isinstance(flow, list):
                    logits = [model(inp) for inp in flow_inputs]
                    output = torch.mean(torch.stack(logits), dim=0)
                else:
                    output = model(model_input)
            else:
                output = model(images)
            loss = criterion(output, target)
        
        # Measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = target.size(0)
        losses.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)
        
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
    
    return {'loss': losses.avg, 'acc1': top1.avg, 'acc5': top5.avg,
            'lr': optimizer.param_groups[0]['lr']}


def validate(val_loader, model, args):
    """Validation function"""
    batch_time = AverageMeter('Time', ':6.2f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')
    use_flow = args.model_type == 'mvit_temporal' or args.model_type in VJEPA2_MODEL_SPECS
    
    # Switch to evaluate mode
    model.eval()
    
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        end = time.time()
        for i, batch_data in enumerate(val_loader):
            # Handle different model types
            if use_flow:
                flow, target = batch_data
                if isinstance(flow, list):
                    logit_allcrops = []
                    for crop in flow:
                        crop = crop.cuda(args.gpu, non_blocking=True)
                        flow_input = prepare_flow_input(crop)
                        logit = model(flow_input)
                        logit_allcrops.append(logit)
                    output = torch.mean(torch.stack(logit_allcrops), dim=0)
                else:
                    flow = flow.cuda(args.gpu, non_blocking=True)
                    flow_input = prepare_flow_input(flow)
                    output = model(flow_input)
            else:  # RGB models
                images, target = batch_data
                
                if isinstance(images, list):
                    # Multiple crops
                    logit_allcrops = []
                    for crop in images:
                        crop = crop.cuda(args.gpu, non_blocking=True)
                        logit = model(crop)
                        logit_allcrops.append(logit)
                    output = torch.mean(torch.stack(logit_allcrops), dim=0)
                else:
                    # Single crop
                    images = images.cuda(args.gpu, non_blocking=True)
                    output = model(images)
            
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
        args.egtea_finetune_type = args.task_type
    
    print(f"=> Creating model: {args.model_type}")
    print(f"=> Number of classes: {args.num_classes}")
    
    # Create model
    if args.model_type == 'mvit_spatial':
        model = MViT_Spatial(args.num_classes, dropout=args.dropout_ratio)
    elif args.model_type == 'mvit_temporal':
        model = MViT_Temporal(args.num_classes, dropout=args.dropout_ratio)
    elif args.model_type in VJEPA2_MODEL_SPECS:
        model = VJEPA2Classifier(args.model_type, args.num_classes, dropout=args.dropout_ratio)
    else:
        raise ValueError(f'Unknown model type: {args.model_type}')
    use_flow = args.model_type == 'mvit_temporal' or args.model_type in VJEPA2_MODEL_SPECS
    
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

        adapted_before_load = False
        if args.model_type in VJEPA2_MODEL_SPECS:
            patch_keys = [k for k in new_state_dict if k.endswith('patch_embed.proj.weight')]
            if patch_keys and new_state_dict[patch_keys[0]].shape[1] == 2:
                adapt_vjepa_for_flow(model)
                adapted_before_load = True
                print("=> Adapted V-JEPA patch embedding for 2-channel flow input (flow checkpoint)")

        model_state = model.state_dict()
        pruned_state_dict = OrderedDict()
        skipped_keys = []
        for key, value in new_state_dict.items():
            if key not in model_state:
                continue
            if model_state[key].shape != value.shape:
                skipped_keys.append((key, tuple(value.shape), tuple(model_state[key].shape)))
                continue
            pruned_state_dict[key] = value

        missing_keys, unexpected_keys = model.load_state_dict(pruned_state_dict, strict=False)
        if skipped_keys:
            print("Skipped loading keys with shape mismatches:")
            for key, loaded_shape, model_shape in skipped_keys:
                print(f"  {key}: checkpoint {loaded_shape} vs model {model_shape}")
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        if args.model_type in VJEPA2_MODEL_SPECS and not adapted_before_load:
            adapt_vjepa_for_flow(model)
            print("=> Adapted V-JEPA patch embedding for 2-channel flow input")
    elif args.model_type in VJEPA2_MODEL_SPECS:
        adapt_vjepa_for_flow(model)
        print("=> Adapted V-JEPA patch embedding for 2-channel flow input")
    
    model.cuda(args.gpu)
    
    # Setup distributed model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], bucket_cap_mb=200,
            find_unused_parameters=args.find_unused_parameters
        )
    
    # Create optimizer
    parameters = model.parameters()
    if args.use_sgd:
        optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=args.betas[0], weight_decay=args.wd)
    else:
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, betas=args.betas, eps=args.eps, weight_decay=args.wd)
    
    # Create gradient scaler for mixed precision
    scaler = amp.GradScaler(enabled=not args.disable_amp)
    
    # Create loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).cuda(args.gpu)
    
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
    if use_flow:
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
    
    if use_flow:
        # Use flow dataset that returns (flow, target)
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
    
    if use_flow:
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
        train_stats = train(train_loader, model, criterion, optimizer, scaler,
                          epoch, lr_schedule, args)
        
        # Evaluate
        if (epoch + 1) % args.eval_freq == 0:
            val_acc1 = validate(val_loader, model, args)
            
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
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': train_stats['loss'],
                        'train_acc1': train_stats['acc1'],
                        'train_acc5': train_stats['acc5'],
                        'val_acc1': val_acc1,
                        'best_acc1': best_acc1,
                        'lr': train_stats['lr'],
                    })
                
                # Log to file
                log_stats = {
                    'epoch': epoch,
                    **{f'train_{k}': v for k, v in train_stats.items()},
                    'val_acc1': val_acc1,
                    'best_acc1': best_acc1,
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
    print(f"Best Acc@1: {best_acc1:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Clean MViT training with LaViLa', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
