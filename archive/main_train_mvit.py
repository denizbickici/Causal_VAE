#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# MViT Training Script - Adapted from LaViLa's main_finetune_classification.py
# This script enables training of MViT models (spatial and temporal) using LaViLa's infrastructure

import argparse
from collections import OrderedDict
import json
import math
import numpy as np
import os
import pandas as pd
import sys
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.nn.parallel
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
from sklearn.metrics import confusion_matrix
import wandb
from accelerate import Accelerator

from lavila.data import datasets
from lavila.data import datasets_flow
from lavila.data.video_transforms import Permute, SpatialCrop, TemporalCrop
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from lavila.models import models
from lavila.models.tokenizer import SimpleTokenizer
from lavila.models.utils import inflate_positional_embeds
# Note: Using Accelerate for distributed training instead of dist_utils
from lavila.utils.evaluation import accuracy, get_mean_accuracy
from lavila.utils.meter import AverageMeter, ProgressMeter
from lavila.utils.preprocess import generate_label_map
from lavila.utils.random import random_seed
from lavila.utils.scheduler import cosine_scheduler
from lavila.utils.evaluation_ek100cls import get_marginal_indexes, marginalize


class MViT_Spatial(nn.Module):
    """MViT model for spatial (RGB) input"""
    def __init__(self, num_classes, dropout=0.5):
        super().__init__()
        # Load pretrained MViT model
        weights = MViT_V2_S_Weights.DEFAULT
        self.mvit = mvit_v2_s(weights=weights)
        
        # Get the feature dimension from the original head
        feature_dim = self.mvit.head[1].in_features
        self.num_features = feature_dim
        
        # Replace the classification head (will match model's dtype automatically)
        self.mvit.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes)
        )
        
    def forward(self, x, use_checkpoint=False):
        # x shape: [batch_size, 3, num_frames, height, width]
        # Let accelerate/autocast handle dtype conversion automatically
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
        )  # Let accelerate handle dtype
        
        # Initialize the new conv layer
        with torch.no_grad():
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
            if new_conv.bias is not None:
                nn.init.constant_(new_conv.bias, 0)
        
        mvit_model.conv_proj = new_conv
        
        # Replace the classification head
        mvit_model.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes)
        )
        
        self.mvit = mvit_model
        
    def forward(self, x, use_checkpoint=False):
        # x shape: [batch_size, 2, num_frames, height, width]
        # Let accelerate/autocast handle dtype conversion automatically
        return self.mvit(x)


def get_args_parser():
    parser = argparse.ArgumentParser(description='MViT training with LaViLa infrastructure', add_help=False)
    # Data
    parser.add_argument('--dataset', default='egtea', type=str,
                        choices=['ek100_cls', 'egtea'])
    parser.add_argument('--root',
                        default='/mnt/j/video_clips/cropped_clips/',
                        type=str, help='path to dataset root')
    parser.add_argument('--metadata-train',
                        default='../data/EGTEA/raw/annotation/split/train_split1.txt',
                        type=str, help='path to metadata file (train set)')
    parser.add_argument('--metadata-val',
                        default='../data/EGTEA/raw/annotation/split/test_split1.txt',
                        type=str, help='path to metadata file (val set)')
    parser.add_argument('--output-dir', default='./', type=str, help='output dir')
    parser.add_argument('--num-crops', default=1, type=int, help='number of crops in transforms for val')
    parser.add_argument('--num-clips', default=1, type=int, help='number of clips for val')
    parser.add_argument('--clip-length', default=16, type=int, help='clip length')
    parser.add_argument('--clip-stride', default=2, type=int, help='clip stride')
    parser.add_argument('--sparse-sample', action='store_true', help='switch to sparse sampling')
    parser.add_argument('--mini-dataset', action='store_true', 
                        help='use mini dataset (first 20 samples) for quick testing')
    
    # Model
    parser.add_argument('--model-type', default='mvit_spatial', type=str,
                        choices=['mvit_spatial', 'mvit_temporal'],
                        help='type of MViT model to use')
    parser.add_argument('--pretrain-model', default='', type=str, help='path to pretrain model')
    parser.add_argument('--resume', default='', type=str, help='path to resume from')
    parser.add_argument('--find-unused-parameters', action='store_true',
                        help='do this during DDP (useful for models with tied weights)')
    parser.add_argument('--drop-path-rate', default=0.1, type=float, help='drop path ratio')
    parser.add_argument('--dropout-ratio', default=0.5, type=float, help='dropout ratio for the last linear layer')
    parser.add_argument('--num-classes', default=106, type=int, help='number of classes for the last linear layer')
    parser.add_argument('--egtea-finetune-type', default='action', type=str,
                        choices=['action', 'verb', 'noun'],
                        help='EGTEA finetuning type')
    parser.add_argument('--use-half', action='store_true', help='use half precision at inference')
    
    # Training
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--warmup-epochs', default=5, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=16, type=int,
                        help='number of samples per-device/per-gpu')
    parser.add_argument('--use-sgd', action='store_true')
    parser.add_argument('--freeze-temperature', action='store_true', help='freeze temperature if set to True')
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--fix-lr', action='store_true', help='disable cosine lr decay if set True')
    parser.add_argument('--lr-start', default=1e-6, type=float,
                        help='initial warmup lr')
    parser.add_argument('--lr-end', default=1e-5, type=float,
                        help='minimum final lr')
    parser.add_argument('--lr-multiplier-on-backbone', default=0.1, type=float, help='lr multiplier for the backbone')
    parser.add_argument('--clip-grad-type', default='norm', choices=['norm', 'value'])
    parser.add_argument('--clip-grad-value', default=None, type=float, help='')
    parser.add_argument('--update-freq', default=1, type=int,
                        help='optimizer update frequency (i.e. gradient accumulation steps)')
    parser.add_argument('--wd', default=0.01, type=float)
    parser.add_argument('--betas', default=(0.9, 0.999), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--label-smoothing', default=0.1, type=float, help='label smoothing')
    parser.add_argument('--eval-freq', default=5, type=int)
    parser.add_argument('--save-freq', default=5, type=int)
    parser.add_argument('--disable-amp', action='store_true',
                        help='disable mixed-precision training (requires more memory and compute)')
    parser.add_argument('--use-zero', action='store_true', help='use ZeroRedundancyOptimizer')
    parser.add_argument('--use-checkpoint', action='store_true', help='use gradient checkpointing during training')
    
    # System
    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')
    
    return parser


def main(args):
    # Initialize Accelerator for mixed precision training
    mixed_precision = None if args.disable_amp else "fp16"
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=args.update_freq,
    )
    
    # For compatibility with existing distributed utils
    args.distributed = accelerator.distributed_type != 'NO'
    args.gpu = accelerator.device
    args.rank = accelerator.process_index
    args.world_size = accelerator.num_processes
    
    print("=> creating output dir: {}".format(args.output_dir))
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    random_seed(args.seed, accelerator.process_index)
    
    # Set number of classes based on dataset and finetune type
    if args.dataset == 'egtea':
        if args.egtea_finetune_type == 'action':
            args.num_classes = 106
        elif args.egtea_finetune_type == 'verb':
            args.num_classes = 19
        elif args.egtea_finetune_type == 'noun':
            args.num_classes = 53
    elif args.dataset == 'ek100_cls':
        if args.egtea_finetune_type == 'action':
            args.num_classes = 3806  # Epic Kitchen 100 has 3806 action classes
        elif args.egtea_finetune_type == 'verb':
            args.num_classes = 97  # Epic Kitchen 100 has 97 verb classes
        elif args.egtea_finetune_type == 'noun':
            args.num_classes = 300  # Epic Kitchen 100 has 300 noun classes
    
    print("=> creating model: {}".format(args.model_type))
    if args.model_type == 'mvit_spatial':
        model = MViT_Spatial(args.num_classes, dropout=args.dropout_ratio)
    elif args.model_type == 'mvit_temporal':
        model = MViT_Temporal(args.num_classes, dropout=args.dropout_ratio)
    else:
        raise NotImplementedError(f"Model type {args.model_type} not implemented")
    
    # Load pretrained weights if provided
    if args.pretrain_model and args.pretrain_model.lower() not in ['', 'none', 'null']:
        print(f"=> loading pretrained model from: {args.pretrain_model}")
        try:
            if args.pretrain_model.endswith('.safetensors'):
                from safetensors.torch import load_file
                state_dict = load_file(args.pretrain_model)
            else:
                checkpoint = torch.load(args.pretrain_model, map_location='cpu', weights_only=False)
                state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            
            # Clean up state dict keys
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                k = k.replace('module.', '')
                new_state_dict[k] = v
            
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
            print(f"=> loaded pretrained model")
            if missing_keys:
                print(f"   Missing keys: {len(missing_keys)} keys")
            if unexpected_keys:
                print(f"   Unexpected keys: {len(unexpected_keys)} keys")
        except FileNotFoundError:
            print(f"=> Warning: Pretrained model file not found at {args.pretrain_model}")
            print("=> Training from scratch with ImageNet pretrained backbone")
    else:
        print("=> No pretrained model specified, training from scratch with ImageNet pretrained backbone")
    
    # Move model to device (accelerator will handle this)
    model = model.to(accelerator.device)
    
    # Initialize RAFT model for temporal mode
    flow_model = None
    if args.model_type == 'mvit_temporal':
        flow_model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False)
        flow_model = flow_model.to(accelerator.device)
        flow_model = flow_model.eval()
        print("=> Initialized RAFT model for optical flow computation")
    
    # Create optimizer
    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        elif p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)
    
    optim_params = [
        {"params": p_wd, "weight_decay": args.wd},
        {"params": p_non_wd, "weight_decay": 0}
    ]
    
    if args.use_zero:
        optimizer = ZeroRedundancyOptimizer(
            optim_params, optimizer_class=torch.optim.SGD if args.use_sgd else torch.optim.AdamW,
            lr=args.lr, betas=args.betas, eps=args.eps, weight_decay=args.wd
        )
    else:
        if args.use_sgd:
            optimizer = torch.optim.SGD(optim_params, lr=args.lr, momentum=args.betas[0], weight_decay=args.wd)
        else:
            optimizer = torch.optim.AdamW(optim_params, lr=args.lr, betas=args.betas,
                                          eps=args.eps, weight_decay=args.wd)
    
    # Accelerator handles mixed precision, no need for scaler
    
    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # Resume from checkpoint
    best_acc1 = 0.
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc1 = checkpoint.get('best_acc1', 0.)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    cudnn.benchmark = True
    
    # Data loading
    print("=> creating dataset")
    
    crop_size = 224
    # Different transforms for spatial (RGB) and temporal (optical flow)
    if args.model_type == 'mvit_spatial':
        # RGB transforms
        train_transform = transforms.Compose([
            Permute([3, 0, 1, 2]),    # T H W C -> C T H W
            transforms.Resize(256),
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms_video.NormalizeVideo(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
        ])
        val_transform = transforms.Compose([
            Permute([3, 0, 1, 2]),    # T H W C -> C T H W
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms_video.NormalizeVideo(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
            # Note: Removed TemporalCrop and SpatialCrop as they return lists
            # The dataset should handle clip extraction
        ])
    else:  # mvit_temporal
        # For temporal model, we don't normalize here since RAFT needs raw RGB values
        # Flow normalization will be done after RAFT computation
        train_transform = transforms.Compose([
            Permute([3, 0, 1, 2]),    # T H W C -> C T H W
            transforms.Resize(256),
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(0.5),
            # No normalization here - RAFT needs raw pixel values
        ])
        val_transform = transforms.Compose([
            Permute([3, 0, 1, 2]),    # T H W C -> C T H W
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            # No normalization here - RAFT needs raw pixel values
        ])
    
    # Build dataset
    _, mapping_vn2act = generate_label_map(args.dataset, args)
    
    tokenizer = SimpleTokenizer()  # Not used for classification, but required by dataset
    
    num_clips_at_val = args.num_clips
    args.num_clips = 1
    
    # Use flow datasets for temporal model
    if args.model_type == 'mvit_temporal':
        train_dataset = datasets_flow.get_downstream_dataset(
            train_transform, tokenizer, args, subset='train', label_mapping=mapping_vn2act,
        )
    else:
        train_dataset = datasets.get_downstream_dataset(
            train_transform, tokenizer, args, subset='train', label_mapping=mapping_vn2act,
        )
    
    args.num_clips = num_clips_at_val
    
    if args.model_type == 'mvit_temporal':
        val_dataset = datasets_flow.get_downstream_dataset(
            val_transform, tokenizer, args, subset='val', label_mapping=mapping_vn2act,
        )
    else:
        val_dataset = datasets.get_downstream_dataset(
            val_transform, tokenizer, args, subset='val', label_mapping=mapping_vn2act,
        )
    
    # Accelerator will handle distributed sampling
    train_sampler = None
    val_sampler = None
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True
    )
    print('len(train_loader) = {}'.format(len(train_loader)))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False
    )
    print('len(val_loader) = {}'.format(len(val_loader)))
    
    # Learning rate schedule
    if args.fix_lr:
        lr_schedule = None
    else:
        lr_schedule = cosine_scheduler(
            args.lr, args.lr_end, args.epochs, len(train_loader) // args.update_freq,
            warmup_epochs=args.warmup_epochs, start_warmup_value=args.lr_start,
        )
    
    # Prepare model, optimizer, and dataloaders with accelerator
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    if accelerator.is_main_process and args.wandb:
        wandb_id = os.path.split(args.output_dir)[-1]
        wandb.init(project='MViT-LaViLa', id=wandb_id, config=args, resume='allow')
    
    print(args)
    
    # Training loop
    # Adjust print frequency for mini dataset
    if args.mini_dataset:
        args.print_freq = 1  # Print every batch in mini mode
        print("=> Mini dataset mode: Setting print_freq to 1")
    
    print("=> beginning training")
    for epoch in range(args.start_epoch, args.epochs):
        # Train for one epoch
        train_stats = train(train_loader, model, criterion, optimizer, accelerator, epoch, lr_schedule, args, flow_model)
        
        is_epoch = ((epoch + 1) % args.save_freq) == 0
        
        # Save checkpoint
        if accelerator.is_main_process:
            print('=> saving checkpoint')
            unwrapped_model = accelerator.unwrap_model(model)
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': unwrapped_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_acc1': best_acc1,
                'args': args,
            }
            if is_epoch:
                checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            else:
                checkpoint_path = os.path.join(args.output_dir, 'checkpoint.pt')
            torch.save(checkpoint, checkpoint_path)
        
        # Evaluate
        if ((epoch + 1) % args.eval_freq) == 0:
            val_stats = validate(val_loader, model, accelerator, args, flow_model)
            if val_stats['acc1'] > best_acc1:
                is_best = True
                best_acc1 = val_stats['acc1']
            else:
                is_best = False
            
            if accelerator.is_main_process:
                print('=> saving checkpoint')
                unwrapped_model = accelerator.unwrap_model(model)
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': unwrapped_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_acc1': best_acc1,
                    'args': args,
                }
                if is_best:
                    checkpoint_path = os.path.join(args.output_dir, 'checkpoint_best.pt')
                    torch.save(checkpoint, checkpoint_path)
            
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in val_stats.items()},
                         'epoch': epoch}
            
            if accelerator.is_main_process:
                if args.wandb:
                    wandb.log(log_stats)
                with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                    f.write(json.dumps(log_stats) + '\n')


def train(train_loader, model, criterion, optimizer, accelerator, epoch, lr_schedule, args, flow_model=None):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5, mem],
        prefix="Epoch: [{}]".format(epoch))
    
    # Switch to train mode
    model.train()
    
    # print(f"Starting training epoch {epoch}, {len(train_loader)} batches")  # Debug
    end = time.time()
    for i, batch in enumerate(train_loader):
        # print(f"Processing batch {i}/{len(train_loader)}")  # Debug
        # Measure data loading time
        data_time.update(time.time() - end)
        
        # Handle different batch formats based on model type
        if args.model_type == 'mvit_temporal':
            # Flow dataset returns (images, images_flow, target)
            if len(batch) == 3:
                images, images_flow, target = batch
            else:
                raise ValueError(f"Expected 3 elements for temporal model, got {len(batch)}")
        else:
            # Regular dataset returns (images, target)
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, target = batch
            else:
                images = batch[0] if isinstance(batch, (list, tuple)) else batch
                target = batch[1] if isinstance(batch, (list, tuple)) and len(batch) > 1 else None
        
        # Handle list of images (from multi-crop transforms)
        if isinstance(images, list):
            images = images[0] if len(images) > 0 else images
        
        # Process based on model type
        if args.model_type == 'mvit_temporal' and flow_model is not None:
            # Compute optical flow for temporal model
            batch_size = images.shape[0]
            num_frames = args.clip_length
            
            # Compute optical flow between consecutive frame pairs
            flow_list = []
            with torch.no_grad():
                with torch.backends.cudnn.flags(enabled=False):
                    for b in range(batch_size):
                        batch_flows = []
                        for t in range(num_frames):
                            # Get frame pair for this timestep
                            frame1 = images[b:b+1, :, t, :, :].contiguous()  # [1, C, H, W]
                            frame2 = images_flow[b:b+1, :, t, :, :].contiguous()  # [1, C, H, W]
                            
                            # Compute flow between frame pair
                            flow_out = flow_model(frame1, frame2)
                            flow_frame = flow_out[-1]  # [1, 2, H, W]
                            batch_flows.append(flow_frame)
                        
                        # Stack flows for this batch item
                        batch_flows = torch.cat(batch_flows, dim=0)  # [T, 2, H, W]
                        flow_list.append(batch_flows.unsqueeze(0))  # [1, T, 2, H, W]
            
            # Combine all batch items
            flow_sequence = torch.cat(flow_list, dim=0)  # [B, T, 2, H, W]
            
            # Normalize optical flow
            flow_normalize = transforms.Normalize(mean=[0, 0], std=[20, 20])
            H, W = images.shape[-2:]
            flow_normalized = flow_normalize(flow_sequence.view(-1, 2, H, W)).view(batch_size, num_frames, 2, H, W)
            
            # Permute to [B, 2, T, H, W] for MViT_Temporal
            model_input = flow_normalized.permute(0, 2, 1, 3, 4)
            
            # Compute output
            output = model(model_input, use_checkpoint=args.use_checkpoint)
        else:
            # Regular spatial model
            output = model(images, use_checkpoint=args.use_checkpoint)
        
        loss = criterion(output, target)
        
        # Measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))
        
        # Compute gradient and do optimizer step
        # Accelerator handles gradient accumulation
        accelerator.backward(loss)
        
        if (i + 1) % args.update_freq == 0 or (i + 1) == len(train_loader):
            if args.clip_grad_value is not None:
                if args.clip_grad_type == 'norm':
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_value)
                elif args.clip_grad_type == 'value':
                    accelerator.clip_grad_value_(model.parameters(), args.clip_grad_value)
                else:
                    assert False, f"Unknown clip_grad_type: {args.clip_grad_type}"
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Update learning rate after optimizer step
            if lr_schedule is not None:
                step = epoch * len(train_loader) // args.update_freq + i // args.update_freq
                if step < len(lr_schedule):
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_schedule[step]
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        mem.update(torch.cuda.max_memory_allocated() // 1e9)
        
        if i % args.print_freq == 0:
            progress.display(i)
    
    # Metrics are automatically synchronized by accelerator
    return {'loss': losses.avg, 'acc1': top1.avg, 'acc5': top5.avg,
            'lr': optimizer.param_groups[0]['lr']}


def validate(val_loader, model, accelerator, args, flow_model=None):
    batch_time = AverageMeter('Time', ':6.2f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')
    
    # Switch to evaluate mode
    model.eval()
    
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(val_loader):
            # Handle different batch formats based on model type
            if args.model_type == 'mvit_temporal':
                # Flow dataset returns (images, images_flow, target)
                if len(batch) == 3:
                    images, images_flow, target = batch
                else:
                    raise ValueError(f"Expected 3 elements for temporal model, got {len(batch)}")
            else:
                # Regular dataset returns (images, target)
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    images, target = batch
                else:
                    images = batch[0] if isinstance(batch, (list, tuple)) else batch
                    target = batch[1] if isinstance(batch, (list, tuple)) and len(batch) > 1 else None
            
            # If images is a list (from multi-crop), take the first one or process all
            if isinstance(images, list):
                # For validation, we'll just use the first crop for simplicity
                # You could also average predictions across all crops for better accuracy
                images = images[0] if len(images) > 0 else images
                if args.model_type == 'mvit_temporal':
                    images_flow = images_flow[0] if isinstance(images_flow, list) and len(images_flow) > 0 else images_flow
            
            # Ensure we have a tensor
            if not isinstance(images, torch.Tensor):
                print(f"Warning: Unexpected image type: {type(images)}")
                continue
            
            # Process based on model type
            if args.model_type == 'mvit_temporal' and flow_model is not None:
                # Compute optical flow for temporal model
                batch_size = images.shape[0]
                num_frames = args.clip_length
                
                # Compute optical flow between consecutive frame pairs
                flow_list = []
                with torch.backends.cudnn.flags(enabled=False):
                    for b in range(batch_size):
                        batch_flows = []
                        for t in range(num_frames):
                            # Get frame pair for this timestep
                            frame1 = images[b:b+1, :, t, :, :].contiguous()  # [1, C, H, W]
                            frame2 = images_flow[b:b+1, :, t, :, :].contiguous()  # [1, C, H, W]
                            
                            # Compute flow between frame pair
                            flow_out = flow_model(frame1, frame2)
                            flow_frame = flow_out[-1]  # [1, 2, H, W]
                            batch_flows.append(flow_frame)
                        
                        # Stack flows for this batch item
                        batch_flows = torch.cat(batch_flows, dim=0)  # [T, 2, H, W]
                        flow_list.append(batch_flows.unsqueeze(0))  # [1, T, 2, H, W]
                
                # Combine all batch items
                flow_sequence = torch.cat(flow_list, dim=0)  # [B, T, 2, H, W]
                
                # Normalize optical flow
                flow_normalize = transforms.Normalize(mean=[0, 0], std=[20, 20])
                H, W = images.shape[-2:]
                flow_normalized = flow_normalize(flow_sequence.view(-1, 2, H, W)).view(batch_size, num_frames, 2, H, W)
                
                # Permute to [B, 2, T, H, W] for MViT_Temporal
                model_input = flow_normalized.permute(0, 2, 1, 3, 4)
                
                # Compute output
                output = model(model_input)
            else:
                # Regular spatial model
                output = model(images)
            
            # Store for confusion matrix
            all_outputs.append(output.cpu())
            all_targets.append(target.cpu())
            
            # Measure accuracy
            if target is not None:
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                batch_size = images.size(0) if isinstance(images, torch.Tensor) else len(images)
                top1.update(acc1.item(), batch_size)
                top5.update(acc5.item(), batch_size)
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % args.print_freq == 0:
                progress.display(i)
    
    # Metrics are automatically synchronized by accelerator
    
    # Compute per-class accuracy if main process
    if accelerator.is_main_process:
        all_outputs = torch.cat(all_outputs)
        all_targets = torch.cat(all_targets)
        
        predictions = all_outputs.argmax(dim=1)
        cm = confusion_matrix(all_targets.numpy(), predictions.numpy())
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        mean_class_acc = np.mean(per_class_acc)
        
        print(f' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Mean Class Acc {mean_class_acc:.3f}')
    else:
        mean_class_acc = 0.
    
    return {'acc1': top1.avg, 'acc5': top5.avg, 'mean_class_acc': mean_class_acc}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MViT training with LaViLa infrastructure', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)