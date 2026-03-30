# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Feature extraction script for MViT and V-JEPA backbones.
Outputs follow the same format as the original extractor:
  {'feats': ..., 'cls_feats': ..., 'outputs': ..., 'targets': ...}
"""

import argparse
from collections import OrderedDict
import os
import sys
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.cuda.amp as amp
import torch.nn.parallel
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video

try:
	from safetensors.torch import load_file as load_safetensors
	HAS_SAFETENSORS = True
except ImportError:
	HAS_SAFETENSORS = False
	print("Warning: safetensors not available. Install with 'pip install safetensors' to load .safetensors files.")

from lavila.data import datasets, datasets_flow
from lavila.data.video_transforms import Permute, SpatialCrop, TemporalCrop
from lavila.models.tokenizer import SimpleTokenizer
from lavila.utils import distributed as dist_utils
from lavila.utils.meter import AverageMeter, ProgressMeter
from lavila.utils.preprocess import generate_label_map
from lavila.utils.random import random_seed

from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights


VJEPA2_MODEL_SPECS = {
	'vjepa2_large': {'hub_name': 'vjepa2_vit_large', 'crop_size': 256, 'repo_dir': 'vjepa2'},
	'vjepa2_huge': {'hub_name': 'vjepa2_vit_huge', 'crop_size': 256, 'repo_dir': 'vjepa2'},
	'vjepa2_giant': {'hub_name': 'vjepa2_vit_giant', 'crop_size': 256, 'repo_dir': 'vjepa2'},
	'vjepa2_giant_384': {'hub_name': 'vjepa2_vit_giant_384', 'crop_size': 384, 'repo_dir': 'vjepa2'},
	'vjepa2_1_vit_base_384': {'hub_name': 'vjepa2_1_vit_base_384', 'crop_size': 384, 'repo_dir': 'vjepa2.1'},
	'vjepa2_1_vit_large_384': {'hub_name': 'vjepa2_1_vit_large_384', 'crop_size': 384, 'repo_dir': 'vjepa2.1'},
	'vjepa2_1_vit_giant_384': {'hub_name': 'vjepa2_1_vit_giant_384', 'crop_size': 384, 'repo_dir': 'vjepa2.1'},
	'vjepa2_1_vit_gigantic_384': {'hub_name': 'vjepa2_1_vit_gigantic_384', 'crop_size': 384, 'repo_dir': 'vjepa2.1'},
}


def _get_vjepa2_repo_root(variant_key: str):
	if variant_key not in VJEPA2_MODEL_SPECS:
		raise ValueError(f'Unsupported V-JEPA variant: {variant_key}')
	repo_dir = VJEPA2_MODEL_SPECS[variant_key].get('repo_dir', 'vjepa2')
	return os.path.join(os.path.dirname(__file__), "thirdparty", repo_dir)


def _ensure_vjepa2_repo_on_path(variant_key: str):
	repo_root = _get_vjepa2_repo_root(variant_key)
	if os.path.isdir(repo_root) and repo_root not in sys.path:
		sys.path.insert(0, repo_root)
	return repo_root


def _load_vjepa2_encoder(variant_key: str, pretrained: bool = True):
	if variant_key not in VJEPA2_MODEL_SPECS:
		raise ValueError(f'Unsupported V-JEPA variant: {variant_key}')
	spec = VJEPA2_MODEL_SPECS[variant_key]
	hub_name = spec['hub_name']
	repo_root = _ensure_vjepa2_repo_on_path(variant_key)
	if os.path.isdir(repo_root):
		try:
			encoder, _ = torch.hub.load(repo_root, hub_name, source='local', pretrained=pretrained)
			return encoder
		except Exception as exc:
			if not pretrained:
				raise
			print(f"Local V-JEPA hub load failed for {variant_key} ({exc}). Falling back to facebookresearch/vjepa2.")
	encoder, _ = torch.hub.load('facebookresearch/vjepa2', hub_name, pretrained=pretrained)
	return encoder


def load_checkpoint(checkpoint_path: str):
	"""Load checkpoint from either .pt/.pth or .safetensors file."""
	if checkpoint_path.endswith('.safetensors'):
		if not HAS_SAFETENSORS:
			raise ImportError("safetensors library is required to load .safetensors files. Install with 'pip install safetensors'")
		print(f"Loading .safetensors file: {checkpoint_path}")
		state_dict = load_safetensors(checkpoint_path)
		return {'state_dict': state_dict, 'epoch': 0, 'best_acc1': 0.0}
	else:
		print(f"Loading PyTorch checkpoint: {checkpoint_path}")
		return torch.load(checkpoint_path, map_location='cpu', weights_only=False)


class VJEPA2FeatureExtractor(nn.Module):
	"""Thin wrapper that exposes logits and token-level features from V-JEPA2."""

	def __init__(self, variant_key: str, num_classes: int, dropout: float = 0.5, pretrained_backbone: bool = True):
		super().__init__()
		self.encoder = _load_vjepa2_encoder(variant_key, pretrained=pretrained_backbone)
		self.num_features = self.encoder.embed_dim
		self.classifier = nn.Sequential(
			nn.Dropout(p=dropout),
			nn.Linear(self.num_features, num_classes),
		)

	def forward(self, x, return_features: bool = False):
		tokens = self.encoder(x)  # [B, N, C]
		pooled = tokens.mean(dim=1)
		logits = self.classifier(pooled)
		if return_features:
			return logits, tokens, pooled
		return logits


class MViT_Spatial(nn.Module):
	"""MViT model for RGB input."""

	def __init__(self, num_classes: int, dropout: float = 0.5):
		super().__init__()
		weights = MViT_V2_S_Weights.DEFAULT
		self.mvit = mvit_v2_s(weights=weights)
		feature_dim = self.mvit.head[1].in_features
		self.num_features = feature_dim
		self.mvit.head = nn.Sequential(
			nn.Dropout(p=dropout),
			nn.Linear(feature_dim, num_classes),
		)

	def forward(self, x):
		return self.mvit(x)


class MViT_Temporal(nn.Module):
	"""MViT model for optical flow input."""

	def __init__(self, num_classes: int, dropout: float = 0.5):
		super().__init__()
		weights = MViT_V2_S_Weights.DEFAULT
		mvit_model = mvit_v2_s(weights=weights)
		feature_dim = mvit_model.head[1].in_features
		self.num_features = feature_dim

		original_conv = mvit_model.conv_proj
		new_conv = nn.Conv3d(
			in_channels=2,
			out_channels=original_conv.out_channels,
			kernel_size=original_conv.kernel_size,
			stride=original_conv.stride,
			padding=original_conv.padding,
			bias=original_conv.bias is not None,
		)
		nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
		if new_conv.bias is not None:
			nn.init.constant_(new_conv.bias, 0)
		mvit_model.conv_proj = new_conv

		mvit_model.head = nn.Sequential(
			nn.Dropout(p=dropout),
			nn.Linear(feature_dim, num_classes),
		)
		self.mvit = mvit_model

	def forward(self, x):
		return self.mvit(x)


def reshape_temporal_features(token_feats: torch.Tensor, clip_length: int):
	"""
	Reshape token features to [B, T, D] by averaging spatial tokens per frame.
	Falls back to flat token layout if reshaping is not possible.
	"""
	token_count = token_feats.shape[1]
	if clip_length > 0 and token_count % clip_length == 0:
		tokens_per_frame = token_count // clip_length
		return token_feats.view(
			token_feats.shape[0], clip_length, tokens_per_frame, token_feats.shape[-1]
		).mean(dim=2)
	return token_feats


def build_transforms(args):
	default_crop_size = 224
	crop_size = VJEPA2_MODEL_SPECS.get(args.model_type, {}).get('crop_size', default_crop_size)
	args.crop_size = crop_size

	if args.model_type == 'mvit_temporal':
		train_transform = transforms.Compose([
			Permute([3, 0, 1, 2]),  # T H W C -> C T H W
			transforms.RandomResizedCrop(crop_size, scale=(0.5, 1.0)),
			transforms.RandomHorizontalFlip(p=0.5),
		])
		val_transform = transforms.Compose([
			Permute([3, 0, 1, 2]),
			transforms.Resize(crop_size),
			transforms.CenterCrop(crop_size),
			TemporalCrop(frames_per_clip=args.clip_length, stride=args.clip_length),
			SpatialCrop(crop_size=crop_size, num_crops=args.num_crops),
		])
	else:
		train_transform = transforms.Compose([
			Permute([3, 0, 1, 2]),
			transforms.RandomResizedCrop(crop_size, scale=(0.5, 1.0)),
			transforms.RandomHorizontalFlip(p=0.5),
			transforms_video.NormalizeVideo(mean=[123.675, 116.28, 103.53],
										   std=[58.395, 57.12, 57.375]),
		])
		val_transform = transforms.Compose([
			Permute([3, 0, 1, 2]),
			transforms.Resize(crop_size),
			transforms.CenterCrop(crop_size),
			transforms_video.NormalizeVideo(mean=[123.675, 116.28, 103.53],
										   std=[58.395, 57.12, 57.375]),
			TemporalCrop(frames_per_clip=args.clip_length, stride=args.clip_length),
			SpatialCrop(crop_size=crop_size, num_crops=args.num_crops),
		])
	return train_transform, val_transform, crop_size


def compute_flow_sequence(images, images_flow, flow_model, args):
	"""Compute optical flow volume using RAFT for a batch."""
	batch_size = images.shape[0]
	num_frames = args.clip_length
	crop_size = args.crop_size

	flow_list = []
	with torch.no_grad():
		with torch.backends.cudnn.flags(enabled=False):
			for b in range(batch_size):
				batch_flows = []
				for t in range(num_frames):
					frame1 = images[b:b + 1, :, t, :, :].contiguous()
					frame2 = images_flow[b:b + 1, :, t, :, :].contiguous()
					flow_out = flow_model(frame1, frame2)
					flow_frame = flow_out[-1]
					batch_flows.append(flow_frame)
				batch_flows = torch.cat(batch_flows, dim=0)
				flow_list.append(batch_flows.unsqueeze(0))

	flow_sequence = torch.cat(flow_list, dim=0)
	flow_normalize = transforms.Normalize(mean=[0, 0], std=[20, 20])
	flow_normalized = flow_normalize(flow_sequence.view(-1, 2, crop_size, crop_size)).view(
		batch_size, num_frames, 2, crop_size, crop_size)
	return flow_normalized.permute(0, 2, 1, 3, 4)


def forward_with_features(model, model_type, activation, inputs, args):
	"""Run model forward pass and return logits, token features and cls features."""
	if model_type in VJEPA2_MODEL_SPECS:
		logits, tokens, pooled = model(inputs, return_features=True)
		feat = reshape_temporal_features(tokens, args.clip_length)
		cls_feat = pooled
	else:
		logits = model(inputs)
		token_output = activation['tokens']
		cls_feat = token_output[:, 0, :]
		patch_tokens = token_output[:, 1:, :]
		feat = reshape_temporal_features(patch_tokens, args.clip_length)
	return logits, feat, cls_feat


def extract_split(loader, model, flow_model, args, subset: str):
	"""Extract features for a single split and save to disk."""
	batch_time = AverageMeter('Time', ':6.2f')
	data_time = AverageMeter('Data', ':6.2f')
	progress = ProgressMeter(len(loader), [batch_time, data_time], prefix=f'{subset}: ')

	model.eval()
	if args.use_half:
		model.half()

	activation = {}
	hook_handle = None
	model_without_ddp = model.module if hasattr(model, 'module') else model
	if args.model_type.startswith('mvit'):
		def hook(_, __, output):
			activation['tokens'] = output.detach()
		hook_handle = model_without_ddp.mvit.blocks[-1].mlp.register_forward_hook(hook)

	all_outputs, all_targets, all_feats, all_cls_feats = [], [], [], []
	end = time.time()

	with torch.no_grad():
		for i, batch_data in enumerate(loader):
			data_time.update(time.time() - end)

			if args.model_type == 'mvit_temporal':
				images, images_flow, target = batch_data
				target_gpu = target.cuda(args.gpu, non_blocking=True)
				if isinstance(images, list):
					logits_crops, feats_crops, cls_crops = [], [], []
					for crop, crop_flow in zip(images, images_flow):
						crop = crop.cuda(args.gpu, non_blocking=True)
						crop_flow = crop_flow.cuda(args.gpu, non_blocking=True)
						flow_input = compute_flow_sequence(crop, crop_flow, flow_model, args)
						if args.use_half:
							flow_input = flow_input.half()
						logits, feat, cls_feat = forward_with_features(
							model, args.model_type, activation, flow_input, args
						)
						logits_crops.append(logits.unsqueeze(1).detach().cpu())
						feats_crops.append(feat.unsqueeze(1).detach().cpu())
						cls_crops.append(cls_feat.unsqueeze(1).detach().cpu())
					all_outputs.append(torch.cat(logits_crops, dim=1))
					all_feats.append(torch.cat(feats_crops, dim=1))
					all_cls_feats.append(torch.cat(cls_crops, dim=1))
				else:
					images = images.cuda(args.gpu, non_blocking=True)
					images_flow = images_flow.cuda(args.gpu, non_blocking=True)
					flow_input = compute_flow_sequence(images, images_flow, flow_model, args)
					if args.use_half:
						flow_input = flow_input.half()
					logits, feat, cls_feat = forward_with_features(
						model, args.model_type, activation, flow_input, args
					)
					all_outputs.append(logits.detach().cpu())
					all_feats.append(feat.detach().cpu())
					all_cls_feats.append(cls_feat.detach().cpu())
				all_targets.append(target_gpu.detach().cpu())
			else:
				images, target = batch_data
				target_gpu = target.cuda(args.gpu, non_blocking=True)
				if isinstance(images, list):
					logits_crops, feats_crops, cls_crops = [], [], []
					for crop in images:
						crop = crop.cuda(args.gpu, non_blocking=True)
						if args.use_half:
							crop = crop.half()
						logits, feat, cls_feat = forward_with_features(
							model, args.model_type, activation, crop, args
						)
						logits_crops.append(logits.unsqueeze(1).detach().cpu())
						feats_crops.append(feat.unsqueeze(1).detach().cpu())
						cls_crops.append(cls_feat.unsqueeze(1).detach().cpu())
					all_outputs.append(torch.cat(logits_crops, dim=1))
					all_feats.append(torch.cat(feats_crops, dim=1))
					all_cls_feats.append(torch.cat(cls_crops, dim=1))
				else:
					images = images.cuda(args.gpu, non_blocking=True)
					if args.use_half:
						images = images.half()
					logits, feat, cls_feat = forward_with_features(
						model, args.model_type, activation, images, args
					)
					all_outputs.append(logits.detach().cpu())
					all_feats.append(feat.detach().cpu())
					all_cls_feats.append(cls_feat.detach().cpu())
				all_targets.append(target_gpu.detach().cpu())

			batch_time.update(time.time() - end)
			end = time.time()

			if i % args.print_freq == 0:
				progress.display(i)

	if hook_handle is not None:
		hook_handle.remove()

	all_feats = torch.cat(all_feats)
	all_cls_feats = torch.cat(all_cls_feats)
	all_outputs = torch.cat(all_outputs)
	all_targets = torch.cat(all_targets)

	save_name = f"{args.dataset}_{subset}_feat.pt"
	save_path = os.path.join(args.output_dir, save_name)
	torch.save({
		'feats': all_feats,
		'cls_feats': all_cls_feats,
		'outputs': all_outputs,
		'targets': all_targets,
	}, save_path)

	print(f"Saved {subset} features to {save_path}")
	print(f"  feats: {all_feats.shape}")
	print(f"  cls_feats: {all_cls_feats.shape}")
	print(f"  outputs: {all_outputs.shape}")
	print(f"  targets: {all_targets.shape}")


def get_args_parser():
	parser = argparse.ArgumentParser(description='V-JEPA / MViT feature extraction', add_help=False)

	# Data
	parser.add_argument('--dataset', default='egtea', type=str, choices=['ek100_cls', 'egtea'])
	parser.add_argument('--task-type', default='action', type=str, choices=['action', 'verb', 'noun'])
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
	parser.add_argument('--root', default='/mnt/j/video_clips/cropped_clips/', type=str, help='path to dataset root')
	parser.add_argument('--metadata-train', default='../data/EGTEA/raw/annotation/split/train_split1.txt', type=str,
						help='path to metadata file (train set)')
	parser.add_argument('--metadata-val', default='../data/EGTEA/raw/annotation/split/test_split1.txt', type=str,
						help='path to metadata file (val set)')
	parser.add_argument('--output-dir', default='./output', type=str, help='directory to save extracted features')
	parser.add_argument('--num-crops', default=1, type=int, help='number of crops for val')
	parser.add_argument('--num-clips', default=1, type=int, help='number of clips for val')
	parser.add_argument('--clip-length', default=16, type=int, help='clip length')
	parser.add_argument('--clip-stride', default=2, type=int, help='clip stride')
	parser.add_argument('--sparse-sample', action='store_true', help='switch to sparse sampling')
	parser.add_argument('--batch-size', default=8, type=int, help='number of samples per GPU for extraction')

	# Model
	parser.add_argument('--model-type', default='vjepa2_large', type=str,
						choices=['mvit_spatial', 'mvit_temporal'] + list(VJEPA2_MODEL_SPECS.keys()),
						help='backbone to use for extraction')
	parser.add_argument('--pretrain-model', default='', type=str, help='path to checkpoint to load')
	parser.add_argument('--resume', default='', type=str, help='alternative checkpoint path')
	parser.add_argument('--skip-checkpoint', action='store_true',
						help='skip loading external checkpoint and use backbone default weights')
	parser.add_argument('--dropout-ratio', default=0.5, type=float, help='dropout ratio in classifier head')
	parser.add_argument('--num-classes', default=None, type=int, help='override class count (otherwise inferred)')
	parser.add_argument('--use-half', action='store_true', help='use half precision at inference')

	# System
	parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
	parser.add_argument('--workers', default=4, type=int, help='data loading workers')
	parser.add_argument('--seed', default=0, type=int)
	parser.add_argument('--gpu', default=None, type=int, help='GPU id to use')

	# Distributed
	parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed extraction')
	parser.add_argument('--rank', default=0, type=int, help='node rank for distributed extraction')
	parser.add_argument("--local_rank", type=int, default=0)
	parser.add_argument('--dist-url', default='env://', type=str, help='url used to set up distributed extraction')
	parser.add_argument('--dist-backend', default='nccl', type=str)
	parser.add_argument('--find-unused-parameters', action='store_true')

	return parser


def main(args):
	dist_utils.init_distributed_mode(args)
	random_seed(args.seed, dist_utils.get_rank())
	print(f'Random seed: {args.seed}')

	# Infer number of classes if not provided
	if args.num_classes is None:
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
	print(f"Using {args.num_classes} classes for task {args.task_type}")

	# This flag controls label selection inside dataset
	args.egtea_finetune_type = args.task_type

	ckpt_path = args.resume or args.pretrain_model
	ckpt_is_empty = not ckpt_path or ckpt_path.strip().lower() in {'none', 'null', 'nil'}
	pretrained_backbone = args.skip_checkpoint or ckpt_is_empty
	if args.model_type in VJEPA2_MODEL_SPECS:
		if pretrained_backbone:
			print("Initializing V-JEPA backbone from default pretrained weights.")
		else:
			print("Initializing V-JEPA backbone architecture only; external checkpoint will provide weights.")

	# Build model
	if args.model_type == 'mvit_spatial':
		model = MViT_Spatial(args.num_classes, dropout=args.dropout_ratio)
	elif args.model_type == 'mvit_temporal':
		model = MViT_Temporal(args.num_classes, dropout=args.dropout_ratio)
	elif args.model_type in VJEPA2_MODEL_SPECS:
		model = VJEPA2FeatureExtractor(
			args.model_type, args.num_classes, dropout=args.dropout_ratio, pretrained_backbone=pretrained_backbone
		)
	else:
		raise ValueError(f'Unknown model type: {args.model_type}')

	# Load checkpoint (optional)
	ckpt_path = args.resume or args.pretrain_model
	ckpt_is_empty = not ckpt_path or ckpt_path.strip().lower() in {'none', 'null', 'nil'}
	if args.skip_checkpoint or ckpt_is_empty:
		print("No checkpoint provided; using backbone default weights.")
	else:
		checkpoint = load_checkpoint(ckpt_path)
		state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
		new_state_dict = OrderedDict((k.replace('module.', ''), v) for k, v in state_dict.items())
		missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
		if missing:
			print(f"Missing keys in state dict: {missing}")
		if unexpected:
			print(f"Unexpected keys in state dict: {unexpected}")

	model.cuda(args.gpu)

	# Initialize RAFT for temporal model
	flow_model = None
	if args.model_type == 'mvit_temporal':
		flow_model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).cuda(args.gpu)
		flow_model.eval()
		print("=> Initialized RAFT model for optical flow computation")

	# Distributed wrapper
	if args.distributed:
		model = torch.nn.parallel.DistributedDataParallel(
			model, device_ids=[args.gpu], bucket_cap_mb=200,
			find_unused_parameters=args.find_unused_parameters
		)

	# Data transforms and datasets
	train_transform, val_transform, crop_size = build_transforms(args)
	tokenizer = SimpleTokenizer()
	_, mapping_vn2act = generate_label_map(args.dataset, args)

	num_clips_at_val = args.num_clips
	args.num_clips = 1
	args.num_crops = 3
	if args.model_type == 'mvit_temporal':
		train_dataset = datasets_flow.get_downstream_dataset_extract(
			train_transform, tokenizer, args, subset='train', label_mapping=mapping_vn2act,
		)
	else:
		train_dataset = datasets.get_downstream_dataset_extract(
			train_transform, tokenizer, args, subset='train', label_mapping=mapping_vn2act,
		)
	args.num_clips = num_clips_at_val
	args.num_crops = 1
	if args.model_type == 'mvit_temporal':
		val_dataset = datasets_flow.get_downstream_dataset_extract(
			val_transform, tokenizer, args, subset='val', label_mapping=mapping_vn2act,
		)
	else:
		val_dataset = datasets.get_downstream_dataset_extract(
			val_transform, tokenizer, args, subset='val', label_mapping=mapping_vn2act,
		)

	if args.distributed:
		train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
		val_sampler = torch.utils.data.SequentialSampler(val_dataset)
	else:
		train_sampler = None
		val_sampler = None

	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=False
	)
	val_loader = torch.utils.data.DataLoader(
		val_dataset, batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False
	)

	print(f"Training samples: {len(train_dataset)}, batches: {len(train_loader)}")
	print(f"Validation samples: {len(val_dataset)}, batches: {len(val_loader)}")

	cudnn.benchmark = True

	# Extract features
	extract_split(train_loader, model, flow_model, args, subset='train')
	extract_split(val_loader, model, flow_model, args, subset='test')


if __name__ == '__main__':
	parser = argparse.ArgumentParser('Feature extraction for V-JEPA / MViT', parents=[get_args_parser()])
	args = parser.parse_args()
	os.makedirs(args.output_dir, exist_ok=True)
	main(args)
