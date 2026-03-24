# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from collections import OrderedDict
import csv
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
from sklearn.metrics import confusion_matrix
import wandb

# For loading .safetensors files
try:
    from safetensors.torch import load_file as load_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    print("Warning: safetensors not available. Install with 'pip install safetensors' to load .safetensors files.")

from lavila.data import datasets
from lavila.data.video_transforms import Permute, SpatialCrop, TemporalCrop
from lavila.models import models
from lavila.models.tokenizer import (MyBertTokenizer, MyDistilBertTokenizer, MyGPT2Tokenizer, SimpleTokenizer)
from lavila.models.utils import inflate_positional_embeds
from lavila.utils import distributed as dist_utils
from lavila.utils.evaluation import accuracy, get_mean_accuracy
from lavila.utils.meter import AverageMeter, ProgressMeter
from lavila.utils.preprocess import generate_label_map
from lavila.utils.random import random_seed
from lavila.utils.scheduler import cosine_scheduler
from lavila.utils.evaluation_ek100cls import get_marginal_indexes, marginalize
from torchvision.models.video import (
	mvit_v2_s,
	MViT_V2_S_Weights,
	MViT_V1_B_Weights,
	mvit_v1_b,
)

def load_checkpoint(checkpoint_path):
	"""
	Load checkpoint from either .pt/.pth or .safetensors file
	Returns a dict with 'state_dict' and possibly other keys like 'epoch', 'optimizer', etc.
	"""
	if checkpoint_path.endswith('.safetensors'):
		if not HAS_SAFETENSORS:
			raise ImportError("safetensors library is required to load .safetensors files. Install with 'pip install safetensors'")
		
		print(f"Loading .safetensors file: {checkpoint_path}")
		state_dict = load_safetensors(checkpoint_path)
		
		# safetensors only contains state_dict, so we create a minimal checkpoint dict
		return {
			'state_dict': state_dict,
			'epoch': 0,  # Default values since safetensors doesn't store training metadata
			'best_acc1': 0.0,
		}
	else:
		# Standard PyTorch checkpoint
		print(f"Loading PyTorch checkpoint: {checkpoint_path}")
		return torch.load(checkpoint_path, map_location='cpu', weights_only=False)


def split_classifier_outputs(output):
	"""Normalize single-head vs multi-head classifier outputs."""
	if isinstance(output, list):
		if len(output) != 3:
			raise ValueError(f"Expected 3 heads for vn-classifier output, got {len(output)}")
		return output[2], {'verb_outputs': output[0], 'noun_outputs': output[1]}
	return output, {}

class MViT_Spatial(nn.Module):
	def __init__(self, num_classes, dropout=0.1):
		super().__init__()

		# Load the pre-trained MViT model
		#weights = MViT_V2_S_Weights.DEFAULT
		self.mvit = mvit_v2_s()

		# Replace the classification head
		self.mvit.head = nn.Sequential(
			nn.Dropout(p=dropout), nn.Linear(768, num_classes)
		)

	def forward(self, x):
		# shape: [batch_size, channels, num_frames, height, width]
		return self.mvit(x)


class DatasetWithSampleMeta(torch.utils.data.Dataset):
	"""Attach stable sample ids without changing the underlying dataset behavior."""

	def __init__(self, dataset, metadata):
		self.dataset = dataset
		self.sample_meta = []
		with open(metadata, newline='') as handle:
			reader = csv.DictReader(handle)
			for row_index, row in enumerate(reader):
				self.sample_meta.append({
					'narration_id': row.get('narration_id', ''),
					'narration_row_index': row_index,
					'video_id': row.get('video_id', ''),
				})
		if len(self.sample_meta) != len(self.dataset):
			raise RuntimeError(
				f"Metadata/sample length mismatch: metadata={len(self.sample_meta)} samples={len(self.dataset)}"
			)

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index):
		frames, label = self.dataset[index]
		return frames, label, self.sample_meta[index]

def get_args_parser():
	parser = argparse.ArgumentParser(description='lavila finetune and evaluation', add_help=False)
	# Data
	parser.add_argument('--dataset', default='ek100_cls', type=str,
						choices=['ek100_cls', 'egtea'])
	parser.add_argument('--egtea_finetune_type', default='action', type=str,
						help='finetune with action, verb or noun')
	parser.add_argument('--model_type', default='lavila', type=str,
						help='lavila or mvit')
	parser.add_argument('--root',
						default='datasets/EK100/video_ht256px/',
						type=str, help='path to dataset root')
	parser.add_argument('--metadata-train',
						default='datasets/EK100/epic-kitchens-100-annotations/EPIC_100_train.csv',
						type=str, help='path to metadata file (train set)')
	parser.add_argument('--metadata-val',
						default='datasets/EK100/epic-kitchens-100-annotations/EPIC_100_validation.csv',
						type=str, help='path to metadata file (val set)')
	parser.add_argument('--relevancy-path',
						default='datasets/EK100/epic-kitchens-100-annotations/retrieval_annotations/relevancy/caption_relevancy_EPIC_100_retrieval_test.pkl',
						type=str, help='path to relevancy matrix (val set)')
	parser.add_argument('--output-dir', default='./output', type=str, help='output dir')
	parser.add_argument('--num-crops', default=1, type=int, help='number of crops in transforms for val')
	parser.add_argument('--num-clips', default=1, type=int, help='number of clips for val')
	parser.add_argument('--clip-length', default=16, type=int, help='clip length')
	parser.add_argument('--clip-stride', default=2, type=int, help='clip stride')
	parser.add_argument('--sparse-sample', action='store_true', help='switch to sparse sampling')
	# Model
	parser.add_argument('--pretrain-model', default='', type=str, help='path to pretrain model')
	parser.add_argument('--resume', default='', type=str, help='path to resume from')
	parser.add_argument('--find-unused-parameters', action='store_true',
						help='do this during DDP (useful for models with tied weights)')
	parser.add_argument('--drop-path-rate', default=0.1, type=float, help='drop path ratio')
	parser.add_argument('--dropout-ratio', default=0.5, type=float, help='dropout ratio for the last linear layer')
	parser.add_argument('--num-classes', default=3806, nargs='+', type=int, help='number of classes for the last linear layer')
	parser.add_argument('--use-vn-classifier', action='store_true')
	parser.add_argument('--use-half', action='store_true', help='use half precision at inference')
	# Training
	parser.add_argument('--epochs', default=100, type=int)
	parser.add_argument('--warmup-epochs', default=1, type=int)
	parser.add_argument('--start-epoch', default=0, type=int)
	parser.add_argument('--batch-size', default=4, type=int,
						help='number of samples per-device/per-gpu')
	parser.add_argument('--use-sgd', action='store_true')
	parser.add_argument('--freeze-temperature', action='store_true', help='freeze temperature if set to True')
	parser.add_argument('--lr', default=3e-3, type=float)
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
	parser.add_argument('--use-zero', action='store_true',
						help='use ZeroRedundancyOptimizer to save memory')
	parser.add_argument('--use-checkpoint', action='store_true',
						help='use gradient checkpointing during training for significantly less GPU usage')
	# System
	parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
	parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
						help='number of data loading workers per process')
	parser.add_argument('--evaluate', action='store_true', help='eval only')
	parser.add_argument('--world-size', default=1, type=int,
						help='number of nodes for distributed training')
	parser.add_argument('--rank', default=0, type=int,
						help='node rank for distributed training')
	parser.add_argument("--local_rank", type=int, default=0)
	parser.add_argument('--dist-url', default='env://', type=str,
						help='url used to set up distributed training')
	parser.add_argument('--dist-backend', default='nccl', type=str)
	parser.add_argument('--seed', default=0, type=int)
	parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
	parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')
	return parser


def main(args):
	dist_utils.init_distributed_mode(args)

	global best_acc1
	random_seed(args.seed, dist_utils.get_rank())
	print('seed', args.seed)

	if args.pretrain_model:
		ckpt_path = args.pretrain_model
	else:
		raise Exception('no checkpoint found')
	ckpt = load_checkpoint(ckpt_path)

	if args.use_vn_classifier:
		assert args.dataset == 'ek100_cls' and len(args.num_classes) == 3

	state_dict = OrderedDict()
	for k, v in ckpt['state_dict'].items():
		state_dict[k.replace('module.', '')] = v

	
	if args.model_type == 'lavila':
		old_args = ckpt['args']
		print("=> creating model: {}".format(old_args.model))
		#print(old_args.model)
		print()
		model = getattr(models, old_args.model)(
			pretrained=old_args.load_visual_pretrained,
			pretrained2d=old_args.load_visual_pretrained is not None,
			text_use_cls_token=old_args.use_cls_token,
			project_embed_dim=old_args.project_embed_dim,
			timesformer_gated_xattn=False,
			timesformer_freeze_space=False,
			num_frames=args.clip_length,
			drop_path_rate=args.drop_path_rate,
		)
		if 'TIMESFORMER' in old_args.model or 'EGOVLP' in old_args.model:
			# inflate weight
			print('=> inflating PE in models due to different frame numbers')
			state_dict = inflate_positional_embeds(
				model.state_dict(), state_dict,
				num_frames=args.clip_length,
				load_temporal_fix='bilinear',
			)
		model.load_state_dict(state_dict, strict=True)
		print("=> loaded resume checkpoint '{}' (epoch {})".format(ckpt_path, ckpt['epoch']))
		

		if args.use_vn_classifier:
			model = models.VideoClassifierMultiHead(
				model.visual,
				dropout=args.dropout_ratio,
				num_classes_list=args.num_classes
			)
		else:
			assert len(args.num_classes) == 1
			model = models.VideoClassifier(
				model.visual,
				dropout=args.dropout_ratio,
				num_classes=args.num_classes[0]
			)
	if args.model_type == 'mvit':
		model = MViT_Spatial(args.num_classes[0])	
		#print(model.mvit.blocks[-1].mlp)
		
		# Load the pretrained weights for MViT model
		print(f"Loading MViT pretrained weights from: {args.pretrain_model}")
		model.load_state_dict(state_dict, strict=False)
		print("=> loaded MViT pretrained model")
		
	print(model)
	model.cuda(args.gpu)
	

	if args.distributed:
		model = torch.nn.parallel.DistributedDataParallel(
			model, device_ids=[args.gpu], bucket_cap_mb=200,
			find_unused_parameters=args.find_unused_parameters
		)

	p_wd, p_non_wd = [], []
	p_head_wd, p_head_non_wd = [], []
	for n, p in model.named_parameters():
		if 'fc_cls' in n:
			if 'bias' in n:
				p_head_non_wd.append(p)
			else:
				p_head_wd.append(p)
		elif not p.requires_grad:
			continue  # frozen weights
		elif p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
			p_non_wd.append(p)
		else:
			p_wd.append(p)

	optim_params = [
		{"params": p_wd, "weight_decay": args.wd,  "lr": args.lr * args.lr_multiplier_on_backbone},
		{"params": p_non_wd, "weight_decay": 0, "lr": args.lr * args.lr_multiplier_on_backbone},
		{"params": p_head_wd, "weight_decay": args.wd},
		{"params": p_head_non_wd, "weight_decay": 0}
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
	scaler = amp.GradScaler(enabled=not args.disable_amp)
	# optionally resume from a checkpoint (takes precedence over autoresume)
	latest = os.path.join(args.output_dir, 'checkpoint.pt')
	if os.path.isfile(latest):
		args.resume = ''
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading resume checkpoint '{}'".format(args.resume))
			checkpoint = load_checkpoint(args.resume)
			epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
			args.start_epoch = epoch
			if not args.distributed:
				state_dict = OrderedDict()
				for k, v in checkpoint['state_dict'].items():
					state_dict[k.replace('module.', '')] = v
				result = model.load_state_dict(state_dict, strict=False)
			else:
				result = model.load_state_dict(checkpoint['state_dict'], strict=False)
			print(result)
			#print(checkpoint['optimizer'])
			#optimizer.load_state_dict(checkpoint['optimizer']) if 'optimizer' in checkpoint else ()
			scaler.load_state_dict(checkpoint['scaler']) if 'scaler' in checkpoint else ()
			best_acc1 = checkpoint.get('best_acc1', 0.0)  # Use .get() for safetensors compatibility
			print("=> loaded resume checkpoint '{}' (epoch {}, best_metric = {})"
				  .format(args.resume, epoch, best_acc1))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))
	else:
		# auto-resume from latest checkpoint in output directory
		latest = os.path.join(args.output_dir, 'checkpoint.pt')
		if os.path.isfile(latest):
			print("=> loading latest checkpoint '{}'".format(latest))
			latest_checkpoint = load_checkpoint(latest)
			args.start_epoch = latest_checkpoint.get('epoch', 0)
			model.load_state_dict(latest_checkpoint['state_dict'])
			if 'optimizer' in latest_checkpoint:
				optimizer.load_state_dict(latest_checkpoint['optimizer'])
			if 'scaler' in latest_checkpoint:
				scaler.load_state_dict(latest_checkpoint['scaler'])
			best_acc1 = latest_checkpoint.get('best_acc1', 0.0)
			print("=> loaded latest checkpoint '{}' (epoch {})"
				  .format(latest, latest_checkpoint.get('epoch', 0)))

	cudnn.benchmark = True

	if args.model_type == 'lavila':
		# Data loading code
		print("=> creating dataset")
		if old_args.model.endswith('DISTILBERT_BASE'):
			tokenizer = MyDistilBertTokenizer('distilbert-base-uncased')
		elif old_args.model.endswith('BERT_BASE'):
			tokenizer = MyBertTokenizer('bert-base-uncased')
		elif old_args.model.endswith('BERT_LARGE'):
			tokenizer = MyBertTokenizer('bert-large-uncased')
		elif old_args.model.endswith('GPT2'):
			tokenizer = MyGPT2Tokenizer('gpt2')
		elif old_args.model.endswith('GPT2_MEDIUM'):
			tokenizer = MyGPT2Tokenizer('gpt2-medium')
		elif old_args.model.endswith('GPT2_LARGE'):
			tokenizer = MyGPT2Tokenizer('gpt2-large')
		elif old_args.model.endswith('GPT2_XL'):
			tokenizer = MyGPT2Tokenizer('gpt2-xl')
		else:
			print("Using SimpleTokenizer because of model '{}'. "
				  "Please check if this is what you want".format(old_args.model))
			tokenizer = SimpleTokenizer()
	else:
		tokenizer = SimpleTokenizer()

	criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).cuda(args.gpu)

	if args.model_type == 'lavila':
		crop_size = 224 if '336PX' not in old_args.model else 336
	if args.model_type == 'mvit':
		crop_size = 224
		
	transforms_list = [
		Permute([3, 0, 1, 2]),	# T H W C -> C T H W
		transforms.RandomResizedCrop(crop_size, scale=(0.5, 1.0)),
		transforms.RandomHorizontalFlip(p=0.5),
	]
	if args.model_type == 'lavila':
		if 'OPENAI' in old_args.model:
			transforms_list.append(transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305]))
		else:
			transforms_list.append(transforms_video.NormalizeVideo(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]))
	if args.model_type == 'mvit':
		transforms_list.append(transforms_video.NormalizeVideo(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]))
	train_transform = transforms.Compose(transforms_list)

	if args.model_type == 'lavila':
		val_transform = transforms.Compose([
				Permute([3, 0, 1, 2]),	# T H W C -> C T H W
				transforms.Resize(crop_size),
				transforms.CenterCrop(crop_size),
				(transforms_video.NormalizeVideo(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]) if 'OPENAI' not in old_args.model else
				 transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305])),
				TemporalCrop(frames_per_clip=args.clip_length, stride=args.clip_length),
				SpatialCrop(crop_size=crop_size, num_crops=args.num_crops),
			])
	if args.model_type == 'mvit':
		val_transform = transforms.Compose([
				Permute([3, 0, 1, 2]),	# T H W C -> C T H W
				transforms.Resize(crop_size),
				transforms.CenterCrop(crop_size),
				#(transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305])),
				(transforms_video.NormalizeVideo(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])),
				TemporalCrop(frames_per_clip=args.clip_length, stride=args.clip_length),
				SpatialCrop(crop_size=crop_size, num_crops=args.num_crops),
			])
	#train_transform = val_transform

	# build dataset
	_, mapping_vn2act = generate_label_map(args.dataset, args)
	if args.dataset == 'ek100_cls':
		args.mapping_act2v = {i: int(vn.split(':')[0]) for (vn, i) in mapping_vn2act.items()}
		args.mapping_act2n = {i: int(vn.split(':')[1]) for (vn, i) in mapping_vn2act.items()}
		args.actions = pd.DataFrame.from_dict({'verb': args.mapping_act2v.values(), 'noun': args.mapping_act2n.values()})
	num_clips_at_val = args.num_clips
	print('num_clips', args.num_clips)
	#if args.egtea_finetune_type == 'action':
	#	args.num_clips = 1
	args.num_clips = 1
	args.num_crops = 3
	print('=> build dataset')
	train_dataset = datasets.get_downstream_dataset_extract(
		train_transform, tokenizer, args, subset='train', label_mapping=mapping_vn2act,
	)
	args.num_clips = num_clips_at_val
	args.num_crops = 1  # Changed from 3 to 1 to match training
	val_dataset = datasets.get_downstream_dataset_extract(
		val_transform, tokenizer, args, subset='val', label_mapping=mapping_vn2act,
	)
	if args.dataset == 'ek100_cls':
		train_dataset = DatasetWithSampleMeta(train_dataset, args.metadata_train)
		val_dataset = DatasetWithSampleMeta(val_dataset, args.metadata_val)
	print('=> build dataset done')

	if args.distributed:
		train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
		val_sampler = torch.utils.data.SequentialSampler(val_dataset)  # disable distributed
	else:
		train_sampler = None
		val_sampler = None

	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=False
	)
	print('len(train_loader) = {}'.format(len(train_loader)))
	val_loader = torch.utils.data.DataLoader(
		val_dataset, batch_size=args.batch_size, shuffle=False,
		#num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False
		num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False
	)
	print('len(val_loader) = {}'.format(len(val_loader)))
	#print(model)

	if args.fix_lr:
		lr_schedule = None
	else:
		lr_schedule = cosine_scheduler(
			args.lr, args.lr_end, args.epochs, len(train_loader) // args.update_freq,
			warmup_epochs=args.warmup_epochs, start_warmup_value=args.lr_start,
		)

	if dist_utils.is_main_process() and args.wandb:
		wandb_id = os.path.split(args.output_dir)[-1]
		wandb.init(project='LaViLa', id=wandb_id, config=args, resume='allow')

	print(args)

	best_metric = 0.
	print("=> beginning training")
	epoch = 0  # Set epoch to 0 for feature extraction

	if args.distributed:
		train_sampler.set_epoch(epoch)

	train_extract(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args)
	validate_extract(val_loader, model, args)

	

def train_extract(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args):
	batch_time = AverageMeter('Time', ':6.2f')
	data_time = AverageMeter('Data', ':6.2f')
	mem = AverageMeter('Mem (GB)', ':6.1f')
	iters_per_epoch = len(train_loader) // args.update_freq
	losses = AverageMeter('Loss', ':.4e')
	top1 = AverageMeter('Acc@1', ':6.2f')
	top5 = AverageMeter('Acc@5', ':6.2f')
	top1_noun = AverageMeter('Noun Acc@1', ':6.2f')
	top1_verb = AverageMeter('Verb Acc@1', ':6.2f')
	progress = ProgressMeter(
		iters_per_epoch,
		[batch_time, data_time, mem, losses, top1, top5, top1_noun, top1_verb],
		prefix="Epoch: [{}]".format(epoch))

	# switch to train mode
	model.train()
	hook_model = model.module if hasattr(model, 'module') else model
	if args.model_type == 'mvit':
		activation = {}
		def getActivation(name):
			# the hook signature
			def hook(model, input, output):
				activation[name] = output.detach()
			return hook
		h1 = hook_model.mvit.blocks[-1].mlp.register_forward_hook(getActivation('mlp'))
	else:
		activation = {}
		def getActivation(name):
			# the hook signature
			def hook(model, input, output):
				activation[name] = output.detach()
			return hook
		h1 = hook_model.visual.blocks[-1].norm3.register_forward_hook(getActivation('norm3'))

	end = time.time()
	total_feat = []
	total_cls_feat = []
	total_target = []
	total_output = []
	total_verb_output = []
	total_noun_output = []
	all_narration_ids = []
	all_narration_row_index = []
	all_video_ids = []
	with torch.no_grad():
		for data_iter, batch_data in enumerate(train_loader):
			if len(batch_data) == 3:
				images, target, sample_meta = batch_data
				all_narration_ids.extend(list(sample_meta['narration_id']))
				row_index_value = sample_meta['narration_row_index']
				if torch.is_tensor(row_index_value):
					all_narration_row_index.append(row_index_value.detach().cpu())
				else:
					all_narration_row_index.append(torch.as_tensor(row_index_value))
				all_video_ids.extend(list(sample_meta['video_id']))
			else:
				images, target = batch_data
			#print(type(images), images.shape)
			if isinstance(images, list):
				with amp.autocast(enabled=not args.disable_amp):
					logit_allcrops = []
					feat_allcrops = []
					cls_feat_allcrops = []
					verb_logit_allcrops = []
					noun_logit_allcrops = []
					for crop in images:
						crop = crop.cuda(args.gpu, non_blocking=True)
						if args.use_half:
							crop = crop.half()
						if args.model_type == 'lavila':
							logit, cls_feat = model(crop, use_checkpoint=args.use_checkpoint)
							logit, aux_outputs = split_classifier_outputs(logit)
							#feat = activation['norm3'][:,1:,:].view(-1,8,49,768)
							feat = activation['norm3'][:,1:,:][:, ::256, :]
							if 'verb_outputs' in aux_outputs:
								verb_logit_allcrops.append(aux_outputs['verb_outputs'].unsqueeze(1).detach().cpu())
								noun_logit_allcrops.append(aux_outputs['noun_outputs'].unsqueeze(1).detach().cpu())
							
						if args.model_type == 'mvit':
							logit = model(crop)
							cls_feat = activation['mlp'][:,0,:]
							feat = activation['mlp'][:,1:,:].view(-1,8,49,768)
							feat = torch.mean(feat, dim=2)

						logit_allcrops.append(logit.unsqueeze(1).detach().cpu())
						feat_allcrops.append(feat.unsqueeze(1).detach().cpu())
						cls_feat_allcrops.append(cls_feat.unsqueeze(1).detach().cpu())
					logit_allcrops = torch.cat(logit_allcrops, 1)
					#logit = logit_allcrops.mean(0)
					#logit = torch.softmax(logit, dim=1)
					target = target.cuda(args.gpu, non_blocking=True)
					cls_feat_allcrops = torch.cat(cls_feat_allcrops, 1)
					feat_allcrops = torch.cat(feat_allcrops, 1)
					#print('feat_allcrops', feat_allcrops.shape)
					#img_feat = feat_allcrops.mean(0)
				total_feat.append(feat_allcrops.detach().cpu())
				total_cls_feat.append(cls_feat_allcrops.detach().cpu())
				total_output.append(logit_allcrops.detach().cpu())
				if len(verb_logit_allcrops) > 0:
					total_verb_output.append(torch.cat(verb_logit_allcrops, 1))
					total_noun_output.append(torch.cat(noun_logit_allcrops, 1))
				total_target.append(target.detach().cpu())
			else:
				images = images.cuda(args.gpu, non_blocking=True)
				target = target.cuda(args.gpu, non_blocking=True)

				# compute output
				with amp.autocast(enabled=not args.disable_amp):
					if args.model_type == 'lavila':
						output, cls_feat = model(images, use_checkpoint=args.use_checkpoint)
						output, aux_outputs = split_classifier_outputs(output)
						img_feat = activation['norm3'][:,1:,:][:, ::256, :]
						total_cls_feat.append(cls_feat.detach().cpu())
						if 'verb_outputs' in aux_outputs:
							total_verb_output.append(aux_outputs['verb_outputs'].detach().cpu())
							total_noun_output.append(aux_outputs['noun_outputs'].detach().cpu())

					if args.model_type == 'mvit':
						output = model(images)						
						cls_feat = activation['mlp'][:,0,:]
						img_feat = activation['mlp'][:,1:,:].view(-1,8,49,768)
						img_feat = torch.mean(img_feat, dim=2)
						total_cls_feat.append(cls_feat.detach().cpu())
					
				#print(img_feat.shape)
			
				total_feat.append(img_feat.detach().cpu())
				total_output.append(output.detach().cpu())
				total_target.append(target.detach().cpu())
				
				
			batch_time.update(time.time() - end)
			end = time.time()

			if data_iter % args.print_freq == 0:
				progress.display(data_iter)
	if args.model_type == 'mvit':
		h1.remove()		  
	total_feat = torch.cat(total_feat)
	print('total_feat', total_feat.shape)
	total_output = torch.cat(total_output)
	print('total_output', total_output.shape)
	total_target = torch.cat(total_target)
	print('total_target', total_target.shape)
	save_payload = {
		'feats': total_feat,
		'cls_feats': torch.cat(total_cls_feat),
		'outputs': total_output,
		'targets': total_target,
	}
	if len(total_verb_output) > 0:
		save_payload['verb_outputs'] = torch.cat(total_verb_output)
		save_payload['noun_outputs'] = torch.cat(total_noun_output)
	if len(all_narration_ids) > 0:
		save_payload['narration_ids'] = all_narration_ids
		save_payload['narration_row_index'] = torch.cat([value.view(-1) for value in all_narration_row_index], dim=0)
		save_payload['video_ids'] = all_video_ids
	save_path = os.path.join(args.output_dir, f'{args.dataset}_train_feat.pt')
	if args.model_type == 'mvit':
		torch.save(save_payload, save_path)
	if args.model_type == 'lavila':
		torch.save(save_payload, save_path)

def validate_extract(val_loader, model, args):
	batch_time = AverageMeter('Time', ':6.2f')
	data_time = AverageMeter('Data', ':6.2f')
	progress = ProgressMeter(
		len(val_loader),
		[batch_time],
		prefix='Test: '
	)
	
	hook_model = model.module if hasattr(model, 'module') else model
	if args.model_type == 'mvit':
		activation = {}
		def getActivation(name):
		  # the hook signature
			def hook(model, input, output):
				activation[name] = output.detach()
			return hook
		h1 = hook_model.mvit.blocks[-1].mlp.register_forward_hook(getActivation('mlp'))
	else:
		activation = {}
		def getActivation(name):
			# the hook signature
			def hook(model, input, output):
				activation[name] = output.detach()
			return hook
		h1 = hook_model.visual.blocks[-1].norm3.register_forward_hook(getActivation('norm3'))

	# switch to eval mode
	model.eval()
	if args.use_half:
		model.half()

	all_outputs = []
	all_verb_outputs = []
	all_noun_outputs = []
	all_targets = []
	all_feats = []
	all_cls_feats = []
	all_narration_ids = []
	all_narration_row_index = []
	all_video_ids = []
	
	with torch.no_grad():
		end = time.time()
		for i, batch_data in enumerate(val_loader):
			if len(batch_data) == 3:
				images, target, sample_meta = batch_data
				all_narration_ids.extend(list(sample_meta['narration_id']))
				row_index_value = sample_meta['narration_row_index']
				if torch.is_tensor(row_index_value):
					all_narration_row_index.append(row_index_value.detach().cpu())
				else:
					all_narration_row_index.append(torch.as_tensor(row_index_value))
				all_video_ids.extend(list(sample_meta['video_id']))
			else:
				images, target = batch_data
			# measure data loading time
			#print(target)
			#print(type(images), len(images), images[0].shape)
			data_time.update(time.time() - end)
			if isinstance(images, list):
				logit_allcrops = []
				feat_allcrops = []
				cls_feat_allcrops = []
				verb_logit_allcrops = []
				noun_logit_allcrops = []
				for crop in images:
					crop = crop.cuda(args.gpu, non_blocking=True)
					if args.use_half:
						crop = crop.half()
						
					if args.model_type == 'lavila':
						logit, cls_feat = model(crop, use_checkpoint=args.use_checkpoint)
						logit, aux_outputs = split_classifier_outputs(logit)
						feat = activation['norm3'][:,1:,:][:, ::256, :]
						cls_feat_allcrops.append(cls_feat.unsqueeze(1).detach().cpu())
						if 'verb_outputs' in aux_outputs:
							verb_logit_allcrops.append(aux_outputs['verb_outputs'].unsqueeze(1).detach().cpu())
							noun_logit_allcrops.append(aux_outputs['noun_outputs'].unsqueeze(1).detach().cpu())
						
					if args.model_type == 'mvit':
						logit = model(crop)
						cls_feat = activation['mlp'][:,0,:]
						feat = activation['mlp'][:,1:,:].view(-1,8,49,768)
						feat = torch.mean(feat, dim=2)
						#print(feat.shape)
						cls_feat_allcrops.append(cls_feat.unsqueeze(1).detach().cpu())

					logit_allcrops.append(logit.unsqueeze(1).detach().cpu())
					feat_allcrops.append(feat.unsqueeze(1).detach().cpu())
					
				logit_allcrops = torch.cat(logit_allcrops, 1)
				#logit = logit_allcrops.mean(0)
				#logit = torch.softmax(logit, dim=1)
				target = target.cuda(args.gpu, non_blocking=True)				
				feat_allcrops = torch.cat(feat_allcrops, 1)

				#if args.model_type == 'mvit':
				cls_feat_allcrops = torch.cat(cls_feat_allcrops, 1)
				if len(verb_logit_allcrops) > 0:
					all_verb_outputs.append(torch.cat(verb_logit_allcrops, 1))
					all_noun_outputs.append(torch.cat(noun_logit_allcrops, 1))
				#feat = feat_allcrops.mean(0)
				
			else:
				images = images.cuda(args.gpu, non_blocking=True)
				target = target.cuda(args.gpu, non_blocking=True)
				if args.use_half:
					images = images.half()

				logit, feat = model(images, use_checkpoint=args.use_checkpoint)
				logit, _ = split_classifier_outputs(logit)
				logit = torch.softmax(logit, dim=1)

				acc1, acc5 = accuracy(logit, target, topk=(1, 5))
				top1.update(acc1.item(), images.size(0))
				top5.update(acc5.item(), images.size(0))

			all_outputs.append(logit_allcrops)
			all_targets.append(target)
			
			all_feats.append(feat_allcrops)
			#if args.model_type == 'mvit':
			all_cls_feats.append(cls_feat_allcrops)
			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % args.print_freq == 0:
				progress.display(i)
	#if args.model_type == 'mvit':
	h1.remove() 
	all_feats = torch.cat(all_feats)
	
	all_targets = torch.cat(all_targets)
	all_outputs = torch.cat(all_outputs)
	all_cls_feats = torch.cat(all_cls_feats)
	save_payload = {
		'feats': all_feats.detach().cpu(),
		'cls_feats': all_cls_feats.detach().cpu(),
		'outputs': all_outputs.detach().cpu(),
		'targets': all_targets.detach().cpu(),
	}
	if len(all_verb_outputs) > 0:
		save_payload['verb_outputs'] = torch.cat(all_verb_outputs).detach().cpu()
		save_payload['noun_outputs'] = torch.cat(all_noun_outputs).detach().cpu()
	if len(all_narration_ids) > 0:
		save_payload['narration_ids'] = all_narration_ids
		save_payload['narration_row_index'] = torch.cat([value.view(-1) for value in all_narration_row_index], dim=0)
		save_payload['video_ids'] = all_video_ids
	save_path = os.path.join(args.output_dir, f'{args.dataset}_test_feat.pt')

	if args.model_type == 'mvit':
		torch.save(save_payload, save_path)
	if args.model_type == 'lavila':
		torch.save(save_payload, save_path)



if __name__ == '__main__':
	parser = argparse.ArgumentParser('lavila finetune and evaluation', parents=[get_args_parser()])
	args = parser.parse_args()
	os.makedirs(args.output_dir, exist_ok=True)
	main(args)
