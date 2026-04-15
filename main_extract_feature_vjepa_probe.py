# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Feature extraction script for MViT and V-JEPA backbones.
Outputs follow the same format as the original extractor:
  {'feats': ..., 'cls_feats': ..., 'outputs': ..., 'targets': ...}
Attentive V-JEPA2 extraction also adds:
  {'temporal_probe_feats': ...}  # [B, T, D] for single-task, head-specific for multitask
Backbone motion extraction adds:
  {'verb_input_feats': ...}
"""

import argparse
from collections import OrderedDict
import csv
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
from lavila.data.datasets import VideoCaptionDatasetBase, get_frame_ids, video_loader_by_frames
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
VJEPA2_MOTION_LAYERS_1BASED = [25, 27, 29, 31]
VJEPA2_HIERARCHICAL_MOTION_MODELS = {
	'vjepa2_1_vit_giant_384',
	'vjepa2_1_vit_gigantic_384',
}


def _validate_motion_layer_indices(encoder, motion_layers, layer_source):
	if motion_layers is None:
		return None
	if len(motion_layers) == 0:
		raise ValueError(f"{layer_source} motion layer list is empty.")
	if hasattr(encoder, "get_num_layers"):
		num_layers = encoder.get_num_layers()
		if min(motion_layers) < 0 or max(motion_layers) >= num_layers:
			raise ValueError(
				f"{layer_source} motion layers {motion_layers} exceed encoder depth {num_layers}."
			)
	hierarchical_layers = list(getattr(encoder, "hierarchical_layers", []) or [])
	if hierarchical_layers:
		invalid = [layer for layer in motion_layers if layer not in hierarchical_layers]
		if invalid:
			raise ValueError(
				f"{layer_source} motion layers {motion_layers} are incompatible with encoder hierarchical "
				f"layers {hierarchical_layers}; invalid layers: {invalid}"
			)
	return motion_layers


def _resolve_motion_layer_indices(encoder, model_type: str, motion_layer_set: str = "auto"):
	motion_layer_set = (motion_layer_set or "auto").lower()
	hierarchical_layers = list(getattr(encoder, "hierarchical_layers", []) or [])
	legacy_layers = [idx - 1 for idx in VJEPA2_MOTION_LAYERS_1BASED]

	if motion_layer_set == "none":
		return None
	if motion_layer_set == "legacy":
		return _validate_motion_layer_indices(encoder, legacy_layers, "Legacy")
	if motion_layer_set == "hierarchical":
		if not hierarchical_layers:
			raise ValueError(
				f"Encoder for {model_type} does not expose hierarchical_layers; cannot use "
				"--vjepa2-motion-layer-set hierarchical."
			)
		return _validate_motion_layer_indices(encoder, hierarchical_layers, "Hierarchical")
	if motion_layer_set != "auto":
		raise ValueError(
			f"Unsupported motion layer set '{motion_layer_set}'. "
			"Choose from: auto, legacy, hierarchical, none."
		)

	try:
		return _validate_motion_layer_indices(encoder, legacy_layers, "Legacy")
	except ValueError as exc:
		if model_type in VJEPA2_HIERARCHICAL_MOTION_MODELS and hierarchical_layers:
			print(
				f"=> Switching {model_type} motion extraction to hierarchical layers "
				f"{hierarchical_layers} because legacy layers are incompatible ({exc})."
			)
			return _validate_motion_layer_indices(encoder, hierarchical_layers, "Hierarchical")
		return None


def _supports_motion_features(encoder, model_type: str, motion_layer_set: str = "auto"):
	if not hasattr(encoder, "out_layers"):
		return False
	try:
		return _resolve_motion_layer_indices(encoder, model_type, motion_layer_set) is not None
	except ValueError:
		return False


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


def _get_vjepa2_attentive_pooler(variant_key: str):
	repo_root = _ensure_vjepa2_repo_on_path(variant_key)
	try:
		from src.models.attentive_pooler import AttentivePooler
	except Exception as exc:
		raise RuntimeError(
			"Failed to import V-JEPA attentive pooler. "
			f"Ensure {repo_root} is present and on the Python path."
		) from exc
	return AttentivePooler


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


class VJEPA2MeanPoolClassifier(nn.Module):
	"""Mean-pool classifier on top of a V-JEPA2 encoder."""

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


class VJEPA2ProbeHead(nn.Module):
	"""Single-query attentive probe head for verb-only or noun-only extraction."""

	def __init__(
		self,
		variant_key: str,
		embed_dim: int,
		num_classes: int,
		num_heads: int,
		depth: int,
		mlp_ratio: float = 4.0,
		dropout: float = 0.0,
		use_activation_checkpointing: bool = True,
	):
		super().__init__()
		AttentivePooler = _get_vjepa2_attentive_pooler(variant_key)
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

	def forward(self, x, return_features: bool = False):
		pooled = self.pooler(x).squeeze(1)
		logits = self.classifier(self.dropout(pooled))
		if return_features:
			return logits, pooled
		return logits


class VJEPA2TemporalProbeHead(nn.Module):
	"""Single-query attentive probe that produces task-conditioned temporal features."""

	def __init__(
		self,
		variant_key: str,
		embed_dim: int,
		num_classes: int,
		num_heads: int,
		depth: int,
		mlp_ratio: float = 4.0,
		dropout: float = 0.0,
		use_activation_checkpointing: bool = True,
	):
		super().__init__()
		AttentivePooler = _get_vjepa2_attentive_pooler(variant_key)
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

	def forward(self, x, temporal_length: int, return_features: bool = False):
		temporal_feats = _temporal_query_pool(self.pooler, x, temporal_length)[:, :, 0, :]
		pooled = temporal_feats.mean(dim=1)
		logits = self.classifier(self.dropout(pooled))
		if return_features:
			return logits, temporal_feats, pooled
		return logits


class VJEPA2ProbeClassifier(nn.Module):
	"""Frozen-style V-JEPA2 encoder with a single attentive probe head."""

	def __init__(
		self,
		variant_key: str,
		num_classes: int,
		probe_num_heads: int = 16,
		probe_num_blocks: int = 4,
		probe_mlp_ratio: float = 4.0,
		probe_dropout: float = 0.0,
		use_activation_checkpointing: bool = True,
		pretrained_backbone: bool = True,
	):
		super().__init__()
		self.encoder = _load_vjepa2_encoder(variant_key, pretrained=pretrained_backbone)
		self.num_features = self.encoder.embed_dim
		self.probe = VJEPA2ProbeHead(
			variant_key=variant_key,
			embed_dim=self.num_features,
			num_classes=num_classes,
			num_heads=probe_num_heads,
			depth=probe_num_blocks,
			mlp_ratio=probe_mlp_ratio,
			dropout=probe_dropout,
			use_activation_checkpointing=use_activation_checkpointing,
		)

	def forward(self, x, return_features: bool = False):
		tokens = self.encoder(x)
		if return_features:
			logits, probe_feat = self.probe(tokens, return_features=True)
			return logits, tokens, probe_feat
		return self.probe(tokens)


class VJEPA2TemporalProbeClassifier(nn.Module):
	"""V-JEPA2 encoder with a temporal-output attentive probe head."""

	def __init__(
		self,
		variant_key: str,
		num_classes: int,
		probe_num_heads: int = 16,
		probe_num_blocks: int = 4,
		probe_mlp_ratio: float = 4.0,
		probe_dropout: float = 0.0,
		use_activation_checkpointing: bool = True,
		pretrained_backbone: bool = True,
	):
		super().__init__()
		self.encoder = _load_vjepa2_encoder(variant_key, pretrained=pretrained_backbone)
		self.num_features = self.encoder.embed_dim
		self.probe = VJEPA2TemporalProbeHead(
			variant_key=variant_key,
			embed_dim=self.num_features,
			num_classes=num_classes,
			num_heads=probe_num_heads,
			depth=probe_num_blocks,
			mlp_ratio=probe_mlp_ratio,
			dropout=probe_dropout,
			use_activation_checkpointing=use_activation_checkpointing,
		)

	def forward(self, x, return_features: bool = False):
		tokens = self.encoder(x)
		temporal_length = _get_token_temporal_length(self.encoder, x, x.shape[2] if x.ndim == 5 else 1)
		if return_features:
			logits, temporal_feats, pooled = self.probe(tokens, temporal_length, return_features=True)
			return logits, tokens, pooled, temporal_feats
		return self.probe(tokens, temporal_length)


class VJEPA2MultiTaskProbeHead(nn.Module):
	"""Attentive probe head with verb/noun/action classifiers."""

	def __init__(
		self,
		variant_key: str,
		embed_dim: int,
		num_verb_classes: int,
		num_noun_classes: int,
		num_action_classes: int,
		num_heads: int,
		depth: int,
		mlp_ratio: float = 4.0,
		dropout: float = 0.0,
		use_activation_checkpointing: bool = True,
	):
		super().__init__()
		AttentivePooler = _get_vjepa2_attentive_pooler(variant_key)
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

	def forward(self, x, return_features: bool = False):
		pooled = self.pooler(x)
		verb_feat, noun_feat, action_feat = pooled[:, 0, :], pooled[:, 1, :], pooled[:, 2, :]
		verb_logits = self.verb_classifier(self.dropout(verb_feat))
		noun_logits = self.noun_classifier(self.dropout(noun_feat))
		action_logits = self.action_classifier(self.dropout(action_feat))
		logits = dict(verb=verb_logits, noun=noun_logits, action=action_logits)
		if return_features:
			feats = dict(verb=verb_feat, noun=noun_feat, action=action_feat)
			return logits, feats
		return logits


class VJEPA2TemporalMultiTaskProbeHead(nn.Module):
	"""Temporal-output attentive probe head with verb/noun/action classifiers."""

	def __init__(
		self,
		variant_key: str,
		embed_dim: int,
		num_verb_classes: int,
		num_noun_classes: int,
		num_action_classes: int,
		num_heads: int,
		depth: int,
		mlp_ratio: float = 4.0,
		dropout: float = 0.0,
		use_activation_checkpointing: bool = True,
	):
		super().__init__()
		AttentivePooler = _get_vjepa2_attentive_pooler(variant_key)
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

	def forward(self, x, temporal_length: int, return_features: bool = False):
		temporal_feats = _temporal_query_pool(self.pooler, x, temporal_length)
		pooled = temporal_feats.mean(dim=1)
		verb_feat, noun_feat, action_feat = pooled[:, 0, :], pooled[:, 1, :], pooled[:, 2, :]
		verb_logits = self.verb_classifier(self.dropout(verb_feat))
		noun_logits = self.noun_classifier(self.dropout(noun_feat))
		action_logits = self.action_classifier(self.dropout(action_feat))
		logits = dict(verb=verb_logits, noun=noun_logits, action=action_logits)
		if return_features:
			clip_feats = dict(verb=verb_feat, noun=noun_feat, action=action_feat)
			temporal = dict(
				verb=temporal_feats[:, :, 0, :],
				noun=temporal_feats[:, :, 1, :],
				action=temporal_feats[:, :, 2, :],
			)
			return logits, clip_feats, temporal
		return logits


class VJEPA2MultiTaskProbeClassifier(nn.Module):
	"""V-JEPA2 encoder with multi-task attentive probes."""

	def __init__(
		self,
		variant_key: str,
		num_verb_classes: int,
		num_noun_classes: int,
		num_action_classes: int,
		probe_num_heads: int = 16,
		probe_num_blocks: int = 4,
		probe_mlp_ratio: float = 4.0,
		probe_dropout: float = 0.0,
		use_activation_checkpointing: bool = True,
		pretrained_backbone: bool = True,
	):
		super().__init__()
		self.encoder = _load_vjepa2_encoder(variant_key, pretrained=pretrained_backbone)
		self.num_features = self.encoder.embed_dim
		self.probe = VJEPA2MultiTaskProbeHead(
			variant_key=variant_key,
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

	def forward(self, x, return_features: bool = False):
		tokens = self.encoder(x)
		if return_features:
			logits, probe_feats = self.probe(tokens, return_features=True)
			return logits, tokens, probe_feats
		return self.probe(tokens)


class VJEPA2TemporalMultiTaskProbeClassifier(nn.Module):
	"""V-JEPA2 encoder with temporal-output multi-task attentive probes."""

	def __init__(
		self,
		variant_key: str,
		num_verb_classes: int,
		num_noun_classes: int,
		num_action_classes: int,
		probe_num_heads: int = 16,
		probe_num_blocks: int = 4,
		probe_mlp_ratio: float = 4.0,
		probe_dropout: float = 0.0,
		use_activation_checkpointing: bool = True,
		pretrained_backbone: bool = True,
	):
		super().__init__()
		self.encoder = _load_vjepa2_encoder(variant_key, pretrained=pretrained_backbone)
		self.num_features = self.encoder.embed_dim
		self.probe = VJEPA2TemporalMultiTaskProbeHead(
			variant_key=variant_key,
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

	def forward(self, x, return_features: bool = False):
		tokens = self.encoder(x)
		temporal_length = _get_token_temporal_length(self.encoder, x, x.shape[2] if x.ndim == 5 else 1)
		if return_features:
			logits, clip_feats, temporal_feats = self.probe(tokens, temporal_length, return_features=True)
			return logits, tokens, clip_feats, temporal_feats
		return self.probe(tokens, temporal_length)


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
	if clip_length > 0 and token_count > 1 and (token_count - 1) % clip_length == 0:
		patch_tokens = token_feats[:, 1:, :]
		tokens_per_frame = patch_tokens.shape[1] // clip_length
		return patch_tokens.view(
			token_feats.shape[0], clip_length, tokens_per_frame, token_feats.shape[-1]
		).mean(dim=2)
	return token_feats


def _reshape_tokens_to_temporal_grid(token_feats: torch.Tensor, clip_length: int):
	"""Reshape token features to [B, T, S, D] for temporal, head-conditioned pooling."""
	if clip_length <= 0:
		raise ValueError(f"clip_length must be > 0, got {clip_length}")
	token_count = token_feats.shape[1]
	if token_count % clip_length == 0:
		tokens_per_frame = token_count // clip_length
		return token_feats.reshape(token_feats.shape[0], clip_length, tokens_per_frame, token_feats.shape[-1])
	if token_count > 1 and (token_count - 1) % clip_length == 0:
		# Handle layouts with a leading global token.
		patch_tokens = token_feats[:, 1:, :]
		tokens_per_frame = patch_tokens.shape[1] // clip_length
		return patch_tokens.reshape(token_feats.shape[0], clip_length, tokens_per_frame, token_feats.shape[-1])
	raise RuntimeError(
		f"Cannot reshape token tensor of shape {tuple(token_feats.shape)} into [B, T, S, D] "
		f"with clip_length={clip_length}."
	)


def _get_token_temporal_length(encoder: nn.Module, inputs: torch.Tensor, clip_length: int):
	"""Infer temporal token length (tubelet-aware) for reshaping [B, N, D] tokens."""
	frame_count = clip_length
	if inputs.ndim == 5:
		frame_count = int(inputs.shape[2])
	tubelet_size = int(getattr(encoder, "tubelet_size", 1) or 1)
	if tubelet_size < 1:
		tubelet_size = 1
	return max(1, frame_count // tubelet_size)


def _apply_pooler_sequence_blocks(pooler: nn.Module, token_feats: torch.Tensor):
	"""Apply AttentivePooler sequence blocks over the full token sequence."""
	seq_tokens = token_feats
	blocks = getattr(pooler, "blocks", None)
	if blocks is not None:
		for blk in blocks:
			seq_tokens = blk(seq_tokens)
	return seq_tokens


def _temporal_query_pool(pooler: nn.Module, token_feats: torch.Tensor, clip_length: int):
	"""Apply the probe queries to each frame after full-sequence refinement."""
	if pooler is None:
		raise RuntimeError("Probe pooler is missing; cannot extract temporal probe features.")
	if not hasattr(pooler, "query_tokens") or not hasattr(pooler, "cross_attention_block"):
		raise RuntimeError("Unexpected pooler implementation; query_tokens/cross_attention_block not found.")

	seq_tokens = _apply_pooler_sequence_blocks(pooler, token_feats)
	token_grid = _reshape_tokens_to_temporal_grid(seq_tokens, clip_length)  # [B, T, S, D]
	batch_size, temporal_dim, tokens_per_frame, feat_dim = token_grid.shape
	frame_tokens = token_grid.reshape(batch_size * temporal_dim, tokens_per_frame, feat_dim)
	query_count = pooler.query_tokens.shape[1]
	queries = pooler.query_tokens.repeat(frame_tokens.shape[0], 1, 1)
	pooled = pooler.cross_attention_block(queries, frame_tokens)  # [B*T, Q, D]
	return pooled.reshape(batch_size, temporal_dim, query_count, feat_dim)


def extract_probe_temporal_features(pooler: nn.Module, token_feats: torch.Tensor, clip_length: int, head_names=None):
	"""
	Extract attentive probe temporal features [B, T, D] or head-wise temporal
	features by applying the probe's learned query tokens to each frame's spatial tokens.
	"""
	pooled = _temporal_query_pool(pooler, token_feats, clip_length)
	batch_size, temporal_dim, query_count, feat_dim = pooled.shape
	if head_names is None:
		if query_count == 1:
			head_names = ("probe",)
		elif query_count == 3:
			head_names = ("verb", "noun", "action")
		else:
			raise RuntimeError(f"Unsupported query count {query_count}; provide explicit head_names.")
	if query_count != len(head_names):
		raise RuntimeError(
			f"Probe query count {query_count} does not match requested head names {head_names}."
		)

	if query_count == 1:
		return pooled[:, :, 0, :]
	return {name: pooled[:, :, idx, :] for idx, name in enumerate(head_names)}


def extract_motion_features(encoder: nn.Module, inputs: torch.Tensor, clip_length: int, motion_layers=None):
	"""Extract concatenated motion-layer tokens and pool spatially to [B, T, D]."""
	if not hasattr(encoder, "out_layers"):
		raise RuntimeError("Encoder does not expose out_layers for intermediate extraction.")
	if motion_layers is None:
		raise RuntimeError("Motion layer indices must be resolved before extracting motion features.")
	prev_out_layers = encoder.out_layers
	encoder.out_layers = motion_layers
	try:
		outs = encoder(inputs)
	finally:
		encoder.out_layers = prev_out_layers
	if not isinstance(outs, (list, tuple)) or len(outs) != len(motion_layers):
		raise RuntimeError("Unexpected intermediate layer outputs from encoder.")
	motion_tokens = torch.cat(outs, dim=-1)
	return reshape_temporal_features(motion_tokens, clip_length)


def build_ek100_multitask_label_maps(train_csv: str):
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


class EK100MultiTaskDataset(VideoCaptionDatasetBase):
	def __init__(
		self,
		args,
		root: str,
		metadata: str,
		transform=None,
		is_training: bool = True,
		label_maps=None,
		filter_actions: bool = False,
		clip_length: int = 32,
		clip_stride: int = 2,
		sparse_sample: bool = False,
	):
		super().__init__(args, "ek100_cls", root, metadata)
		self.transform = transform
		self.is_training = is_training
		self.label_maps = label_maps
		self.clip_length = clip_length
		self.clip_stride = clip_stride
		self.sparse_sample = sparse_sample
		self.sample_meta = []
		with open(metadata, newline='') as handle:
			reader = csv.DictReader(handle)
			for row_index, row in enumerate(reader):
				self.sample_meta.append({
					'narration_id': row.get('narration_id', ''),
					'narration_row_index': row_index,
					'video_id': row.get('video_id', ''),
				})
		if len(self.sample_meta) != len(self.samples):
			raise RuntimeError(
				f"Metadata/sample length mismatch: metadata={len(self.sample_meta)} samples={len(self.samples)}"
			)
		if filter_actions:
			print('=> filter_actions requested but disabled; keeping all validation samples for alignment with LaViLa')

	def __getitem__(self, i):
		vid_path, start_frame, end_frame, _, verb, noun = self.samples[i]
		frame_ids = get_frame_ids(start_frame, end_frame, num_segments=self.clip_length, jitter=self.is_training)
		frames = video_loader_by_frames(self.root, vid_path, frame_ids)

		if self.transform is not None:
			frames = self.transform(frames)

		verb_key = str(verb)
		noun_key = str(noun)
		action_key = f"{verb_key}:{noun_key}"
		# Keep raw EK100 verb/noun ids so extraction works on the full validation set.
		verb_label = int(verb_key)
		noun_label = int(noun_key)
		action_label = self.label_maps["action"].get(action_key, -1)
		return frames, verb_label, noun_label, action_label, self.sample_meta[i]


class EK100ExtractDataset(VideoCaptionDatasetBase):
	"""Single-task EK100 extractor dataset that carries stable sample ids."""

	def __init__(
		self,
		args,
		root: str,
		metadata: str,
		transform,
		is_training: bool,
		label_mapping,
		clip_length: int = 32,
		clip_stride: int = 2,
		sparse_sample: bool = False,
	):
		super().__init__(args, "ek100_cls", root, metadata)
		self.transform = transform
		self.is_training = is_training
		self.label_mapping = label_mapping
		self.clip_length = clip_length
		self.clip_stride = clip_stride
		self.sparse_sample = sparse_sample
		self.sample_meta = []
		with open(metadata, newline='') as handle:
			reader = csv.DictReader(handle)
			for row_index, row in enumerate(reader):
				self.sample_meta.append({
					'narration_id': row.get('narration_id', ''),
					'narration_row_index': row_index,
					'video_id': row.get('video_id', ''),
				})
		if len(self.sample_meta) != len(self.samples):
			raise RuntimeError(
				f"Metadata/sample length mismatch: metadata={len(self.sample_meta)} samples={len(self.samples)}"
			)

	def __getitem__(self, i):
		vid_path, start_frame, end_frame, _, verb, noun = self.samples[i]
		frame_ids = get_frame_ids(start_frame, end_frame, num_segments=self.clip_length, jitter=self.is_training)
		frames = video_loader_by_frames(self.root, vid_path, frame_ids)

		if self.transform is not None:
			frames = self.transform(frames)

		if self.args.task_type == 'verb':
			label = str(verb)
		elif self.args.task_type == 'noun':
			label = str(noun)
		else:
			label = f'{verb}:{noun}'

		if self.label_mapping is not None:
			label = self.label_mapping[label]

		return frames, label, self.sample_meta[i]


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
	"""Run model forward pass and return logits, token features and auxiliary features."""
	temporal_probe_feat = None
	verb_input_feat = None
	if model_type in VJEPA2_MODEL_SPECS:
		model_without_ddp = model.module if hasattr(model, 'module') else model
		encoder = getattr(model_without_ddp, 'encoder', None)
		if encoder is None:
			raise RuntimeError("V-JEPA2 model is missing encoder; cannot extract features.")
		token_temporal_length = _get_token_temporal_length(encoder, inputs, args.clip_length)
		if args.multi_task:
			if getattr(args, 'vjepa2_head', 'attentive') == 'temporal_attentive':
				logits, tokens, probe_feats, temporal_probe_feat = model(inputs, return_features=True)
			else:
				logits, tokens, probe_feats = model(inputs, return_features=True)
			feat = reshape_temporal_features(tokens, token_temporal_length)
			cls_feat = probe_feats
			if temporal_probe_feat is None:
				pooler = getattr(getattr(model_without_ddp, 'probe', None), 'pooler', None)
				temporal_probe_feat = extract_probe_temporal_features(
					pooler, tokens, token_temporal_length, head_names=("verb", "noun", "action")
				)
		else:
			if getattr(args, 'vjepa2_head', 'attentive') == 'temporal_attentive':
				logits, tokens, pooled, temporal_probe_feat = model(inputs, return_features=True)
			else:
				logits, tokens, pooled = model(inputs, return_features=True)
			feat = reshape_temporal_features(tokens, token_temporal_length)
			cls_feat = pooled
			if temporal_probe_feat is None and getattr(args, 'vjepa2_head', 'attentive') == 'attentive' and hasattr(model_without_ddp, 'probe'):
				pooler = getattr(model_without_ddp.probe, 'pooler', None)
				temporal_probe_feat = extract_probe_temporal_features(pooler, tokens, token_temporal_length)
			motion_layers = _resolve_motion_layer_indices(
				encoder,
				model_type,
				getattr(args, 'vjepa2_motion_layer_set', 'auto'),
			)
			if motion_layers is not None:
				verb_input_feat = extract_motion_features(
					encoder,
					inputs,
					token_temporal_length,
					motion_layers=motion_layers,
				)
	else:
		logits = model(inputs)
		token_output = activation['tokens']
		cls_feat = token_output[:, 0, :]
		patch_tokens = token_output[:, 1:, :]
		feat = reshape_temporal_features(patch_tokens, args.clip_length)
	return logits, feat, cls_feat, temporal_probe_feat, verb_input_feat


def extract_split(loader, model, flow_model, args, subset: str, head: str = None):
	"""Extract features for a single split and save to disk."""
	if args.multi_task:
		if head is None:
			raise ValueError("Multi-task extraction requires a head name.")
		if head not in {"verb", "noun", "action"}:
			raise ValueError(f"Unknown multi-task head: {head}")
		if args.model_type == 'mvit_temporal':
			raise ValueError("Multi-task extraction is not supported for temporal flow models.")
	batch_time = AverageMeter('Time', ':6.2f')
	data_time = AverageMeter('Data', ':6.2f')
	if args.multi_task:
		progress_prefix = f'{subset}-{head}: '
	else:
		progress_prefix = f'{subset}: '
	progress = ProgressMeter(len(loader), [batch_time, data_time], prefix=progress_prefix)

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
	save_temporal_probe = args.multi_task or (
		args.model_type in VJEPA2_MODEL_SPECS and getattr(args, 'vjepa2_head', 'attentive') in {'attentive', 'temporal_attentive'}
	)
	all_temporal_probe_feats = [] if save_temporal_probe else None
	all_verb_input_feats = [] if args.model_type in VJEPA2_MODEL_SPECS else None
	all_narration_ids = []
	all_narration_row_index = []
	all_video_ids = []
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
						logits, feat, cls_feat, _, _ = forward_with_features(
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
					logits, feat, cls_feat, _, _ = forward_with_features(
						model, args.model_type, activation, flow_input, args
					)
					all_outputs.append(logits.detach().cpu())
					all_feats.append(feat.detach().cpu())
					all_cls_feats.append(cls_feat.detach().cpu())
				all_targets.append(target_gpu.detach().cpu())
			else:
				if args.multi_task:
					images, verb_target, noun_target, action_target, sample_meta = batch_data
					target_map = {
						"verb": verb_target,
						"noun": noun_target,
						"action": action_target,
					}
					target = target_map[head]
					all_narration_ids.extend(list(sample_meta["narration_id"]))
					row_index_value = sample_meta["narration_row_index"]
					if torch.is_tensor(row_index_value):
						all_narration_row_index.append(row_index_value.detach().cpu())
					else:
						all_narration_row_index.append(torch.as_tensor(row_index_value))
					all_video_ids.extend(list(sample_meta["video_id"]))
				else:
					if len(batch_data) == 3:
						images, target, sample_meta = batch_data
						all_narration_ids.extend(list(sample_meta["narration_id"]))
						row_index_value = sample_meta["narration_row_index"]
						if torch.is_tensor(row_index_value):
							all_narration_row_index.append(row_index_value.detach().cpu())
						else:
							all_narration_row_index.append(torch.as_tensor(row_index_value))
						all_video_ids.extend(list(sample_meta["video_id"]))
					else:
						images, target = batch_data
				target_gpu = target.cuda(args.gpu, non_blocking=True)
				if isinstance(images, list):
					logits_crops, feats_crops, cls_crops = [], [], []
					temporal_probe_crops = [] if save_temporal_probe else None
					verb_input_crops = [] if args.model_type in VJEPA2_MODEL_SPECS else None
					for crop in images:
						crop = crop.cuda(args.gpu, non_blocking=True)
						if args.use_half:
							crop = crop.half()
						logits, feat, cls_feat, temporal_probe_feat, verb_input_feat = forward_with_features(
							model, args.model_type, activation, crop, args
						)
						if args.multi_task:
							logits = logits[head]
							cls_feat = cls_feat[head]
							if temporal_probe_feat is None or not isinstance(temporal_probe_feat, dict):
								raise RuntimeError("Temporal probe features unavailable; expected head-wise temporal features.")
							if head not in temporal_probe_feat:
								raise RuntimeError(f"Temporal probe features missing head '{head}'.")
							temporal_probe_crops.append(temporal_probe_feat[head].unsqueeze(1).detach().cpu())
						elif temporal_probe_crops is not None and temporal_probe_feat is not None:
							temporal_probe_crops.append(temporal_probe_feat.unsqueeze(1).detach().cpu())
						if verb_input_feat is not None:
							verb_input_crops.append(verb_input_feat.unsqueeze(1).detach().cpu())
						logits_crops.append(logits.unsqueeze(1).detach().cpu())
						feats_crops.append(feat.unsqueeze(1).detach().cpu())
						cls_crops.append(cls_feat.unsqueeze(1).detach().cpu())
					all_outputs.append(torch.cat(logits_crops, dim=1))
					all_feats.append(torch.cat(feats_crops, dim=1))
					all_cls_feats.append(torch.cat(cls_crops, dim=1))
					if temporal_probe_crops is not None and len(temporal_probe_crops) > 0:
						all_temporal_probe_feats.append(torch.cat(temporal_probe_crops, dim=1))
					if verb_input_crops is not None and len(verb_input_crops) > 0:
						all_verb_input_feats.append(torch.cat(verb_input_crops, dim=1))
				else:
					images = images.cuda(args.gpu, non_blocking=True)
					if args.use_half:
						images = images.half()
					logits, feat, cls_feat, temporal_probe_feat, verb_input_feat = forward_with_features(
						model, args.model_type, activation, images, args
					)
					if args.multi_task:
						logits = logits[head]
						cls_feat = cls_feat[head]
						if temporal_probe_feat is None or not isinstance(temporal_probe_feat, dict):
							raise RuntimeError("Temporal probe features unavailable; expected head-wise temporal features.")
						if head not in temporal_probe_feat:
							raise RuntimeError(f"Temporal probe features missing head '{head}'.")
						all_temporal_probe_feats.append(temporal_probe_feat[head].detach().cpu())
					elif all_temporal_probe_feats is not None and temporal_probe_feat is not None:
						all_temporal_probe_feats.append(temporal_probe_feat.detach().cpu())
					if verb_input_feat is not None:
						all_verb_input_feats.append(verb_input_feat.detach().cpu())
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
	if len(all_narration_row_index) > 0:
		all_narration_row_index = torch.cat([value.view(-1) for value in all_narration_row_index], dim=0)
	if all_temporal_probe_feats is not None:
		if len(all_temporal_probe_feats) > 0:
			all_temporal_probe_feats = torch.cat(all_temporal_probe_feats)
		else:
			all_temporal_probe_feats = None
	if all_verb_input_feats is not None:
		if len(all_verb_input_feats) > 0:
			all_verb_input_feats = torch.cat(all_verb_input_feats)
		else:
			all_verb_input_feats = None

	if args.multi_task:
		save_name = f"{args.dataset}_{subset}_{head}_feat.pt"
	else:
		save_name = f"{args.dataset}_{subset}_feat.pt"
	save_path = os.path.join(args.output_dir, save_name)
	save_payload = {
		'feats': all_feats,
		'cls_feats': all_cls_feats,
		'outputs': all_outputs,
		'targets': all_targets,
	}
	if len(all_narration_ids) > 0:
		save_payload['narration_ids'] = all_narration_ids
		save_payload['narration_row_index'] = all_narration_row_index
		save_payload['video_ids'] = all_video_ids
	if all_temporal_probe_feats is not None:
		save_payload['temporal_probe_feats'] = all_temporal_probe_feats
	if all_verb_input_feats is not None:
		save_payload['verb_input_feats'] = all_verb_input_feats
	torch.save(save_payload, save_path)

	if args.multi_task:
		print(f"Saved {subset} {head} features to {save_path}")
	else:
		print(f"Saved {subset} features to {save_path}")
	print(f"  feats: {all_feats.shape}")
	print(f"  cls_feats: {all_cls_feats.shape}")
	if all_temporal_probe_feats is not None:
		print(f"  temporal_probe_feats: {all_temporal_probe_feats.shape}")
		if args.multi_task:
			print(f"  narration_ids: {len(all_narration_ids)}")
			print(f"  narration_row_index: {all_narration_row_index.shape}")
	if all_verb_input_feats is not None:
		print(f"  verb_input_feats: {all_verb_input_feats.shape}")
	print(f"  outputs: {all_outputs.shape}")
	print(f"  targets: {all_targets.shape}")


def get_args_parser():
	parser = argparse.ArgumentParser(description='V-JEPA / MViT feature extraction', add_help=False)

	# Data
	parser.add_argument('--dataset', default='egtea', type=str, choices=['ek100_cls', 'egtea'])
	parser.add_argument('--task-type', default='action', type=str, choices=['action', 'verb', 'noun'])
	parser.add_argument('--multi-task', action='store_true',
						help='extract verb/noun/action probe features jointly (EK100 + V-JEPA2 only)')
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
	parser.add_argument('--use-timestamps', action='store_true',
						help='use timestamps instead of frame numbers for EK100')
	parser.add_argument('--batch-size', default=8, type=int, help='number of samples per GPU for extraction')

	# Model
	parser.add_argument('--model-type', default='vjepa2_large', type=str,
						choices=['mvit_spatial', 'mvit_temporal'] + list(VJEPA2_MODEL_SPECS.keys()),
						help='backbone to use for extraction')
	parser.add_argument('--vjepa2-head', default='attentive', type=str,
						choices=['attentive', 'temporal_attentive', 'meanpool'],
						help='single-task V-JEPA2 head to instantiate during extraction')
	parser.add_argument('--pretrain-model', default='', type=str, help='path to checkpoint to load')
	parser.add_argument('--resume', default='', type=str, help='alternative checkpoint path')
	parser.add_argument('--skip-checkpoint', action='store_true',
						help='skip loading external checkpoint and use backbone default weights')
	parser.add_argument('--dropout-ratio', default=0.5, type=float, help='dropout ratio in classifier head')
	parser.add_argument('--num-classes', default=None, type=int, help='override class count (otherwise inferred)')
	parser.add_argument('--use-half', action='store_true', help='use half precision at inference')
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
	parser.add_argument('--vjepa2-motion-layer-set', default='auto', type=str,
						choices=['auto', 'legacy', 'hierarchical', 'none'],
						help='layer selection for optional verb_input_feats extraction; auto keeps legacy behavior except giant/gigantic V-JEPA2.1 backbones switch to hierarchical layers')

	# System
	parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
	parser.add_argument('--workers', default=4, type=int, help='data loading workers')
	parser.add_argument('--val-only', action='store_true', help='extract only validation/test features and skip train extraction')
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

	label_maps = None
	if args.multi_task:
		if args.dataset != 'ek100_cls':
			raise ValueError("Multi-task extraction is only supported for EK100.")
		if args.model_type not in VJEPA2_MODEL_SPECS:
			raise ValueError("Multi-task extraction is only supported for V-JEPA2 models.")
		train_csv = args.ek100_train_csv
		if not os.path.exists(train_csv):
			train_csv = args.metadata_train
			print(f"=> Using metadata_train for label maps: {train_csv}")
		label_maps = build_ek100_multitask_label_maps(train_csv)
		args.num_classes_action = len(label_maps["action"])
		args.num_classes_verb = len(label_maps["verb"])
		args.num_classes_noun = len(label_maps["noun"])
		args.num_classes = args.num_classes_action
		print(
			"Using multi-task classes (action/verb/noun): "
			f"{args.num_classes_action}/{args.num_classes_verb}/{args.num_classes_noun}"
		)
	else:
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
	args.egtea_finetune_type = 'action' if args.multi_task else args.task_type

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
			if args.multi_task:
				if args.vjepa2_head == 'temporal_attentive':
					model = VJEPA2TemporalMultiTaskProbeClassifier(
						args.model_type,
						num_verb_classes=args.num_classes_verb,
						num_noun_classes=args.num_classes_noun,
						num_action_classes=args.num_classes_action,
						probe_num_heads=args.probe_num_heads,
						probe_num_blocks=args.probe_num_blocks,
						probe_mlp_ratio=args.probe_mlp_ratio,
						probe_dropout=args.probe_dropout,
						use_activation_checkpointing=args.probe_use_activation_checkpointing,
						pretrained_backbone=pretrained_backbone,
					)
				else:
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
						pretrained_backbone=pretrained_backbone,
					)
			elif args.vjepa2_head == 'meanpool':
				model = VJEPA2MeanPoolClassifier(
					args.model_type, args.num_classes, dropout=args.dropout_ratio, pretrained_backbone=pretrained_backbone
				)
			elif args.vjepa2_head == 'temporal_attentive':
				model = VJEPA2TemporalProbeClassifier(
					args.model_type,
					args.num_classes,
					probe_num_heads=args.probe_num_heads,
					probe_num_blocks=args.probe_num_blocks,
					probe_mlp_ratio=args.probe_mlp_ratio,
					probe_dropout=args.probe_dropout,
					use_activation_checkpointing=args.probe_use_activation_checkpointing,
					pretrained_backbone=pretrained_backbone,
				)
			else:
				model = VJEPA2ProbeClassifier(
					args.model_type,
					args.num_classes,
					probe_num_heads=args.probe_num_heads,
					probe_num_blocks=args.probe_num_blocks,
					probe_mlp_ratio=args.probe_mlp_ratio,
					probe_dropout=args.probe_dropout,
					use_activation_checkpointing=args.probe_use_activation_checkpointing,
					pretrained_backbone=pretrained_backbone,
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
	mapping_vn2act = None
	if not args.multi_task:
		_, mapping_vn2act = generate_label_map(args.dataset, args)

	num_clips_at_val = args.num_clips
	args.num_clips = 1
	args.num_crops = 3
	if args.multi_task:
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
		train_dataset = datasets_flow.get_downstream_dataset_extract(
			train_transform, tokenizer, args, subset='train', label_mapping=mapping_vn2act,
		)
	else:
		if args.dataset == 'ek100_cls':
			train_dataset = EK100ExtractDataset(
				args,
				args.root,
				args.metadata_train,
				transform=train_transform,
				is_training=True,
				label_mapping=mapping_vn2act,
				clip_length=args.clip_length,
				clip_stride=args.clip_stride,
				sparse_sample=args.sparse_sample,
			)
		else:
			train_dataset = datasets.get_downstream_dataset_extract(
				train_transform, tokenizer, args, subset='train', label_mapping=mapping_vn2act,
			)
	args.num_clips = num_clips_at_val
	args.num_crops = 1
	if args.multi_task:
		val_dataset = EK100MultiTaskDataset(
			args,
			args.root,
			args.metadata_val,
			transform=val_transform,
			is_training=False,
			label_maps=label_maps,
			filter_actions=False,
			clip_length=args.clip_length,
			clip_stride=args.clip_stride,
			sparse_sample=args.sparse_sample,
		)
	elif args.model_type == 'mvit_temporal':
		val_dataset = datasets_flow.get_downstream_dataset_extract(
			val_transform, tokenizer, args, subset='val', label_mapping=mapping_vn2act,
		)
	else:
		if args.dataset == 'ek100_cls':
			val_dataset = EK100ExtractDataset(
				args,
				args.root,
				args.metadata_val,
				transform=val_transform,
				is_training=False,
				label_mapping=mapping_vn2act,
				clip_length=args.clip_length,
				clip_stride=args.clip_stride,
				sparse_sample=args.sparse_sample,
			)
		else:
			val_dataset = datasets.get_downstream_dataset_extract(
				val_transform, tokenizer, args, subset='val', label_mapping=mapping_vn2act,
			)

	if args.distributed and dist_utils.get_world_size() > 1:
		raise RuntimeError('Feature extraction must run with a single process to preserve sample order.')

	train_sampler = torch.utils.data.SequentialSampler(train_dataset)
	val_sampler = torch.utils.data.SequentialSampler(val_dataset)

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
	if args.multi_task:
		if not args.val_only:
			extract_split(train_loader, model, flow_model, args, subset='train', head='verb')
			extract_split(train_loader, model, flow_model, args, subset='train', head='noun')
		extract_split(val_loader, model, flow_model, args, subset='test', head='verb')
		extract_split(val_loader, model, flow_model, args, subset='test', head='noun')
	else:
		if not args.val_only:
			extract_split(train_loader, model, flow_model, args, subset='train')
		extract_split(val_loader, model, flow_model, args, subset='test')


if __name__ == '__main__':
	parser = argparse.ArgumentParser('Feature extraction for V-JEPA / MViT', parents=[get_args_parser()])
	args = parser.parse_args()
	os.makedirs(args.output_dir, exist_ok=True)
	main(args)
