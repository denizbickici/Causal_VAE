# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
MViT Model Wrapper for LaViLa Infrastructure
This module provides a unified interface for MViT models (spatial and temporal)
that is compatible with LaViLa's training and evaluation pipelines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import mvit_v2_s, mvit_v1_b, MViT_V2_S_Weights, MViT_V1_B_Weights


class MViTWrapper(nn.Module):
    """
    Unified wrapper for MViT models that works with LaViLa's infrastructure.
    Supports both spatial (RGB) and temporal (optical flow) inputs.
    """
    
    def __init__(
        self,
        model_type='mvit_spatial',
        num_classes=106,
        dropout=0.5,
        mvit_variant='v2_s',  # 'v2_s', 'v1_b'
        pretrained=True,
        freeze_backbone=False,
    ):
        super().__init__()
        
        self.model_type = model_type
        self.num_classes = num_classes
        self.mvit_variant = mvit_variant
        
        # Initialize MViT model based on variant
        if mvit_variant == 'v2_s':
            weights = MViT_V2_S_Weights.DEFAULT if pretrained else None
            self.mvit = mvit_v2_s(weights=weights)
            feature_dim = 768  # MViT-v2-S feature dimension
        elif mvit_variant == 'v1_b':
            weights = MViT_V1_B_Weights.DEFAULT if pretrained else None
            self.mvit = mvit_v1_b(weights=weights)
            feature_dim = 768  # MViT-v1-B feature dimension
        else:
            raise ValueError(f"Unknown MViT variant: {mvit_variant}")
        
        # Store feature dimension for compatibility with LaViLa
        self.num_features = feature_dim
        
        # Modify first layer for temporal (optical flow) input if needed
        if model_type == 'mvit_temporal':
            self._adapt_for_optical_flow()
        
        # Replace the classification head
        self.dropout = nn.Dropout(p=dropout)
        self.fc_cls = nn.Linear(feature_dim, num_classes)
        
        # Initialize the classification head
        self.fc_cls.weight.data.normal_(mean=0.0, std=0.01)
        self.fc_cls.bias.data.zero_()
        
        # Remove original head
        self.mvit.head = nn.Identity()
        
        # Optionally freeze the backbone
        if freeze_backbone:
            self._freeze_backbone()
    
    def _adapt_for_optical_flow(self):
        """Adapt the first convolutional layer for 2-channel optical flow input."""
        original_conv = self.mvit.conv_proj
        
        # Create new conv layer with 2 input channels
        new_conv = nn.Conv3d(
            in_channels=2,  # Optical flow has 2 channels (u, v)
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )
        
        # Initialize the new conv layer
        # Option 1: Initialize randomly
        nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        
        # Option 2: Initialize by averaging RGB channels (alternative approach)
        # with torch.no_grad():
        #     # Average the weights across the 3 RGB channels to get 2 channels
        #     original_weights = original_conv.weight.data
        #     new_conv.weight[:, 0, :, :, :] = original_weights[:, :2, :, :, :].mean(dim=1)
        #     new_conv.weight[:, 1, :, :, :] = original_weights[:, 1:, :, :, :].mean(dim=1)
        
        if new_conv.bias is not None:
            if original_conv.bias is not None:
                new_conv.bias.data = original_conv.bias.data.clone()
            else:
                nn.init.constant_(new_conv.bias, 0)
        
        # Replace the conv layer
        self.mvit.conv_proj = new_conv
    
    def _freeze_backbone(self):
        """Freeze the MViT backbone, keeping only the classification head trainable."""
        for param in self.mvit.parameters():
            param.requires_grad = False
    
    def forward(self, x, use_checkpoint=False, return_features=False):
        """
        Forward pass through the MViT model.
        
        Args:
            x: Input tensor of shape [B, C, T, H, W]
               - For spatial: C=3 (RGB)
               - For temporal: C=2 (optical flow)
            use_checkpoint: Whether to use gradient checkpointing (not implemented in torchvision MViT)
            return_features: Whether to return features before classification
        
        Returns:
            If return_features is False: logits of shape [B, num_classes]
            If return_features is True: (logits, features) tuple
        """
        # Extract features using MViT backbone
        features = self.mvit(x)
        
        # Apply dropout and classification head
        features_dropout = self.dropout(features)
        logits = self.fc_cls(features_dropout)
        
        if return_features:
            return logits, features
        else:
            return logits
    
    def extract_features(self, x):
        """Extract features without the classification head."""
        return self.mvit(x)


class MViTMultiHead(nn.Module):
    """
    MViT with multiple classification heads (e.g., for verb, noun, action).
    Compatible with LaViLa's multi-head classification setup.
    """
    
    def __init__(
        self,
        model_type='mvit_spatial',
        num_classes_list=[19, 53, 106],  # [verb, noun, action]
        dropout=0.5,
        mvit_variant='v2_s',
        pretrained=True,
        freeze_backbone=False,
    ):
        super().__init__()
        
        self.model_type = model_type
        self.num_classes_list = num_classes_list
        self.mvit_variant = mvit_variant
        
        # Initialize MViT backbone
        if mvit_variant == 'v2_s':
            weights = MViT_V2_S_Weights.DEFAULT if pretrained else None
            self.mvit = mvit_v2_s(weights=weights)
            feature_dim = 768
        elif mvit_variant == 'v1_b':
            weights = MViT_V1_B_Weights.DEFAULT if pretrained else None
            self.mvit = mvit_v1_b(weights=weights)
            feature_dim = 768
        else:
            raise ValueError(f"Unknown MViT variant: {mvit_variant}")
        
        self.num_features = feature_dim
        
        # Modify for optical flow if needed
        if model_type == 'mvit_temporal':
            self._adapt_for_optical_flow()
        
        # Remove original head
        self.mvit.head = nn.Identity()
        
        # Create multiple classification heads
        self.dropout = nn.Dropout(p=dropout)
        self.fc_cls = nn.ModuleList([
            nn.Linear(feature_dim, num_classes) for num_classes in num_classes_list
        ])
        
        # Initialize classification heads
        for fc in self.fc_cls:
            fc.weight.data.normal_(mean=0.0, std=0.01)
            fc.bias.data.zero_()
        
        # Optionally freeze backbone
        if freeze_backbone:
            self._freeze_backbone()
    
    def _adapt_for_optical_flow(self):
        """Adapt for optical flow input (same as MViTWrapper)."""
        original_conv = self.mvit.conv_proj
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
        self.mvit.conv_proj = new_conv
    
    def _freeze_backbone(self):
        """Freeze the MViT backbone."""
        for param in self.mvit.parameters():
            param.requires_grad = False
    
    def forward(self, x, use_checkpoint=False, return_features=False):
        """
        Forward pass through multi-head MViT.
        
        Returns:
            If return_features is False: list of logits for each head
            If return_features is True: (logit_list, features) tuple
        """
        # Extract features
        features = self.mvit(x)
        
        # Apply dropout and multiple classification heads
        features_dropout = self.dropout(features)
        logit_list = [fc(features_dropout) for fc in self.fc_cls]
        
        if return_features:
            return logit_list, features
        else:
            return logit_list


def create_mvit_model(
    model_type='mvit_spatial',
    num_classes=106,
    dropout=0.5,
    mvit_variant='v2_s',
    pretrained=True,
    multi_head=False,
    num_classes_list=None,
    freeze_backbone=False,
):
    """
    Factory function to create MViT models.
    
    Args:
        model_type: 'mvit_spatial' or 'mvit_temporal'
        num_classes: Number of output classes (for single head)
        dropout: Dropout rate before classification head
        mvit_variant: 'v2_s' or 'v1_b'
        pretrained: Whether to use pretrained weights
        multi_head: Whether to use multiple classification heads
        num_classes_list: List of num_classes for each head (if multi_head)
        freeze_backbone: Whether to freeze the backbone weights
    
    Returns:
        MViT model instance
    """
    if multi_head:
        if num_classes_list is None:
            raise ValueError("num_classes_list must be provided for multi_head models")
        return MViTMultiHead(
            model_type=model_type,
            num_classes_list=num_classes_list,
            dropout=dropout,
            mvit_variant=mvit_variant,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    else:
        return MViTWrapper(
            model_type=model_type,
            num_classes=num_classes,
            dropout=dropout,
            mvit_variant=mvit_variant,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )


# Compatibility aliases for LaViLa integration
MViT_Spatial = lambda num_classes, dropout=0.5: create_mvit_model(
    'mvit_spatial', num_classes, dropout
)
MViT_Temporal = lambda num_classes, dropout=0.5: create_mvit_model(
    'mvit_temporal', num_classes, dropout
)