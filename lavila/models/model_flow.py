# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


def _locate_patch_embed(model):
    if hasattr(model, "patch_embed"):
        return model, model.patch_embed
    if hasattr(model, "encoder") and hasattr(model.encoder, "patch_embed"):
        return model.encoder, model.encoder.patch_embed
    raise AttributeError("Could not find patch_embed on the provided model.")


def adapt_vjepa_for_flow(vjepa_model):
    target, patch_embed = _locate_patch_embed(vjepa_model)
    proj = getattr(patch_embed, "proj", None)
    if not isinstance(proj, nn.Conv3d):
        raise TypeError("Expected patch_embed.proj to be a Conv3d layer.")

    new_proj = nn.Conv3d(
        in_channels=2,
        out_channels=proj.out_channels,
        kernel_size=proj.kernel_size,
        stride=proj.stride,
        padding=proj.padding,
        dilation=proj.dilation,
        groups=proj.groups,
        bias=proj.bias is not None,
        padding_mode=proj.padding_mode,
    ).to(device=proj.weight.device, dtype=proj.weight.dtype)

    with torch.no_grad():
        weight_mean = proj.weight.mean(dim=1, keepdim=True)
        new_proj.weight.copy_(weight_mean.repeat(1, 2, 1, 1, 1))
        if proj.bias is not None:
            new_proj.bias.copy_(proj.bias)

    target.patch_embed.proj = new_proj
    return vjepa_model
