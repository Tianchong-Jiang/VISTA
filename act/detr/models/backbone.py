# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict
from functools import partial

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.ops import MLP
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
import einops

from ..util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding, PositionEmbeddingSine, PositionEmbeddingLearned

from torchvision.models.vision_transformer import EncoderBlock

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

class BackboneResNet(nn.Module):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 dilation: bool = False):
        super().__init__()
        resnet = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            weights="DEFAULT", norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
        # resnet = getattr(torchvision.models, name)(
        #     replace_stride_with_dilation=[False, False, dilation],
        #     pretrained=False)
        # new_conv1 = nn.Conv2d(3 + 6,
        #                     resnet.conv1.out_channels,
        #                     kernel_size=resnet.conv1.kernel_size,
        #                     stride=resnet.conv1.stride,
        #                     padding=resnet.conv1.padding,
        #                     bias=False)
        # resnet.conv1 = new_conv1

        self.backbone = IntermediateLayerGetter(resnet, return_layers={'layer4': "0"})

        self.num_channels = 512 if name in ('resnet18', 'resnet34') else 2048

    def forward(self, image):
        rgb = image[:, :3]
        out = self.backbone(rgb)
        return out
    
class LinProj(nn.Module):
    def __init__(self,
                    patch_size: int = 16,
                    hidden_dim: int = 256,
                    plucker_as_pe: bool = True):
        super().__init__()

        self.image_hw = (256, 256)
        self.patch_size = patch_size
        self.seq_length = (self.image_hw[0] // self.patch_size) * (self.image_hw[1] // self.patch_size)
        self.hidden_dim = hidden_dim
        self.num_channels = hidden_dim
        self.plucker_as_pe = plucker_as_pe

        assert self.image_hw[0] % self.patch_size == 0 and self.image_hw[1] % self.patch_size == 0, \
            f"Image dimensions {self.image_hw[0]}x{self.image_hw[1]} must be divisible by patch size {self.patch_size}"

        self.conv_proj = nn.Conv2d(
            in_channels=3, #if plucker_as_pe else 9,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        # self.emb_proj = nn.Linear(6, self.hidden_dim)
        self.emb_proj = MLP(in_channels=6,
                            hidden_channels=[hidden_dim, hidden_dim, hidden_dim])
                            

        # self.pe = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.pe = nn.Parameter(torch.empty(1, hidden_dim, self.image_hw[0] // self.patch_size, self.image_hw[1] // self.patch_size))
        nn.init.normal_(self.pe, std=0.5)


    def forward(self, image):
        if self.plucker_as_pe:
            # get plucker embedding as positional encoding
            # image: (b, c, h, w)
            plucker = image[:, 3:]
            plucker = plucker[:, :, self.patch_size//2::self.patch_size, self.patch_size//2::self.patch_size]
            h_patch, h_patch = plucker.shape[2], plucker.shape[3]
            plucker = einops.rearrange(plucker, 'b c h w -> b (h w) c')
            plucker = self.emb_proj(plucker)
            plucker = einops.rearrange(plucker, 'b (h w) c -> b c h w', h=h_patch, w=h_patch)
            rgb = image[:, :3]
            rgb = self.conv_proj(rgb)
            return [rgb], [plucker]
        else:
            rgb = image[:, :3]
            rgb = self.conv_proj(rgb)
            pe = self.pe.repeat(rgb.shape[0], 1, 1, 1)
            return [rgb], [pe]


class SinusoidalEncoder(nn.Module):
    def __init__(self, n_freq_bands=64):
        super().__init__()
        self.n_freq_bands = n_freq_bands

    def forward(self, x):
        """Input shape: [..., 6] → Output shape: [..., 6*2*n_freq_bands]"""
        freqs = torch.linspace(1.0, 1e4, self.n_freq_bands, device=x.device)
        x = x.unsqueeze(-1)  # [..., 6, 1]
        sin_enc = torch.sin(x * freqs)  # [..., 6, n_freq_bands]
        cos_enc = torch.cos(x * freqs)
        encoded = torch.cat([sin_enc, cos_enc], dim=-1)  # [..., 6, 2*n_freq_bands]
        return encoded.flatten(-2)  # [..., 6*2*n_freq_bands]

class BackboneViT(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        patch_size: int = 32,
        num_layers: int = 6,
        num_heads: int = 8,
        hidden_dim: int = 768,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        norm_layer = partial(nn.LayerNorm, eps=1e-6),
        plucker_as_pe: bool = True,
        embed_method: str = 'lin', # 'lin', 'sin', 'sinlin', 'mlp'
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default

        self.image_hw = (256, 256)
        self.patch_size = patch_size
        self.seq_length = (self.image_hw[0] // self.patch_size) * (self.image_hw[1] // self.patch_size)
        self.hidden_dim = hidden_dim
        self.num_channels = hidden_dim
        self.plucker_as_pe = plucker_as_pe

        assert self.image_hw[0] % self.patch_size == 0 and self.image_hw[1] % self.patch_size == 0, \
            f"Image dimensions {self.image_hw[0]}x{self.image_hw[1]} must be divisible by patch size {self.patch_size}"

        self.conv_proj = nn.Conv2d(
            in_channels=3 if plucker_as_pe else 9,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        if isinstance(self.conv_proj, nn.Conv2d):
            import math
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)

        if plucker_as_pe:
            if embed_method == 'lin':
                self.emb_proj = nn.Linear(6, hidden_dim)
            elif embed_method == 'sin':
                self.emb_proj = SinusoidalEncoder()
            elif embed_method == 'sinlin':
                sin_enc = SinusoidalEncoder()
                self.emb_proj = nn.Sequential(sin_enc, nn.Linear(6*2*sin_enc.n_freq_bands, hidden_dim))
            elif embed_method == 'mlp':
                self.emb_proj = nn.Sequential(
                    nn.Linear(6, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            else:
                raise NotImplementedError(f"Embedding method {embed_method} not implemented")
        else:
            self.pos_embedding = nn.Parameter(torch.empty(1, self.seq_length, hidden_dim).normal_(std=0.02))  # from BERT

        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def _sinusoidal_encode(self, plucker):
        freqs = torch.linspace(1.0, 1e4, 64, device=plucker.device)

        # Expand input for broadcasting: [..., 6, 1] * [n_freq_bands] -> [..., 6, n_freq_bands]
        plucker = plucker.unsqueeze(-1)  # Add frequency dimension
        sin_enc = torch.sin(plucker * freqs)
        cos_enc = torch.cos(plucker * freqs)

        # Concatenate and flatten last two dimensions
        encoded = torch.cat([sin_enc, cos_enc], dim=-1)  # [..., 6, 2*n_freq_bands]
        return encoded.flatten(-2)  # [..., 6*2*n_freq_bands]

    def _get_pos_embedding(self, x):
        # get plucker embedding as positional encoding
        # x: (b, c, h, w)
        plucker = x[:, 3:]
        plucker = plucker[:, :, self.patch_size//2::self.patch_size, self.patch_size//2::self.patch_size]
        plucker = plucker.flatten(2).transpose(1, 2)
        plucker = self.emb_proj(plucker)

        rgb = x[:, :3]
        rgb = self.conv_proj(rgb)
        rgb = rgb.flatten(2).transpose(1, 2)

        x = rgb + plucker

        return x

    def forward(self, x):
        if self.plucker_as_pe:
            x = self._get_pos_embedding(x)

        if not self.plucker_as_pe:
            x = self.conv_proj(x)
            x = x.flatten(2).transpose(1, 2)
            x = x + self.pos_embedding

        x = self.ln(self.layers(self.dropout(x)))

        # transform to (b, hidden_dim, n_patches, n_patches)
        x = einops.rearrange(x, 'b (h w) f -> b f h w', h=8, w=8)

        return {'0': x}


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list):
        xs = self[0](tensor_list)
        out = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)

    if 'resnet' in args.backbone:
        backbone = BackboneResNet(args.backbone, args.dilation)
        model = Joiner(backbone, position_embedding)
    elif 'vit' in args.backbone:
        backbone = BackboneViT(
            patch_size=args.patch_size,
            plucker_as_pe=args.plucker_as_pe,
            embed_method=args.embed_method,
        )
        model = Joiner(backbone, position_embedding)
    elif 'lin' in args.backbone:
        model = LinProj(
            patch_size=args.patch_size,
            plucker_as_pe=args.plucker_as_pe,
            hidden_dim=args.hidden_dim
        )
    else:
        raise NotImplementedError(f"Backbone {args.backbone} not implemented")

    return model
