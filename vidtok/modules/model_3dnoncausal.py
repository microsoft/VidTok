import functools
import math
import time
import warnings
from typing import Any, Callable, Optional

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from beartype.typing import List, Optional, Tuple, Union
from einops import rearrange
from packaging import version

from .model_3dcausal import spatial_temporal_resblk, nonlinearity, Normalize, AttnBlock
from .util import checkpoint


def make_attn(in_channels, use_checkpoint=False, norm_type="groupnorm"):
    return AttnBlockWrapper(in_channels, use_checkpoint=use_checkpoint, norm_type=norm_type)


class AttnBlockWrapper(AttnBlock):
    def __init__(self, in_channels, use_checkpoint=False, norm_type="groupnorm"):
        super().__init__(in_channels, use_checkpoint=use_checkpoint, norm_type=norm_type)
        self.q = torch.nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def attention(self, h_: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, t, h, w = q.shape
        q, k, v = map(lambda x: rearrange(x, "b c t h w -> b t (h w) c").contiguous(), (q, k, v))
        h_ = torch.nn.functional.scaled_dot_product_attention(q, k, v)  # scale is dim ** -0.5 per default
        return rearrange(h_, "b t (h w) c -> b c t h w", h=h, w=w, c=c, b=b)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.in_channels = in_channels
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x.to(torch.float32), scale_factor=2.0, mode="nearest").to(x.dtype)
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.in_channels = in_channels
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class TimeDownsampleRes2x(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        mix_factor: float = 2.0,
    ):
        super().__init__()
        self.kernel_size = (3, 3, 3)
        self.avg_pool = nn.AvgPool3d((3, 1, 1), stride=(2, 1, 1))
        self.conv = nn.Conv3d(in_channels, out_channels, 3, stride=(2, 1, 1), padding=(0, 1, 1))
        # https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/opensora/models/causalvideovae/model/modules/updownsample.py
        self.mix_factor = torch.nn.Parameter(torch.Tensor([mix_factor]))

    def forward(self, x):
        alpha = torch.sigmoid(self.mix_factor)
        pad = (0, 0, 0, 0, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x1 = self.avg_pool(x)
        x2 = self.conv(x)
        return alpha * x1 + (1 - alpha) * x2


class TimeUpsampleRes2x(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        mix_factor: float = 2.0,
    ):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        # https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/opensora/models/causalvideovae/model/modules/updownsample.py
        self.mix_factor = torch.nn.Parameter(torch.Tensor([mix_factor]))

    def forward(self, x):
        alpha = torch.sigmoid(self.mix_factor)
        xlst = [
            torch.nn.functional.interpolate(
                sx.unsqueeze(0).to(torch.float32), scale_factor=[2.0, 1.0, 1.0], mode="nearest"
            ).to(x.dtype)
            for sx in x
        ]
        x = torch.cat(xlst, dim=0)
        x_ = self.conv(x)
        return alpha * x + (1 - alpha) * x_


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
        use_checkpoint=False,
        norm_type="groupnorm",
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm_type = norm_type

        self.norm1 = Normalize(in_channels, norm_type=self.norm_type)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels, norm_type=self.norm_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.use_checkpoint = use_checkpoint

    def forward(self, x, temb):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            if x.grad is not None or x.grad_fn is not None:
                use_checkpoint = True
            else:
                use_checkpoint = False

        if use_checkpoint:
            assert temb is None, "checkpointing not supported with temb"
            return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)
        else:
            return self._forward(x, temb)

    def _forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class ResnetBlock1D(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
        zero_init=False,
        use_checkpoint=False,
        norm_type="groupnorm",
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm_type = norm_type

        self.norm1 = Normalize(in_channels, norm_type=self.norm_type)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels, norm_type=self.norm_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        if zero_init:
            self.conv2.weight.data.zero_()
            self.conv2.bias.data.zero_()

        self.use_checkpoint = use_checkpoint

    def forward(self, x, temb):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            if x.grad is not None or x.grad_fn is not None:
                use_checkpoint = True
            else:
                use_checkpoint = False

        if use_checkpoint:
            assert temb is None, "checkpointing not supported with temb"
            return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)
        else:
            return self._forward(x, temb)

    def _forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class ResnetNoncausalBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
        use_checkpoint=False,
        norm_type="groupnorm",
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm_type = norm_type

        self.norm1 = Normalize(in_channels, norm_type=self.norm_type)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels, norm_type=self.norm_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=1)
        self.use_checkpoint = use_checkpoint

    def forward(self, x, temb):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            if x.grad is not None or x.grad_fn is not None:
                use_checkpoint = True
            else:
                use_checkpoint = False

        if use_checkpoint:
            assert temb is None, "checkpointing not supported with temb"
            return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)
        else:
            return self._forward(x, temb)

    def _forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class Encoder3D(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch=8,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        z_channels,
        double_z=True,
        norm_type="groupnorm",
        **ignore_kwargs,
    ):
        super().__init__()
        use_checkpoint = ignore_kwargs.get("use_checkpoint", False)
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.fix_encoder = ignore_kwargs.get("fix_encoder", False)
        self.tempo_ds = [self.num_resolutions - 2, self.num_resolutions - 3]
        self.norm_type = norm_type

        # downsampling
        make_conv_cls = self._make_conv()
        make_attn_cls = self._make_attn()
        make_resblock_cls = self._make_resblock()

        self.conv_in = make_conv_cls(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        self.down_temporal = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_temporal = nn.ModuleList()
            attn_temporal = nn.ModuleList()

            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        use_checkpoint=use_checkpoint,
                        norm_type=self.norm_type,
                    )
                )
                block_temporal.append(
                    ResnetBlock1D(
                        in_channels=block_out,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        zero_init=True,
                        use_checkpoint=use_checkpoint,
                        norm_type=self.norm_type,
                    )
                )
                block_in = block_out

            down = nn.Module()
            down.block = block
            down.attn = attn

            down_temporal = nn.Module()
            down_temporal.block = block_temporal
            down_temporal.attn = attn_temporal

            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                if i_level in self.tempo_ds:
                    down_temporal.downsample = TimeDownsampleRes2x(block_in, block_in)

            self.down.append(down)
            self.down_temporal.append(down_temporal)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = make_resblock_cls(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
            norm_type=self.norm_type,
        )
        self.mid.attn_1 = make_attn(block_in, norm_type=self.norm_type)
        self.mid.block_2 = make_resblock_cls(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
            norm_type=self.norm_type,
        )

        # end
        self.norm_out = Normalize(block_in, norm_type=self.norm_type)
        self.conv_out = make_conv_cls(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        if self.fix_encoder:
            for param in self.parameters():
                param.requires_grad = False

    def _make_attn(self) -> Callable:
        return make_attn

    def _make_resblock(self) -> Callable:
        return ResnetNoncausalBlock

    def _make_conv(self) -> Callable:
        return nn.Conv3d

    def forward(self, x):
        temb = None
        B, _, T, _, _ = x.shape

        # downsampling
        if x.shape[1] == 4 and self.conv_in.in_channels == 3:
            raise ValueError("Mismatched number of input channels")
        hs = [self.conv_in(x)]

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = spatial_temporal_resblk(
                    hs[-1], self.down[i_level].block[i_block], self.down_temporal[i_level].block[i_block], temb
                )
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                # spatial downsample
                htmp = einops.rearrange(hs[-1], "b c t h w -> (b t) c h w")
                htmp = self.down[i_level].downsample(htmp)
                htmp = einops.rearrange(htmp, "(b t) c h w -> b c t h w", b=B, t=T)
                if i_level in self.tempo_ds:
                    # temporal downsample
                    htmp = self.down_temporal[i_level].downsample(htmp)
                hs.append(htmp)
                B, _, T, _, _ = htmp.shape

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder3D(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=8,
        z_channels,
        give_pre_end=False,
        tanh_out=False,
        norm_type="groupnorm",
        **ignorekwargs,
    ):
        super().__init__()
        use_checkpoint = ignorekwargs.get("use_checkpoint", False)

        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.fix_decoder = ignorekwargs.get("fix_decoder", False)
        self.tempo_us = [1, 2]
        self.norm_type = norm_type

        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]

        make_attn_cls = self._make_attn()
        make_resblock_cls = self._make_resblock()
        make_conv_cls = self._make_conv()
        self.conv_in = make_conv_cls(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = make_resblock_cls(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
            norm_type=self.norm_type,
        )
        self.mid.attn_1 = make_attn_cls(
            block_in, use_checkpoint=use_checkpoint, norm_type=self.norm_type
        )
        self.mid.block_2 = make_resblock_cls(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
            norm_type=self.norm_type,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        use_checkpoint=use_checkpoint,
                        norm_type=self.norm_type,
                    )
                )
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
            self.up.insert(0, up)

        self.up_temporal = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock1D(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        zero_init=True,
                        use_checkpoint=use_checkpoint,
                        norm_type=self.norm_type,
                    )
                )
                block_in = block_out
            up_temporal = nn.Module()
            up_temporal.block = block
            up_temporal.attn = attn
            if i_level in self.tempo_us:
                up_temporal.upsample = TimeUpsampleRes2x(block_in, block_in)

            self.up_temporal.insert(0, up_temporal) 

        # end
        self.norm_out = Normalize(block_in, norm_type=self.norm_type)
        self.conv_out = make_conv_cls(block_in, out_ch, kernel_size=3, stride=1, padding=1)

        if self.fix_decoder:
            for param in self.parameters():
                param.requires_grad = False

    def _make_attn(self) -> Callable:
        return make_attn

    def _make_resblock(self) -> Callable:
        return ResnetNoncausalBlock

    def _make_conv(self) -> Callable:
        return nn.Conv3d

    def get_last_layer(self, **kwargs):
        return self.conv_out.weight

    def forward(self, z, **kwargs):
        temb = None
        B, _, T, _, _ = z.shape

        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb, **kwargs)
        h = self.mid.attn_1(h, **kwargs)
        h = self.mid.block_2(h, temb, **kwargs)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = spatial_temporal_resblk(
                    h, self.up[i_level].block[i_block], self.up_temporal[i_level].block[i_block], temb
                )
            if i_level != 0:
                # spatial upsample
                h = einops.rearrange(h, "b c t h w -> (b t) c h w")
                h = self.up[i_level].upsample(h)
                h = einops.rearrange(h, "(b t) c h w -> b c t h w", b=B, t=T)
                if i_level in self.tempo_us:
                    # temporal upsample
                    h = self.up_temporal[i_level].upsample(h)
                B, _, T, _, _ = h.shape
        # end
        if self.give_pre_end:
            return h
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h, **kwargs)
        if self.tanh_out:
            h = torch.tanh(h)
        return h
