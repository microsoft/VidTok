from typing import Callable
from beartype import beartype
from beartype.typing import Tuple, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .util import checkpoint


def spatial_temporal_resblk(x, block_s, block_t, temb):
    assert len(x.shape) == 5, "input should be 5D tensor, but got {}D tensor".format(len(x.shape))
    B, C, T, H, W = x.shape
    x = einops.rearrange(x, "b c t h w -> (b t) c h w")
    x = block_s(x, temb)
    x = einops.rearrange(x, "(b t) c h w -> b c t h w", b=B, t=T)
    x = einops.rearrange(x, "b c t h w -> (b h w) c t")
    x = block_t(x, temb)
    x = einops.rearrange(x, "(b h w) c t -> b c t h w", b=B, h=H, w=W)
    return x


def nonlinearity(x):
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32, norm_type="groupnorm"):
    if norm_type == "groupnorm":
        return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == "layernorm":
        return LayerNorm(num_channels=in_channels, eps=1e-6)


def pad_at_dim(t, pad, dim=-1, pad_mode="constant", value=0.0):
    assert pad_mode in ["constant", "replicate", "reflect"]
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    if pad_mode == "constant":
        return F.pad(t, (*zeros, *pad), value=value)
    return F.pad(t, (*zeros, *pad), mode=pad_mode)


def divisible_by(num, den):
    return (num % den) == 0


def is_odd(n):
    return not divisible_by(n, 2)


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


def make_attn(in_channels, use_checkpoint=False, norm_type="groupnorm"):
    return AttnBlockWrapper(in_channels, use_checkpoint=use_checkpoint, norm_type=norm_type)


class LayerNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = torch.nn.LayerNorm(num_channels, eps=eps, elementwise_affine=True)

    def forward(self, x):
        if x.dim() == 5:
            x = rearrange(x, "b c t h w -> b t h w c")
            x = self.norm(x)
            x = rearrange(x, "b t h w c -> b c t h w")
        elif x.dim() == 4:
            x = rearrange(x, "b c h w -> b h w c")
            x = self.norm(x)
            x = rearrange(x, "b h w c -> b c h w")
        else:
            x = rearrange(x, "b c s -> b s c")
            x = self.norm(x)
            x = rearrange(x, "b s c -> b c s")
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_channels, use_checkpoint=False, norm_type="groupnorm"):
        super().__init__()
        self.in_channels = in_channels
        self.norm_type = norm_type

        self.norm = Normalize(in_channels, norm_type=self.norm_type)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        self.use_checkpoint = use_checkpoint

    def attention(self, h_: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q, k, v = map(lambda x: rearrange(x, "b c h w -> b 1 (h w) c").contiguous(), (q, k, v))
        h_ = torch.nn.functional.scaled_dot_product_attention(q, k, v)  # scale is dim ** -0.5 per default
        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x, **kwargs):
        if self.use_checkpoint:
            return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)
        else:
            return self._forward(x)

    def _forward(self, x, **kwargs):
        h_ = x
        h_ = self.attention(h_)
        h_ = self.proj_out(h_)
        return x + h_


class AttnBlockWrapper(AttnBlock):
    def __init__(self, in_channels, use_checkpoint=False, norm_type="groupnorm"):
        super().__init__(in_channels, use_checkpoint=use_checkpoint, norm_type=norm_type)
        self.q = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.k = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.v = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.proj_out = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)

    def attention(self, h_: torch.Tensor) -> torch.Tensor:
        B = h_.shape[0]
        h_ = rearrange(h_, "b c t h w -> (b t) c h w")
        h_ = self.norm(h_)
        h_ = rearrange(h_, "(b t) c h w -> b c t h w", b=B)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, t, h, w = q.shape
        q, k, v = map(lambda x: rearrange(x, "b c t h w -> b t (h w) c").contiguous(), (q, k, v))
        h_ = torch.nn.functional.scaled_dot_product_attention(q, k, v)  # scale is dim ** -0.5 per default
        return rearrange(h_, "b t (h w) c -> b c t h w", h=h, w=w, c=c, b=b)


class CausalConv1d(nn.Module):
    @beartype
    def __init__(self, chan_in, chan_out, kernel_size: int, pad_mode="constant", **kwargs):
        super().__init__()
        dilation = kwargs.pop("dilation", 1)
        stride = kwargs.pop("stride", 1)
        self.pad_mode = pad_mode
        self.time_pad = dilation * (kernel_size - 1) + (1 - stride)
        self.time_causal_padding = (self.time_pad, 0)

        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        pad_mode = self.pad_mode if self.time_pad < x.shape[2] else "constant"
        x = F.pad(x, self.time_causal_padding, mode=pad_mode)
        return self.conv(x)


class CausalConv3d(nn.Module):
    @beartype
    def __init__(self, chan_in, chan_out, kernel_size: Union[int, Tuple[int, int, int]], pad_mode="constant", **kwargs):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)
        dilation = kwargs.pop("dilation", 1)
        stride = kwargs.pop("stride", 1)
        dilation = cast_tuple(dilation, 3)
        stride = cast_tuple(stride, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        self.pad_mode = pad_mode
        time_pad = dilation[0] * (time_kernel_size - 1) + (1 - stride[0])
        height_pad = dilation[1] * (height_kernel_size - 1) + (1 - stride[1])
        width_pad = dilation[2] * (height_kernel_size - 1) + (1 - stride[2])

        self.time_pad = time_pad
        self.time_causal_padding = (
            width_pad // 2,
            width_pad - width_pad // 2,
            height_pad // 2,
            height_pad - height_pad // 2,
            time_pad,
            0,
        )

        self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        pad_mode = self.pad_mode if self.time_pad < x.shape[2] else "constant"

        x = F.pad(x, self.time_causal_padding, mode=pad_mode)
        return self.conv(x)


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


class TimeDownsampleResCausal2x(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        mix_factor: float = 2.0,
    ):
        super().__init__()
        self.kernel_size = (3, 3, 3)
        self.avg_pool = nn.AvgPool3d((3, 1, 1), stride=(2, 1, 1))
        self.conv = CausalConv3d(in_channels, out_channels, 3, stride=(2, 1, 1))
        # https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/opensora/models/causalvideovae/model/modules/updownsample.py
        self.mix_factor = torch.nn.Parameter(torch.Tensor([mix_factor]))

    def forward(self, x):
        alpha = torch.sigmoid(self.mix_factor)
        pad = (0, 0, 0, 0, 1, 0)
        x1 = self.avg_pool(torch.nn.functional.pad(x, pad, mode="constant", value=0))
        x2 = self.conv(x)
        return alpha * x1 + (1 - alpha) * x2


class TimeUpsampleResCausal2x(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        mix_factor: float = 2.0,
    ):
        super().__init__()
        self.conv = CausalConv3d(in_channels, out_channels, 3)
        # https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/opensora/models/causalvideovae/model/modules/updownsample.py
        self.mix_factor = torch.nn.Parameter(torch.Tensor([mix_factor]))

    def forward(self, x):
        alpha = torch.sigmoid(self.mix_factor)
        x = torch.nn.functional.interpolate(x.to(torch.float32), scale_factor=[2.0, 1.0, 1.0], mode="nearest").to(
            x.dtype
        )
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
        if self.use_checkpoint:
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


class ResnetCausalBlock(nn.Module):
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
        self.conv1 = CausalConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels, norm_type=self.norm_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = CausalConv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = CausalConv3d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                )
            else:
                self.nin_shortcut = CausalConv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                )
        self.use_checkpoint = use_checkpoint

    def forward(self, x, temb):
        if self.use_checkpoint:
            assert temb is None, "checkpointing not supported with temb"
            return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)
        else:
            return self._forward(x, temb)

    def _forward(self, x, temb=None):
        B = x.shape[0]
        h = x
        h = rearrange(h, "b c t h w -> (b t) c h w")
        h = self.norm1(h)
        h = nonlinearity(h)
        h = rearrange(h, "(b t) c h w -> b c t h w", b=B)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = rearrange(h, "b c t h w -> (b t) c h w")
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = rearrange(h, "(b t) c h w -> b c t h w", b=B)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class ResnetCausalBlock1D(nn.Module):
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
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size=3, stride=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels, norm_type=self.norm_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size=3, stride=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = CausalConv1d(in_channels, out_channels, kernel_size=3, stride=1)
            else:
                self.nin_shortcut = CausalConv1d(in_channels, out_channels, kernel_size=1, stride=1)

        if zero_init:
            self.conv2.conv.weight.data.zero_()
            self.conv2.conv.bias.data.zero_()

        self.use_checkpoint = use_checkpoint

    def forward(self, x, temb):
        if self.use_checkpoint:
            assert temb is None, "checkpointing not supported with temb"
            return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)
        else:
            return self._forward(x, temb)

    def _forward(self, x, temb=None):
        B = x.shape[0]
        h = x

        h = rearrange(h, "(b s) c t -> (b t) c s", b=B)
        h = self.norm1(h)
        h = nonlinearity(h)
        h = rearrange(h, "(b t) c s -> (b s) c t", b=B)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = rearrange(h, "(b s) c t -> (b t) c s", b=B)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = rearrange(h, "(b t) c s -> (b s) c t", b=B)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class EncoderCausal3D(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        spatial_ds=None,
        tempo_ds=None,
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
        self.norm_type = norm_type
        self.fix_encoder = ignore_kwargs.get("fix_encoder", False)

        make_conv_cls = self._make_conv()
        make_attn_cls = self._make_attn()
        make_resblock_cls = self._make_resblock()

        self.conv_in = make_conv_cls(in_channels, self.ch, kernel_size=3, stride=1)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.spatial_ds = list(range(0, self.num_resolutions - 1)) if spatial_ds is None else spatial_ds
        self.tempo_ds = [self.num_resolutions - 2, self.num_resolutions - 3] if tempo_ds is None else tempo_ds
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
                    ResnetCausalBlock1D(
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

            if i_level in self.spatial_ds:
                down.downsample = Downsample(block_in, resamp_with_conv)
                if i_level in self.tempo_ds:
                    down_temporal.downsample = TimeDownsampleResCausal2x(block_in, block_in)

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
        self.mid.attn_1 = make_attn_cls(block_in, norm_type=self.norm_type)
        
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
        )

    def _make_attn(self) -> Callable:
        return make_attn

    def _make_resblock(self) -> Callable:
        return ResnetCausalBlock

    def _make_conv(self) -> Callable:
        return CausalConv3d

    def forward(self, x):
        temb = None
        B, _, T, H, W = x.shape
        hs = [self.conv_in(x)]

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = spatial_temporal_resblk(
                    hs[-1], self.down[i_level].block[i_block], self.down_temporal[i_level].block[i_block], temb
                )
                hs.append(h)

            if i_level in self.spatial_ds:
                # spatial downsample
                htmp = einops.rearrange(hs[-1], "b c t h w -> (b t) c h w")
                htmp = self.down[i_level].downsample(htmp)
                htmp = einops.rearrange(htmp, "(b t) c h w -> b c t h w", b=B, t=T)

                # temporal downsample
                B, _, T, H, W = htmp.shape
                if i_level in self.tempo_ds:
                    htmp = self.down_temporal[i_level].downsample(htmp)

                hs.append(htmp)
                B, _, T, H, W = htmp.shape

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        B, C, T, H, W = h.shape
        h = einops.rearrange(h, "b c t h w -> (b t) c h w")
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = einops.rearrange(h, "(b t) c h w -> b c t h w", b=B)
        h = self.conv_out(h)

        return h


class EncoderCausal3DPadding(EncoderCausal3D):
    def __init__(self, *args, **ignore_kwargs):
        super().__init__(*args, **ignore_kwargs)

        self.time_downsample_factor = ignore_kwargs.get("time_downsample_factor", 4)
        self.init_pad_mode = ignore_kwargs.get("init_pad_mode", "replicate")
        self.time_padding = self.time_downsample_factor - 1
        if self.fix_encoder:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        video_len = x.shape[2]
        if video_len % self.time_downsample_factor != 0:
            x = pad_at_dim(x, (self.time_padding, 0), dim=2, pad_mode=self.init_pad_mode, value=0.0)
        return super().forward(x)


class DecoderCausal3D(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        spatial_us=None,
        tempo_us=None,
        num_res_blocks,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
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
        self.norm_type = norm_type
        self.fix_decoder = ignorekwargs.get("fix_decoder", False)

        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]

        make_attn_cls = self._make_attn()
        make_resblock_cls = self._make_resblock()
        make_conv_cls = self._make_conv()

        self.conv_in = make_conv_cls(z_channels, block_in, kernel_size=3, stride=1)

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
        self.spatial_us = list(range(1, self.num_resolutions)) if spatial_us is None else spatial_us
        self.tempo_us = [1, 2] if tempo_us is None else tempo_us
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
            if i_level in self.spatial_us:
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
                    ResnetCausalBlock1D(
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
                up_temporal.upsample = TimeUpsampleResCausal2x(block_in, block_in)
            self.up_temporal.insert(0, up_temporal)

        # end
        self.norm_out = Normalize(block_in, norm_type=self.norm_type)
        self.conv_out = make_conv_cls(block_in, out_ch, kernel_size=3, stride=1)

    def _make_attn(self) -> Callable:
        return make_attn

    def _make_resblock(self) -> Callable:
        return ResnetCausalBlock

    def _make_conv(self) -> Callable:
        return CausalConv3d

    def get_last_layer(self, **kwargs):
        try:
            return self.conv_out.conv.weight
        except:
            return self.conv_out.weight

    def forward(self, z, **kwargs):
        temb = None
        B, _, T, H, W = z.shape
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

            if i_level in self.spatial_us:
                # spatial upsample
                h = einops.rearrange(h, "b c t h w -> (b t) c h w")
                h = self.up[i_level].upsample(h)
                h = einops.rearrange(h, "(b t) c h w -> b c t h w", b=B, t=T)

                # temporal upsample
                B, _, T, H, W = h.shape
                if i_level in self.tempo_us:
                    h = self.up_temporal[i_level].upsample(h)
                B, _, T, H, W = h.shape

        # end
        if self.give_pre_end:
            return h

        B, C, T, H, W = h.shape
        h = einops.rearrange(h, "b c t h w -> (b t) c h w")
        h = self.norm_out(h)
        h = rearrange(h, "(b t) c h w -> b c t h w", b=B)
        h = nonlinearity(h)
        h = self.conv_out(h, **kwargs)

        if self.tanh_out:
            h = torch.tanh(h)

        return h


class DecoderCausal3DPadding(DecoderCausal3D):
    def __init__(self, *args, **ignore_kwargs):
        super().__init__(*args, **ignore_kwargs)

        self.time_downsample_factor = ignore_kwargs.get("time_downsample_factor", 4)
        self.time_padding = self.time_downsample_factor - 1
        if self.fix_decoder:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = super().forward(x)
        return x[:, :, self.time_padding :, :, :]
