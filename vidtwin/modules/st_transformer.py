import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp
import torch.nn.functional as F
approx_gelu = lambda: nn.GELU(approximate="tanh")

from collections.abc import Iterable

from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from pathlib import Path
from omegaconf import ListConfig
from torch.cuda.amp import autocast

from einops import rearrange, repeat, reduce, pack, unpack
import pickle

def set_grad_checkpoint(model, use_fp32_attention=False, gc_step=1):
    assert isinstance(model, nn.Module)

    def set_attr(module):
        module.grad_checkpointing = True
        module.fp32_attention = use_fp32_attention
        module.grad_checkpointing_step = gc_step

    model.apply(set_attr)


def auto_grad_checkpoint(module, *args, **kwargs):
    if getattr(module, "grad_checkpointing", False):
        if not isinstance(module, Iterable):
            return checkpoint(module, *args, **kwargs)
        gc_step = module[0].grad_checkpointing_step
        return checkpoint_sequential(module, gc_step, *args, **kwargs)
    return module(*args, **kwargs)


def get_layernorm(hidden_size: torch.Tensor, eps: float, affine: bool, use_kernel: bool):
    if use_kernel:
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(hidden_size, elementwise_affine=affine, eps=eps)
        except ImportError:
            raise RuntimeError("FusedLayerNorm not available. Please install apex.")
    else:
        return nn.LayerNorm(hidden_size, eps, elementwise_affine=affine)


def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


class T2IFinalLayer(nn.Module):
    """
    The final layer of PixArt.
    """

    def __init__(self, hidden_size, num_patch, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, num_patch * out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size**0.5)
        self.out_channels = out_channels

    def forward(self, x):
        shift, scale = (self.scale_shift_table[None]).chunk(2, dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        enable_flashattn: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flashattn = enable_flashattn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, causal: bool) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
        if self.enable_flashattn:
            qkv_permute_shape = (2, 0, 1, 3, 4)
        else:
            qkv_permute_shape = (2, 0, 3, 1, 4)
        qkv = qkv.view(qkv_shape).permute(qkv_permute_shape)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.enable_flashattn:
            from flash_attn import flash_attn_func
            
            x = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale,
                causal=causal,
            )
        else:
            # raise NotImplementedError
            dtype = q.dtype
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # translate attn to float32
            attn = attn.to(torch.float32)
            attn = attn.softmax(dim=-1)
            attn = attn.to(dtype)  # cast back attn to original dtype
            attn = self.attn_drop(attn)
            x = attn @ v

        x_output_shape = (B, N, C)
        if not self.enable_flashattn:
            x = x.transpose(1, 2)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class GroupAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        enable_flashattn: bool = False,
        group_size: int = 4,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flashattn = enable_flashattn
        self.group_size = group_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, causal: bool) -> torch.Tensor:
        B, N, C = x.shape
        assert N % self.group_size == 0, "sequence length should be divisible by group_size"
        G = N // self.group_size
        if self.enable_flashattn:
            qkv_permute_shape = (2, 0, 1, 3, 4)
        else:
            qkv_permute_shape = (2, 0, 3, 1, 4)
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim).permute(qkv_permute_shape)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        

        if self.enable_flashattn:
            # reshape to (B, G, 4, H, D)
            q = q.view(B * G, self.group_size, self.num_heads, self.head_dim)
            k = k.view(B * G, self.group_size, self.num_heads, self.head_dim)
            v = v.view(B * G, self.group_size, self.num_heads, self.head_dim)
            from flash_attn import flash_attn_func

            # modify flash_attn_func to support the new shape
            x = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale,
                causal=causal,
            ).reshape(B, N, C)
        else:
            q = rearrange(q, "B H S D -> (B G) H N D", G=G)
            k = rearrange(k, "B H S D -> (B G) H N D", G=G)
            v = rearrange(v, "B H S D -> (B G) H N D", G=G)
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v)
            x = rearrange(x, "(B G) H N D -> B S (H D)", G=G, S=N)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class PatchEmbed3D(nn.Module):
    """Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self,
        patch_size=(2, 4, 4),
        in_chans=3,
        embed_dim=96,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # (B 768, 16, 14, 14) patchify, for each patch, we use 768 vector to represent it
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCTHW -> BNC
        return x



class STBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        d_s=None,
        d_t=None,
        mlp_ratio=4.0,
        drop_path=0.0,
        enable_flashattn=True,
        enable_layernorm_kernel=False,
        temporal_casual=True,
        no_temporal=False,
        temporal_group = False,
        group_size = 1
        # enable_sequence_parallelism=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.enable_flashattn = enable_flashattn
        
        self.attn_cls = Attention
        self.no_temporal = no_temporal
        self.attn_group = GroupAttention
        self.temporal_group = temporal_group
        self.group_size = group_size

        self.norm1 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.attn = self.attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            enable_flashattn=enable_flashattn,
        )
        self.norm2 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

        # temporal attention
        self.d_s = d_s
        self.d_t = d_t
        if self.temporal_group:
            self.attn_temp = self.attn_group(
                hidden_size,
                num_heads=num_heads,
                qkv_bias=True,
                enable_flashattn=self.enable_flashattn,
                group_size=self.group_size,
            )
        else:
            self.attn_temp = self.attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            enable_flashattn=self.enable_flashattn,
            )
        self.temporal_casual = temporal_casual

    def forward(self, x, tpe=None):

        # B, T, S, C = x.shape[0]

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] 
        ).chunk(6, dim=1)
        x = x.to(torch.float64)
        x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa).to(torch.float64)

        # spatial branch
        x_s = rearrange(x_m, "B T S C -> (B T) S C", T=self.d_t, S=self.d_s)
        # print(x_s.dtype)
        # x_s = x_s.to(torch.float32)
        x_s = x_s.to(torch.bfloat16)
        x_s = self.attn(x_s, causal=False,).to(torch.bfloat16)
        x_s = rearrange(x_s, "(B T) S C -> B T S C", T=self.d_t, S=self.d_s)
        x = x + self.drop_path(gate_msa * x_s)

        if not self.no_temporal:
            # temporal branch
            x_t = rearrange(x, "B T S C -> (B S) T C", T=self.d_t, S=self.d_s)
            
            if tpe is not None:
                x_t = x_t + tpe
            x_t = x_t.to(torch.bfloat16)
            x_t = self.attn_temp(x_t, causal=self.temporal_casual,)
            x_t = rearrange(x_t, "(B S) T C -> B T S C", T=self.d_t, S=self.d_s).to(torch.bfloat16)
            x = x + self.drop_path(gate_msa * x_t)
        
        # mlp
        x = x.to(torch.float32)
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))
        x = x.to(torch.float32)
        
        return x


def get_1d_sincos_pos_embed(embed_dim, length, scale=1.0):
    pos = np.arange(0, length)[..., None] / scale
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, scale=1.0, base_size=None):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / scale
    if base_size is not None:
        grid_h *= base_size / grid_size[0]
        grid_w *= base_size / grid_size[1]
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def is_odd(n):
    return not divisible_by(n, 2)

def cast_tuple(t, length = 1):
    if isinstance(t, ListConfig):
        return tuple(t)
    return t if isinstance(t, tuple) else ((t,) * length)

class DepthToSpace(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.bs, self.bt = block_size

    def forward(self, x):
        B, C, N, H, W = x.size()
        x = x.view(B, self.bt, self.bs, self.bs, C // ((self.bs ** 2) * self.bt), N,  H, W)  # (B, bs, bs, bs, C//bs^2, N, H, W)
        x = x.permute(0, 4, 5, 1, 6, 2, 7, 3).contiguous()  # (B, C//bs^3, N, bs, H, bs, W, bs)
        x = x.view(B, C // ((self.bs ** 2) * self.bt), N * self.bt,  H * self.bs, W * self.bs)  # (B, C//bs^3, N * bs, H * bs, W * bs)
        # remove the first frame
        if self.bt > 1:
            x = x[:, :, 1:, :, :]
        else:
            x = x
        return x


# Swish Function 
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)



class STTransformer(nn.Module):
    def __init__(
        self,
        input_size=(1, 32, 32),
        in_channels=4,
        patch_size=(1, 2, 2),
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        pred_sigma=False,
        drop_path=0.0,
        no_temporal_pos_emb=False,
        space_scale=1.0,
        time_scale=1.0,
        freeze=None,
        enable_flashattn=False,
        enable_layernorm_kernel=False,
        temporal_casual=True,
        no_temporal=False,
        temporal_group=False,
        group_size=1,
    ):
        super().__init__()
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.input_size = input_size
        num_patches = np.prod([input_size[i] // patch_size[i] for i in range(3)])
        self.num_patches = num_patches
        self.num_temporal = input_size[0] // patch_size[0]
        self.num_spatial = num_patches // self.num_temporal
        self.num_heads = num_heads
        self.no_temporal_pos_emb = no_temporal_pos_emb
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.enable_flashattn = enable_flashattn
        self.enable_layernorm_kernel = enable_layernorm_kernel
        self.space_scale = space_scale
        self.time_scale = time_scale
        self.temporal_casual = temporal_casual
        self.temporal_group = temporal_group
        self.group_size = group_size

        self.register_buffer("pos_embed", self.get_spatial_pos_embed())
        self.register_buffer("pos_embed_temporal", self.get_temporal_pos_embed())

        self.x_embedder = PatchEmbed3D(patch_size, in_channels, hidden_size)
        self.no_temporal = no_temporal
       
        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList(
            [
                STBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    drop_path=drop_path[i],
                    enable_flashattn=self.enable_flashattn,
                    enable_layernorm_kernel=self.enable_layernorm_kernel,
                    d_t=self.num_temporal,
                    d_s=self.num_spatial,
                    temporal_casual=self.temporal_casual,
                    no_temporal=self.no_temporal,
                    temporal_group = self.temporal_group,
                    group_size = self.group_size
                )
                for i in range(self.depth)
            ]
        )
        self.final_layer = T2IFinalLayer(hidden_size, np.prod(self.patch_size), self.out_channels)

        # init model
        self.initialize_weights()
        self.initialize_temporal()
        if freeze is not None:
            assert freeze in ["not_temporal", "text"]
            if freeze == "not_temporal":
                self.freeze_not_temporal()
            elif freeze == "text":
                self.freeze_text()



    def forward(self, x):
        """
        Forward pass of STDiT.
        Args:
            x (torch.Tensor): latent representation of video; of shape [B, C, T, H, W]

        Returns:
            x (torch.Tensor): output latent representation; of shape [B, C, T, H, W]
        """

        x = rearrange(x, "B (T S) C -> B T S C", T=self.num_temporal, S=self.num_spatial)
        x = x + self.pos_embed

        with autocast(enabled=True):
            for i, block in enumerate(self.blocks):
                if i == 0:
                    tpe = self.pos_embed_temporal
                else:
                    tpe = None
                x = auto_grad_checkpoint(block, x, tpe)

        x = rearrange(x, "B T S C -> B (T S) C", T=self.num_temporal, S=self.num_spatial)
        return x

    def unpatchify(self, x):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """

        N_t, N_h, N_w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        T_p, H_p, W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )
        return x

    def unpatchify_old(self, x):
        c = self.out_channels
        t, h, w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        pt, ph, pw = self.patch_size

        x = x.reshape(shape=(x.shape[0], t, h, w, pt, ph, pw, c))
        x = rearrange(x, "n t h w r p q c -> n c t r h p w q")
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
        return imgs

    def get_spatial_pos_embed(self, grid_size=None):
        if grid_size is None:
            grid_size = self.input_size[1:]
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size,
            (grid_size[0] // self.patch_size[1], grid_size[1] // self.patch_size[2]),
            scale=self.space_scale,
        )
        pos_embed = torch.from_numpy(pos_embed).unsqueeze(0).requires_grad_(False)
        return pos_embed

    def get_temporal_pos_embed(self):
        pos_embed = get_1d_sincos_pos_embed(
            self.hidden_size,
            self.input_size[0] // self.patch_size[0],
            scale=self.time_scale,
        )
        pos_embed = torch.from_numpy(pos_embed).unsqueeze(0).requires_grad_(False)
        return pos_embed

    def freeze_not_temporal(self):
        for n, p in self.named_parameters():
            if "attn_temp" not in n:
                p.requires_grad = False

    def freeze_text(self):
        for n, p in self.named_parameters():
            if "cross_attn" in n:
                p.requires_grad = False

    def initialize_temporal(self):
        for block in self.blocks:
            nn.init.constant_(block.attn_temp.proj.weight, 0)
            nn.init.constant_(block.attn_temp.proj.bias, 0)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

class STTEncoder(STTransformer):
    def __init__(self, input_size=(1, 32, 32), in_channels=3, patch_size=(1, 2, 2), hidden_size=64, depth=12, num_heads=8, mlp_ratio=4, pred_sigma=False, drop_path=0, no_temporal_pos_emb=False, space_scale=1, time_scale=1, freeze=None, enable_flashattn=True, enable_layernorm_kernel=False, temporal_casual=True, no_temporal=False, temporal_group=False, group_size=1):
        super().__init__(input_size, in_channels, patch_size, hidden_size, depth, num_heads, mlp_ratio, pred_sigma, drop_path, no_temporal_pos_emb, space_scale, time_scale, freeze, enable_flashattn, enable_layernorm_kernel, temporal_casual, no_temporal, temporal_group, group_size)
        
    def forward(self, x):
        x = self.x_embedder(x)
        y = super().forward(x)
        y = rearrange(y, "B (T H W) C -> B C T H W", T=self.input_size[0], H=self.input_size[1]//self.patch_size[1], W=self.input_size[2]//self.patch_size[2])
        return y
    
    @property
    def device(self):
        return self.zero.device

    @classmethod
    def init_and_load_from(cls, path, strict = True):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = 'cpu')

        assert 'config' in pkg, 'model configs were not found in this saved checkpoint'

        config = pickle.loads(pkg['config'])
        tokenizer = cls(**config)
        tokenizer.load(path, strict = strict)
        return tokenizer

    def save(self, path, overwrite = True):
        path = Path(path)
        assert overwrite or not path.exists(), f'{str(path)} already exists'

        pkg = dict(
            model_state_dict = self.state_dict(),
            version =self.__version__,
            config = self._configs
        )

        torch.save(pkg, str(path))

    def load(self, path, strict = True):
        path = Path(path)
        assert path.exists()

        pkg = torch.load(str(path))
        state_dict = pkg.get('model_state_dict')
        version = pkg.get('version')

        assert exists(state_dict)

        if exists(version):
            print(f'loading checkpointed tokenizer from version {version}')

        self.load_state_dict(state_dict, strict = strict)

        
    @torch.no_grad()
    def tokenize(self, video):
        self.eval()
        return self.forward(video, return_codes = True)
    
    def debug_model(self, x, layer):
        if torch.isnan(x).any():
            print('x has nan')
            print(layer)
            import sys
            sys.exit()



class STTDecoder(STTransformer):
    def __init__(self, input_size=(1, 32, 32), in_channels=3, patch_size=(1, 2, 2), hidden_size=1152, depth=12, num_heads=16, mlp_ratio=4, pred_sigma=False, drop_path=0, no_temporal_pos_emb=False,  space_scale=1, time_scale=1, freeze=None, enable_flashattn=True, enable_layernorm_kernel=False, temporal_casual=True, no_temporal=False):
        super().__init__(input_size, in_channels, patch_size, hidden_size, depth, num_heads, mlp_ratio,pred_sigma, drop_path, no_temporal_pos_emb, space_scale, time_scale, freeze, enable_flashattn, enable_layernorm_kernel, temporal_casual, no_temporal)
        self.final_layer = T2IFinalLayer(hidden_size, np.prod(self.patch_size), self.out_channels)
         
    def forward(self, x):
        x = rearrange(x, "B C T H W -> B (T H W) C")
        y = super().forward(x)
        y = self.final_layer(y)
        y = self.unpatchify(y)
        return y
    
    @property
    def device(self):
        return self.zero.device

    @classmethod
    def init_and_load_from(cls, path, strict = True):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = 'cpu')

        assert 'config' in pkg, 'model configs were not found in this saved checkpoint'

        config = pickle.loads(pkg['config'])
        tokenizer = cls(**config)
        tokenizer.load(path, strict = strict)
        return tokenizer

    def save(self, path, overwrite = True):
        path = Path(path)
        assert overwrite or not path.exists(), f'{str(path)} already exists'

        pkg = dict(
            model_state_dict = self.state_dict(),
            version = self.__version__,
            config = self._configs
        )

        torch.save(pkg, str(path))

    def load(self, path, strict = True):
        path = Path(path)
        assert path.exists()

        pkg = torch.load(str(path))
        state_dict = pkg.get('model_state_dict')
        version = pkg.get('version')

        assert exists(state_dict)

        if exists(version):
            print(f'loading checkpointed tokenizer from version {version}')

        self.load_state_dict(state_dict, strict = strict)

        
    @torch.no_grad()
    def tokenize(self, video):
        self.eval()
        return self.forward(video, return_codes = True)
    
    def debug_model(self, x, layer):
        if torch.isnan(x).any():
            print('x has nan')
            print(layer)
            import sys
            sys.exit()
    
    def get_last_layer(self):
        return self.final_layer.linear.weight