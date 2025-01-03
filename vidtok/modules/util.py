import importlib
import random
import os
import einops
import numpy as np
from inspect import isfunction
from rich import print
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR, StepLR

def get_valid_dirs(dir1: str, dir2: str, dir3: Union[None, str] = None) -> Union[None, str]:
    if (dir1 is not None) and os.path.isdir(dir1):
        return dir1
    elif (dir2 is not None) and os.path.isdir(dir2):
        return dir2
    elif (dir3 is not None) and os.path.isdir(dir3):
        return dir3
    else:
        return None


def get_valid_paths(path1: str, path2: str, path3: Union[None, str] = None) -> Union[None, str]:
    if (path1 is not None) and os.path.isfile(path1):
        return path1
    elif (path2 is not None) and os.path.isfile(path2):
        return path2
    elif (path3 is not None) and os.path.isfile(path3):
        return path3
    else:
        return None


@rank_zero_only
def print0(*args, **kwargs):
    print(*args, **kwargs)


def seed_anything(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def isheatmap(x):
    if not isinstance(x, torch.Tensor):
        return False

    return x.ndim == 2


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False, invalidate_cache=True):
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def checkpoint(func, inputs, params, flag):
    # https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/nn.py
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    # https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/nn.py
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        ctx.gpu_autocast_kwargs = {
            "enabled": torch.is_autocast_enabled(),
            "dtype": torch.get_autocast_gpu_dtype(),
            "cache_enabled": torch.is_autocast_cache_enabled(),
        }
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        # Ensure all tensors have requires_grad set to True
        ctx.input_params = [p.requires_grad_(True) for p in ctx.input_params]
        with torch.enable_grad(), torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def compute_psnr(x, y):
    if x.dim() == 5:
        x = einops.rearrange(x, "b c t h w -> (b t) c h w")
        assert y.dim() == 5
        y = einops.rearrange(y, "b c t h w -> (b t) c h w")
    EPS = 1e-8
    mse = torch.mean((x - y) ** 2, dim=[1, 2, 3])
    psnr = -10 * torch.log10(mse + EPS)
    return psnr.mean(dim=0)


def compute_ssim(x, y):
    if x.dim() == 5:
        x = einops.rearrange(x, "b c t h w -> (b t) c h w")
        assert y.dim() == 5
        y = einops.rearrange(y, "b c t h w -> (b t) c h w")
    kernel_size = 11
    kernel_sigma = 1.5
    k1 = 0.01
    k2 = 0.03

    f = max(1, round(min(x.size()[-2:]) / 256))
    if f > 1:
        x = F.avg_pool2d(x, kernel_size=f)
        y = F.avg_pool2d(y, kernel_size=f)

    kernel = gaussian_filter(kernel_size, kernel_sigma, device=x.device, dtype=x.dtype).repeat(x.size(1), 1, 1, 1)

    _compute_ssim_per_channel = _ssim_per_channel_complex if x.dim() == 5 else _ssim_per_channel
    ssim_map, cs_map = _compute_ssim_per_channel(x=x, y=y, kernel=kernel, data_range=1, k1=k1, k2=k2)
    ssim_val = ssim_map.mean(1)

    return ssim_val.mean(dim=0)


def _ssim_per_channel(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: torch.Tensor,
    data_range: Union[float, int] = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Calculate Structural Similarity (SSIM) index for X and Y per channel.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        kernel: 2D Gaussian kernel.
        data_range: Maximum value range of images (usually 1.0 or 255).
        k1: Algorithm parameter, K1 (small constant, see [1]).
        k2: Algorithm parameter, K2 (small constant, see [1]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.

    Returns:
        Full Value of Structural Similarity (SSIM) index.
    """
    if x.size(-1) < kernel.size(-1) or x.size(-2) < kernel.size(-2):
        raise ValueError(
            f"Kernel size can't be greater than actual input size. "
            f"Input size: {x.size()}. Kernel size: {kernel.size()}"
        )

    c1 = k1**2
    c2 = k2**2
    n_channels = x.size(1)
    mu_x = F.conv2d(x, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu_y = F.conv2d(y, weight=kernel, stride=1, padding=0, groups=n_channels)

    mu_xx = mu_x**2
    mu_yy = mu_y**2
    mu_xy = mu_x * mu_y

    sigma_xx = F.conv2d(x**2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xx
    sigma_yy = F.conv2d(y**2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_yy
    sigma_xy = F.conv2d(x * y, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xy

    # Contrast sensitivity (CS) with alpha = beta = gamma = 1.
    cs = (2.0 * sigma_xy + c2) / (sigma_xx + sigma_yy + c2)

    # Structural similarity (SSIM)
    ss = (2.0 * mu_xy + c1) / (mu_xx + mu_yy + c1) * cs

    ssim_val = ss.mean(dim=(-1, -2))
    cs = cs.mean(dim=(-1, -2))
    return ssim_val, cs


def _ssim_per_channel_complex(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: torch.Tensor,
    data_range: Union[float, int] = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Calculate Structural Similarity (SSIM) index for Complex X and Y per channel.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W, 2)`.
        y: A target tensor. Shape :math:`(N, C, H, W, 2)`.
        kernel: 2-D gauss kernel.
        data_range: Maximum value range of images (usually 1.0 or 255).
        k1: Algorithm parameter, K1 (small constant, see [1]).
        k2: Algorithm parameter, K2 (small constant, see [1]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.

    Returns:
        Full Value of Complex Structural Similarity (SSIM) index.
    """
    n_channels = x.size(1)
    if x.size(-2) < kernel.size(-1) or x.size(-3) < kernel.size(-2):
        raise ValueError(
            f"Kernel size can't be greater than actual input size. Input size: {x.size()}. "
            f"Kernel size: {kernel.size()}"
        )

    c1 = k1**2
    c2 = k2**2

    x_real = x[..., 0]
    x_imag = x[..., 1]
    y_real = y[..., 0]
    y_imag = y[..., 1]

    mu1_real = F.conv2d(x_real, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu1_imag = F.conv2d(x_imag, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu2_real = F.conv2d(y_real, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu2_imag = F.conv2d(y_imag, weight=kernel, stride=1, padding=0, groups=n_channels)

    mu1_sq = mu1_real.pow(2) + mu1_imag.pow(2)
    mu2_sq = mu2_real.pow(2) + mu2_imag.pow(2)
    mu1_mu2_real = mu1_real * mu2_real - mu1_imag * mu2_imag
    mu1_mu2_imag = mu1_real * mu2_imag + mu1_imag * mu2_real

    compensation = 1.0

    x_sq = x_real.pow(2) + x_imag.pow(2)
    y_sq = y_real.pow(2) + y_imag.pow(2)
    x_y_real = x_real * y_real - x_imag * y_imag
    x_y_imag = x_real * y_imag + x_imag * y_real

    sigma1_sq = F.conv2d(x_sq, weight=kernel, stride=1, padding=0, groups=n_channels) - mu1_sq
    sigma2_sq = F.conv2d(y_sq, weight=kernel, stride=1, padding=0, groups=n_channels) - mu2_sq
    sigma12_real = F.conv2d(x_y_real, weight=kernel, stride=1, padding=0, groups=n_channels) - mu1_mu2_real
    sigma12_imag = F.conv2d(x_y_imag, weight=kernel, stride=1, padding=0, groups=n_channels) - mu1_mu2_imag
    sigma12 = torch.stack((sigma12_imag, sigma12_real), dim=-1)
    mu1_mu2 = torch.stack((mu1_mu2_real, mu1_mu2_imag), dim=-1)
    # Set alpha = beta = gamma = 1.
    cs_map = (sigma12 * 2 + c2 * compensation) / (sigma1_sq.unsqueeze(-1) + sigma2_sq.unsqueeze(-1) + c2 * compensation)
    ssim_map = (mu1_mu2 * 2 + c1 * compensation) / (mu1_sq.unsqueeze(-1) + mu2_sq.unsqueeze(-1) + c1 * compensation)
    ssim_map = ssim_map * cs_map

    ssim_val = ssim_map.mean(dim=(-2, -3))
    cs = cs_map.mean(dim=(-2, -3))

    return ssim_val, cs


def gaussian_filter(
    kernel_size: int, sigma: float, device: Optional[str] = None, dtype: Optional[type] = None
) -> torch.Tensor:
    r"""Returns 2D Gaussian kernel N(0,`sigma`^2)
    Args:
        kernel_size: Size of the kernel
        sigma: Std of the distribution
        device: target device for kernel generation
        dtype: target data type for kernel generation
    Returns:
        gaussian_kernel: Tensor with shape (1, kernel_size, kernel_size)
    """
    coords = torch.arange(kernel_size, dtype=dtype, device=device)
    coords -= (kernel_size - 1) / 2.0

    g = coords**2
    g = (-(g.unsqueeze(0) + g.unsqueeze(1)) / (2 * sigma**2)).exp()

    g /= g.sum()
    return g.unsqueeze(0)


def instantiate_lrscheduler_from_config(optimizer, config, name='main-LR'):
    """
    Instantiate a learning rate scheduler from a config dict.
    If use timm, must add the following codes to the LightningModule:

    def lr_scheduler_step(self, scheduler, metric):
        if 'timm.scheduler' in self.lr_scheduler_config.target:
            scheduler.step(epoch=self.current_epoch)
        else:
            if metric is None:
                scheduler.step()
            else:
                scheduler.step(metric)
    """
    assert 'target' in config, 'Expected key `target` to instantiate.'
    if ('torch.optim' in config.target) or ('timm.scheduler' in config.target):
        scheduler = get_obj_from_str(config["target"])(optimizer, **config.get("params", dict()))
        lr_scheduler = {
            'scheduler': scheduler,
            'name': name
            }
    else:
        scheduler_init = instantiate_from_config(config)
        scheduler = LambdaLR(optimizer, lr_lambda=scheduler_init.schedule)
        lr_scheduler = {
            'scheduler': LambdaLR(optimizer, lr_lambda=scheduler_init.schedule),
            'name': name,
            'interval': 'step',
            'frequency': 1
            }
    return scheduler
