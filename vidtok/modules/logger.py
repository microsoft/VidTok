import os
import numpy as np
import einops
import imageio
from typing import Union
from matplotlib import pyplot as plt
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # UnidentifiedImageError: https://github.com/python-pillow/Pillow/issues/5631
from pathlib import Path

import torch
import torchvision
import wandb

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from .util import exists, isheatmap


class ImageVideoLogger(Callback):
    def __init__(
        self,
        batch_frequency,
        max_samples,
        clamp=True,
        increase_log_steps=True,
        batch_frequency_val=None,
        video_fps=8,
        rescale=True,
        disabled=False,
        log_on_batch_idx=True,  # log on batch_idx instead of global_step. global_step is fixed in validation. batch_idx restarts at each validation
        log_first_step=True,
        log_images_kwargs=None,
        log_videos_kwargs=None,
        log_before_first_step=True,
        enable_autocast=True,
    ):
        super().__init__()
        self.enable_autocast = enable_autocast
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.batch_freq_val = batch_frequency_val if batch_frequency_val is not None else batch_frequency
        self.video_fps = video_fps
        self.max_samples = max_samples
        self.log_steps = [2**n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_videos_kwargs = log_videos_kwargs if log_videos_kwargs else {}
        self.log_first_step = log_first_step
        self.log_before_first_step = log_before_first_step

    @rank_zero_only
    def log_img_local(
        self,
        save_dir,
        split,
        images,
        global_step,
        current_epoch,
        batch_idx,
        pl_module: Union[None, pl.LightningModule] = None,
    ):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            if isheatmap(images[k]):
                fig, ax = plt.subplots()
                ax = ax.matshow(
                    images[k].cpu().numpy(), cmap="hot", interpolation="lanczos"
                )
                plt.colorbar(ax)
                plt.axis("off")

                filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                    k, global_step, current_epoch, batch_idx
                )
                os.makedirs(root, exist_ok=True)
                path = os.path.join(root, filename)
                plt.savefig(path)
                plt.close()
            else:
                if images[k].ndim == 5:
                    images[k] = einops.rearrange(images[k], "b c t h w -> (b t) c h w")
                nrow = self.log_images_kwargs.get("n_rows", 8)
                grid = torchvision.utils.make_grid(images[k], nrow=nrow)
                if self.rescale:
                    grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                grid = grid.numpy()
                grid = (grid * 255).astype(np.uint8)
                filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                    k, global_step, current_epoch, batch_idx
                )
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                img = Image.fromarray(grid)
                img.save(path)
                if exists(pl_module):
                    assert isinstance(
                        pl_module.logger, WandbLogger
                    ), "logger_log_image only supports WandbLogger currently"
                    pl_module.logger.log_image(
                        key=f"{split}/{k}",
                        images=[
                            img,
                        ],
                        step=pl_module.global_step,
                    )

    @rank_zero_only
    def log_vid_local(
        self,
        save_dir,
        split,
        videos,
        global_step,
        current_epoch,
        batch_idx,
        pl_module: Union[None, pl.LightningModule] = None,
    ):
        root = os.path.join(save_dir, "videos", split)
        for k in videos:
            # if is video, we can add captions
            if isinstance(videos[k], torch.Tensor) and videos[k].ndim == 5:
                if self.rescale:
                    videos[k] = (videos[k] + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                frames = [videos[k][:, :, i] for i in range(videos[k].shape[2])]
                frames = [torchvision.utils.make_grid(each, nrow=4) for each in frames]
                frames = [einops.rearrange(each, "c h w -> 1 c h w") for each in frames]
                frames = torch.clamp(torch.cat(frames, dim=0), min=0.0, max=1.0)
                frames = (frames.numpy() * 255).astype(np.uint8)

                filename = "{}_gs-{:06}_e-{:06}_b-{:06}.gif".format(
                    k, global_step, current_epoch, batch_idx
                )
                os.makedirs(root, exist_ok=True)
                path = os.path.join(root, filename)
                save_numpy_as_gif(frames, path, duration=1 / self.video_fps)
                if exists(pl_module):
                    assert isinstance(
                        pl_module.logger, WandbLogger
                    ), "log_videos only supports WandbLogger currently"
                    wandb.log({f"{split}/{k}": wandb.Video(frames, fps=self.video_fps)})  # k is str

    @rank_zero_only
    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (
            (self.check_frequency(check_idx) or self.check_frequency_val(batch_idx, split))
            and hasattr(pl_module, "log_images")  # batch_idx % self.batch_freq == 0
            and callable(pl_module.log_images)
            and self.max_samples > 0
        ):
            logger = type(pl_module.logger)
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad(), torch.autocast(enabled=self.enable_autocast, device_type="cuda"):
                images = pl_module.log_images(batch)

            for k in images:
                N = min(images[k].shape[0], self.max_samples)
                if not isheatmap(images[k]):
                    images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().float().cpu()
                    if self.clamp and not isheatmap(images[k]):
                        images[k] = torch.clamp(images[k], -1.0, 1.0)

            self.log_img_local(
                pl_module.logger.save_dir,
                split,
                images,
                pl_module.global_step,
                pl_module.current_epoch,
                batch_idx,
                pl_module=pl_module
                if isinstance(pl_module.logger, WandbLogger)
                else None,
            )

            if is_train:
                pl_module.train()

    @rank_zero_only
    def log_vid(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (
            (self.check_frequency(check_idx) or self.check_frequency_val(batch_idx, split))
            and hasattr(pl_module, "log_videos")  # batch_idx % self.batch_freq == 0
            and callable(pl_module.log_videos)
            and self.max_samples > 0
        ):
            logger = type(pl_module.logger)
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad(), torch.autocast(enabled=self.enable_autocast, device_type="cuda"):
                videos = pl_module.log_videos(
                    batch, split=split, **self.log_videos_kwargs
                )

            for k in videos:
                N = min(videos[k].shape[0], self.max_samples)
                videos[k] = videos[k][:N]
                if isinstance(videos[k], torch.Tensor):
                    videos[k] = videos[k].detach().float().cpu()
                    if self.clamp:
                        videos[k] = torch.clamp(videos[k], -1.0, 1.0)

            self.log_vid_local(
                pl_module.logger.save_dir,
                split,
                videos,
                pl_module.global_step,
                pl_module.current_epoch,
                batch_idx,
                pl_module=pl_module
                if isinstance(pl_module.logger, WandbLogger)
                else None,
            )

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
            check_idx > 0 or self.log_first_step
        ):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                pass
            return True
        return False

    def check_frequency_val(self, check_idx, split):
        if 'val' in split:
            if ((check_idx % self.batch_freq_val) == 0) and (
                check_idx > 0 or self.log_first_step):
                return True
        return False

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")
            self.log_vid(pl_module, batch, batch_idx, split="train")

    @rank_zero_only
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.log_before_first_step and pl_module.global_step == 0:
            self.log_img(pl_module, batch, batch_idx, split="train")
            self.log_vid(pl_module, batch, batch_idx, split="train")

    @rank_zero_only
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs
    ):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
            self.log_vid(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, "calibrate_grad_norm"):
            if (
                pl_module.calibrate_grad_norm and batch_idx % 25 == 0
            ) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


def save_numpy_as_gif(frames, path, duration=None):
    """
    save numpy array as gif file
    """
    image_list = []
    for frame in frames:
        image = frame.transpose(1, 2, 0)
        image_list.append(image)
    if duration:
        imageio.mimsave(path, image_list, format="GIF", duration=duration, loop=0)
    else:
        imageio.mimsave(path, image_list, format="GIF", loop=0)
