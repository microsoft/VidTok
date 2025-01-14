import argparse
import os
import sys
sys.path.append(os.getcwd())

import warnings
warnings.filterwarnings("ignore")

import time
from contextlib import nullcontext
from omegaconf import OmegaConf
from torch import autocast
from tqdm import tqdm

import numpy as np
import torch
from einops import rearrange
from lightning.pytorch import seed_everything

from vidtok.data.vidtok import VidTokValDataset
from vidtok.modules.lpips import LPIPS
from vidtok.modules.util import (compute_psnr, compute_ssim,
                                 instantiate_from_config, print0)


def load_model_from_config(config, ckpt, ignore_keys=[], verbose=False):
    config = OmegaConf.load(config)
    config.model.params.ckpt_path = ckpt
    config.model.params.ignore_keys = ignore_keys
    config.model.params.verbose = verbose
    model = instantiate_from_config(config.model)
    return model


class MultiVideoDataset(VidTokValDataset):
    def __init__(
        self,
        data_dir,
        meta_path=None,
        input_height=256,
        input_width=256,
        num_frames_per_batch=17,
        sample_fps=30,
    ):
        super().__init__(
            data_dir=data_dir,
            meta_path=meta_path,
            video_params={
                "input_height": input_height,
                "input_width": input_width,
                "sample_num_frames": num_frames_per_batch,
                "sample_fps": sample_fps,
            },
            pre_load_frames=True,
            last_frames_handle="repeat",
        )

    def __getitem__(self, idx):
        frames = super().__getitem__(idx)["jpg"]
        return frames


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="full"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/vidtok_kl_causal_488_4chn.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/vidtok_kl_causal_488_4chn.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--input_video_path",
        type=str,
        default="assets/example.mp4",
        help="path to the input video",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./",
        help="root folder",
    )
    parser.add_argument(
        "--meta_path",
        type=str,
        default=None,
        help="path to the .csv meta file",
    )
    parser.add_argument(
        "--input_height",
        type=int,
        default=256,
        help="height of the input video",
    )
    parser.add_argument(
        "--input_width",
        type=int,
        default=256,
        help="width of the input video",
    )
    parser.add_argument(
        "--num_frames_per_batch",
        type=int,
        default=17,
        help="number of frames per batch",
    )
    parser.add_argument(
        "--sample_fps",
        type=int,
        default=30,
        help="sample fps",
    )

    args = parser.parse_args()
    seed_everything(args.seed)

    print0(f"[bold red]\[scripts.inference_evaluate][/bold red] Evaluating model {args.ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    precision_scope = autocast if args.precision == "autocast" else nullcontext

    model = load_model_from_config(args.config, args.ckpt)
    model.to(device).eval()

    dataset = MultiVideoDataset(
        data_dir=args.data_dir,
        meta_path=args.meta_path,
        input_height=args.input_height,
        input_width=args.input_width,
        num_frames_per_batch=args.num_frames_per_batch,
        sample_fps=args.sample_fps
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    perceptual_loss = LPIPS().eval()
    perceptual_loss = perceptual_loss.to(device)

    psnrs, ssims, lpipss = [], [], []

    with torch.no_grad(), precision_scope("cuda"):
        tic = time.time()
        for i, input in tqdm(enumerate(dataloader)):
            input = input.to(device)
            _, output, reg_log = model(input)

            output = output.clamp(-1, 1)
            input, output = map(lambda x: (x + 1) / 2, (input, output))

            if input.dim() == 5:
                input = rearrange(input, "b c t h w -> (b t) c h w")
                assert output.dim() == 5
                output = rearrange(output, "b c t h w -> (b t) c h w")

            psnr = compute_psnr(input, output)
            ssim = compute_ssim(input, output)
            lpips = perceptual_loss(input * 2 - 1, output * 2 - 1).mean()

            psnrs.append(psnr.item())
            ssims.append(ssim.item())
            lpipss.append(lpips.item())

        toc = time.time()
        print0(
            f"[bold red]\[scripts.inference_evaluate][/bold red] PSNR: {np.mean(psnrs):.4f}, SSIM: {np.mean(ssims):.4f}, LPIPS: {np.mean(lpipss):.4f}"
        )
        print0(f"[bold red]\[scripts.inference_evaluate][/bold red] Time taken: {toc - tic:.2f}s")


if __name__ == "__main__":
    main()
