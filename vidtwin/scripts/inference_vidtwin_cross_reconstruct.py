import argparse
import datetime
import glob
import inspect
import os
import re
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from inspect import Parameter
from typing import Union
from matplotlib import pyplot as plt
from natsort import natsorted
from omegaconf import OmegaConf
from packaging import version
from PIL import Image
from pathlib import Path
from tqdm import tqdm

import torch
import torchvision
import wandb

import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only

import decord
import time
from einops import rearrange
from contextlib import nullcontext
from torch import autocast
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.io import write_video
from safetensors.torch import load_file as load_safetensors

from vidtok.modules.util import instantiate_from_config, print0


def load_model_from_config(config, ckpt, verbose=False):
    config = OmegaConf.load(config)
    print0(f"[bold red]\[vidtwin.scripts.inference_vidtwin_cross_reconstruct][/bold red] Loading model from {ckpt}")
    model = instantiate_from_config(config.model)

    if ckpt.endswith("ckpt"):
        sd = torch.load(ckpt, map_location="cpu")["state_dict"]
    elif ckpt.endswith("safetensors"):
        sd = load_safetensors(ckpt)
    else:
        raise NotImplementedError(f"Unknown checkpoint: {ckpt}")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print0(
        f"[bold red]\[vidtwin.scripts.inference_vidtwin_cross_reconstruct][/bold red] Restored from {ckpt} with {len(missing)} missing and {len(unexpected)} unexpected keys"
    )
    if len(missing) > 0:
        print0(f"[bold red]\[vidtwin.scripts.inference_vidtwin_cross_reconstruct][/bold red] Missing Keys: {missing}")
    if len(unexpected) > 0:
        print0(f"[bold red]\[vidtwin.scripts.inference_vidtwin_cross_reconstruct][/bold red] Unexpected Keys: {unexpected}")
    return model


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, video_path, input_height=128, input_width=128, sample_fps=8, num_frames_per_batch=16):
        decord.bridge.set_bridge("torch")
        self.video_path = video_path
        normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        self.transform = transforms.Compose([transforms.Resize(input_height, antialias=True),
                                             transforms.CenterCrop((input_height, input_width)),
                                             normalize,])

        self.video_reader = decord.VideoReader(video_path, num_threads=0)
        total_frames = len(self.video_reader)
        fps = self.video_reader.get_avg_fps()  # float

        interval = round(fps / sample_fps)
        frame_ids = list(range(0, total_frames, interval))
        self.frame_ids_batch = []
        for x in range(0, len(frame_ids), num_frames_per_batch):
            if len(frame_ids[x:x+num_frames_per_batch]) == num_frames_per_batch:
                self.frame_ids_batch.append(frame_ids[x:x+num_frames_per_batch])

    def __len__(self):
        return len(self.frame_ids_batch)

    def __getitem__(self, idx):
        frame_ids = self.frame_ids_batch[idx]
        frames = self.video_reader.get_batch(frame_ids).permute(0, 3, 1, 2).float() / 255.
        frames = self.transform(frames).permute(1, 0, 2, 3)
        return frames


def tensor_to_uint8(tensor):
    tensor = torch.clamp(tensor, -1.0, 1.0)
    tensor = (tensor + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
    tensor = (tensor.cpu().numpy() * 255).astype(np.uint8)
    return tensor


def main():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="full"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tvae3d/webvid_kl_f_16_128_884_8chn_80G4.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="xxxxx.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--output_video_dir",
        type=str,
        default="tmp",
        help="path to save the outputs",
    )
    parser.add_argument(
        "--input_video_path_structure",
        type=str,
        default="logs/assets/Nik.mp4",
        help="path to the input video",
    )
    parser.add_argument(
        "--input_video_path_dynamics",
        type=str,
        default="logs/assets/Nik.mp4",
        help="path to the input video",
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
        "--sample_fps",
        type=int,
        default=4,
        help="",
    )
    parser.add_argument(
        "--num_frames_per_batch",
        type=int,
        default=16,
        help="",
    )
    parser.add_argument(
        "--concate_input",
        type=str2bool,
        const=True,
        default=True,
        nargs="?",
        help="",
    )
    parser.add_argument(
        "--dynamics_split",
        type=str2bool,
        default=True,
        nargs="?",
        help="",
    )

    args = parser.parse_args()
    seed_everything(args.seed)

    print0(f"[bold red]\[vidtwin.scripts.inference_vidtwin_cross_reconstruct][/bold red] Evaluating model {args.ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    precision_scope = autocast if args.precision == "autocast" else nullcontext

    os.makedirs(args.output_video_dir, exist_ok=True)
    print(args.ckpt)
    print(args.config)
    model = load_model_from_config(args.config, args.ckpt)
    model.to(device).eval()

    dataset_structure = VideoDataset(args.input_video_path_structure, args.input_height, args.input_width, args.sample_fps, args.num_frames_per_batch)
    dataset_dynamics = VideoDataset(args.input_video_path_dynamics, args.input_height, args.input_width, args.sample_fps, args.num_frames_per_batch)
    min_len = min(len(dataset_structure), len(dataset_dynamics))
    dataset_structure = torch.utils.data.Subset(dataset_structure, range(min_len))
    dataset_dynamics = torch.utils.data.Subset(dataset_dynamics, range(min_len))
    dataloader_structure = torch.utils.data.DataLoader(dataset_structure, batch_size=1, shuffle=False)
    dataloader_dynamics = torch.utils.data.DataLoader(dataset_dynamics, batch_size=1, shuffle=False)

    inputs_structure = []
    inputs_dynamics = []
    outputs = []
    with torch.no_grad(), precision_scope("cuda"):
        tic = time.time()
        for i, input_structure, input_dynamics in zip(tqdm(range(min_len)), dataloader_structure, dataloader_dynamics):
            if input_structure.shape[2] <= 5:
                continue
            input_structure = input_structure.to(device)
            input_dynamics = input_dynamics.to(device)
            if args.dynamics_split:
                z, z_structure, *_ = model.encode(input_structure)
                _, _, z_dynamics_x, z_dynamics_y = model.encode(input_dynamics)
                xrec = model.decode(z, z_structure, z_dynamics_x, z_dynamics_y)
            else:
                z, z_structure, *_ = model.encode(input_structure)
                _, _, z_dynamics = model.encode(input_dynamics)
                xrec = model.decode(z, z_structure, z_dynamics)
            input_structure = rearrange(input_structure, "b c t h w -> (b t) c h w")
            inputs_structure.append(input_structure)
            input_dynamics = rearrange(input_dynamics, "b c t h w -> (b t) c h w")
            inputs_dynamics.append(input_dynamics)
            xrec = rearrange(xrec, "b c t h w -> (b t) c h w")
            outputs.append(xrec)
        toc = time.time()

    # save the outputs as videos
    inputs_structure = tensor_to_uint8(torch.cat(inputs_structure, dim=0))
    inputs_structure = rearrange(inputs_structure, "t c h w -> t h w c")
    inputs_dynamics = tensor_to_uint8(torch.cat(inputs_dynamics, dim=0))
    inputs_dynamics = rearrange(inputs_dynamics, "t c h w -> t h w c")
    outputs = tensor_to_uint8(torch.cat(outputs, dim=0))
    outputs = rearrange(outputs, "t c h w -> t h w c")
    min_len = min(inputs_structure.shape[0],inputs_dynamics.shape[0], outputs.shape[0])
    final = np.concatenate([inputs_structure[:min_len], inputs_dynamics[:min_len], outputs[:min_len]], axis=2) if args.concate_input else outputs[:min_len]

    output_video_path = os.path.join(args.output_video_dir, f"structure_{Path(args.input_video_path_structure).stem}_dynamics_{Path(args.input_video_path_dynamics).stem}_reconstructed.mp4")
    write_video(output_video_path, final, args.sample_fps)
    print0(f"[bold red]\[vidtwin.scripts.inference_vidtwin_cross_reconstruct][/bold red] Saved the reconstructed video to {output_video_path}")
    print0(f"[bold red]\[vidtwin.scripts.inference_vidtwin_cross_reconstruct][/bold red] Time taken: {toc - tic:.2f}s")

if __name__ == "__main__":
    main()
