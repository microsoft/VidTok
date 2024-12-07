import os
import sys
sys.path.append(os.getcwd())

import argparse
import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
from contextlib import nullcontext
from pathlib import Path

import torch
from einops import rearrange
from lightning.pytorch import seed_everything
from torch import autocast
from torchvision.io import write_video
from tqdm import tqdm

from scripts.inference_evaluate import print0, load_model_from_config, transforms, decord


class SingleVideoDataset(torch.utils.data.Dataset):
    def __init__(self, video_path, input_height=128, input_width=128, num_frames_per_batch=16, sample_fps=8):
        decord.bridge.set_bridge("torch")
        self.video_path = video_path
        normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        self.transform = transforms.Compose(
            [
                transforms.Resize(input_height, antialias=True),
                transforms.CenterCrop((input_height, input_width)),
                normalize,
            ]
        )

        self.video_reader = decord.VideoReader(video_path, num_threads=0)
        total_frames = len(self.video_reader)
        fps = self.video_reader.get_avg_fps()  # float

        interval = round(fps / sample_fps)
        frame_ids = list(range(0, total_frames, interval))
        self.frame_ids_batch = []
        for x in range(0, len(frame_ids), num_frames_per_batch):
            if len(frame_ids[x : x + num_frames_per_batch]) == num_frames_per_batch:
                self.frame_ids_batch.append(frame_ids[x : x + num_frames_per_batch])

    def __len__(self):
        return len(self.frame_ids_batch)

    def __getitem__(self, idx):
        frame_ids = self.frame_ids_batch[idx]
        frames = self.video_reader.get_batch(frame_ids).permute(0, 3, 1, 2).float() / 255.0
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
        "--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="full"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/kl_causal_488_4chn.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/kl_causal_488_4chn.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--output_video_dir",
        type=str,
        default="tmp",
        help="path to save the outputs",
    )
    parser.add_argument(
        "--input_video_path",
        type=str,
        default="assets/example.mp4",
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
    parser.add_argument(
        "--concate_input",
        type=str2bool,
        const=True,
        default=True,
        nargs="?",
        help="",
    )

    args = parser.parse_args()
    seed_everything(args.seed)

    print0(f"[bold red]\[scripts.inference_reconstruct][/bold red] Evaluating model {args.ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    precision_scope = autocast if args.precision == "autocast" else nullcontext

    os.makedirs(args.output_video_dir, exist_ok=True)

    model = load_model_from_config(args.config, args.ckpt)
    model.to(device).eval()

    dataset = SingleVideoDataset(
        args.input_video_path, args.input_height, args.input_width, args.num_frames_per_batch, args.sample_fps
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    inputs = []
    outputs = []
    with torch.no_grad(), precision_scope("cuda"):
        tic = time.time()
        for i, input in tqdm(enumerate(dataloader)):
            input = input.to(device)
            _, xrec, _ = model(input)
            input = rearrange(input, "b c t h w -> (b t) c h w")
            inputs.append(input)
            xrec = rearrange(xrec, "b c t h w -> (b t) c h w")
            outputs.append(xrec)

        toc = time.time()

    # save the outputs as videos
    inputs = tensor_to_uint8(torch.cat(inputs, dim=0))
    inputs = rearrange(inputs, "t c h w -> t h w c")
    outputs = tensor_to_uint8(torch.cat(outputs, dim=0))
    outputs = rearrange(outputs, "t c h w -> t h w c")
    min_len = min(inputs.shape[0], outputs.shape[0])
    final = np.concatenate([inputs[:min_len], outputs[:min_len]], axis=2) if args.concate_input else outputs[:min_len]

    output_video_path = os.path.join(args.output_video_dir, f"{Path(args.input_video_path).stem}_reconstructed.mp4")
    write_video(output_video_path, final, args.sample_fps)

    print0(f"[bold red]Results saved in: {output_video_path}[/bold red]")
    print0(f"[bold red]\[scripts.inference_reconstruct][/bold red] Time taken: {toc - tic:.2f}s")


if __name__ == "__main__":
    main()
