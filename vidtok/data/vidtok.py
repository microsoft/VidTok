import os
import random
import glob
from typing import Union

import decord
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2
from tqdm import trange

from vidtok.modules.util import print0
from .video_read import read_frames_with_decord


class VidTokDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        meta_path: str,
        video_params: dict,
        data_frac: float = 1.0,
        is_strict_loading: bool = False,
        skip_missing_files: bool = True,
        start_index: Union[None, int] = None
    ):
        super().__init__()

        self.data_dir = data_dir
        print0(f"[bold yellow]\[vidtok.data.vidtok][VidTokDataset][/bold yellow] Use data dir: {self.data_dir}")

        self.meta_path = meta_path
        print0(f"[bold yellow]\[vidtok.data.vidtok][VidTokDataset][/bold yellow] Use meta path: {self.meta_path}")

        self.video_params = video_params

        self.data_frac = data_frac
        self.is_strict_loading = is_strict_loading
        self.skip_missing_files = skip_missing_files
        self.start_index = start_index
        self.transforms = self._get_transforms(
            video_params["input_height"],
            video_params["input_width"],
        )

        self.missing_files = []
        self._load_metadata()

    def _get_transforms(self, input_height, input_width, norm_mean=[0.5, 0.5, 0.5], norm_std=[0.5, 0.5, 0.5]):
        normalize = v2.Normalize(mean=norm_mean, std=norm_std)
        return v2.Compose(
            [
                v2.Resize(input_height, antialias=True),
                v2.CenterCrop((input_height, input_width)),
                normalize,
            ]
        )

    def _load_metadata(self):
        metadata = pd.read_csv(
            self.meta_path,
            on_bad_lines="skip",
            encoding="ISO-8859-1",
            engine="python",
            sep=",",)

        if self.data_frac < 1:
            metadata = metadata.sample(frac=self.data_frac)
        self.metadata = metadata
        self.metadata.dropna(inplace=True)

    def _get_video_path(self, sample):
        """reduce the access to the disk
        """
        rel_video_fp = str(sample["videos"])
        abs_video_fp = os.path.join(self.data_dir, rel_video_fp)
        return abs_video_fp, rel_video_fp

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp, _ = self._get_video_path(sample)

        try:
            if os.path.isfile(video_fp):
                imgs, idxs = read_frames_with_decord(
                    video_path=video_fp,
                    sample_num_frames=self.video_params["sample_num_frames"],
                    sample_fps=self.video_params["sample_fps"],
                    start_index=self.start_index
                )
            else:
                # if the video file is missing
                if video_fp not in self.missing_files:
                    self.missing_files.append(video_fp)
                # resample another video or not
                if self.skip_missing_files:
                    print0(f"[bold yellow]\[vidtok.data.vidtok][VidTokDataset][/bold yellow] Warning: missing video file {video_fp}. Resampling another video.")
                    return self.__getitem__(np.random.choice(self.__len__()))
                else:
                    raise ValueError(f"Video file {video_fp} is missing, skip_missing_files={self.skip_missing_files}.")
        except Exception as e:
            # if the video exists, but loading failed
            if self.is_strict_loading:
                raise ValueError(f"Video loading failed for {video_fp}, is_strict_loading={self.is_strict_loading}.") from e
            else:
                print0("[bold yellow]\[vidtok.data.vidtok][VidTokDataset][/bold yellow] Warning: using the pure black image as the frame sample")
                imgs = Image.new("RGB", (self.video_params["input_width"], self.video_params["input_height"]), (0, 0, 0))
                imgs = v2.ToTensor()(imgs).unsqueeze(0)

        if self.transforms is not None:
            # imgs: (T, C, H, W)
            imgs = self.transforms(imgs)

        if imgs.shape[0] < self.video_params["sample_num_frames"]:
            imgs = torch.cat([imgs, imgs[-1].unsqueeze(0).repeat(self.video_params["sample_num_frames"] - imgs.shape[0], 1, 1, 1)], dim=0)

        imgs = imgs.permute(1, 0, 2, 3)  # (C, T, H, W)

        return {
            'jpg': imgs,
            "path": video_fp
        }


class VidTokValDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        video_params: dict,
        meta_path: Union[None, str] = None,
        pre_load_frames: bool = True,
        is_strict_loading: bool = True,
        last_frames_handle: str = "repeat",  # 'repeat', 'drop'
        skip_missing_files: bool = False
    ):
        super().__init__()

        self.data_dir = data_dir
        print0(
            f"[bold yellow]\[vidtok.data.vidtok][VidTokValDataset][/bold yellow] Use data dir: {self.data_dir}"
        )

        self.meta_path = meta_path
        print0(
            f"[bold yellow]\[vidtok.data.vidtok][VidTokValDataset][/bold yellow] Use meta path: {self.meta_path}"
        )

        self.video_params = video_params

        self.is_strict_loading = is_strict_loading
        self.last_frames_handle = last_frames_handle
        self.skip_missing_files = skip_missing_files
        self.transforms = self._get_transforms(
            video_params["input_height"],
            video_params["input_width"],
        )

        self.missing_files = []
        self._load_metadata()
        self._load_every_frame_from_meta()

        if pre_load_frames:
            print0(
                f"[bold yellow]\[vidtok.data.vidtok][VidTokValDataset][/bold yellow] Pre-loading all frames into CPU..."
            )
            self._pre_load_frames()

    def _get_transforms(self, input_height, input_width, norm_mean=[0.5, 0.5, 0.5], norm_std=[0.5, 0.5, 0.5]):
        normalize = v2.Normalize(mean=norm_mean, std=norm_std)
        return v2.Compose(
            [
                v2.Resize(input_height, antialias=True),
                v2.CenterCrop((input_height, input_width)),
                normalize,
            ]
        )

    def _load_metadata(self):
        if self.meta_path is not None:
            metadata = pd.read_csv(
                self.meta_path,
                on_bad_lines="skip",
                encoding="ISO-8859-1",
                engine="python",
                sep=",",
            )
            self.metadata = metadata
            self.metadata.dropna(inplace=True)
        else:
            self.metadata = glob.glob(os.path.join(self.data_dir, '**', '*.mp4'), recursive=True)

    def _load_every_frame_from_meta(self):
        decord.bridge.set_bridge("torch")
        self.frames_batch = []
        for video_idx in range(len(self.metadata)):
            try:
                sample = self.metadata.iloc[video_idx]
                video_fp, _ = self._get_video_path(sample)
            except:
                video_fp = self.metadata[video_idx]
            if os.path.isfile(video_fp):
                video_reader = decord.VideoReader(video_fp, num_threads=0)
                total_frames = len(video_reader)
                fps = video_reader.get_avg_fps()  # float
                interval = round(fps / self.video_params["sample_fps"])
                frame_ids = list(range(0, total_frames, interval))
                for x in range(0, len(frame_ids), self.video_params["sample_num_frames"]):
                    num_frames_ids = frame_ids[x : x + self.video_params["sample_num_frames"]]
                    if len(num_frames_ids) < self.video_params["sample_num_frames"]:
                        if self.last_frames_handle == "repeat":
                            num_frames_ids += [num_frames_ids[-1]] * (
                                self.video_params["sample_num_frames"] - len(num_frames_ids)
                            )
                        elif self.last_frames_handle == "drop":
                            continue
                        else:
                            raise ValueError(f"Invalid last_frames_handle: {self.last_frames_handle}")
                    self.frames_batch.append(
                        {
                            "video_fp": video_fp,
                            "num_frames_ids": num_frames_ids,
                        }
                    )
        print0(
            f"[bold yellow]\[vidtok.data.vidtok][VidTokValDataset][/bold yellow] Loaded all frames index from {len(self.metadata)} videos."
        )

    def _pre_load_frames(self):
        last_video_fp = None
        for idx in trange(len(self.frames_batch), desc="Pre-loading all frames"):
            if self.frames_batch[idx]["video_fp"] != last_video_fp:
                video_reader = decord.VideoReader(self.frames_batch[idx]["video_fp"], num_threads=0)
            last_video_fp = self.frames_batch[idx]["video_fp"]
            self.frames_batch[idx]["frames"] = (
                video_reader.get_batch(self.frames_batch[idx]["num_frames_ids"]).permute(0, 3, 1, 2).float()
                / 255.0
            )

    def _get_video_path(self, sample):
        """reduce the access to the disk"""
        rel_video_fp = str(sample["videos"])
        abs_video_fp = os.path.join(self.data_dir, rel_video_fp)
        return abs_video_fp, rel_video_fp

    def __len__(self):
        return len(self.frames_batch)

    def __getitem__(self, item):
        video_fp = self.frames_batch[item]["video_fp"]

        try:
            if "frames" in self.frames_batch[item]:
                imgs = self.frames_batch[item]["frames"]
            elif os.path.isfile(video_fp):
                video_reader = decord.VideoReader(video_fp, num_threads=0)
                imgs = (
                    video_reader.get_batch(self.frames_batch[item]["num_frames_ids"]).permute(0, 3, 1, 2).float()
                    / 255.0
                )
            else:
                # if the video file is missing
                if video_fp not in self.missing_files:
                    self.missing_files.append(video_fp)
                # resample another video or not
                if self.skip_missing_files:
                    print0(
                        f"[bold yellow]\[vidtok.data.vidtok][VidTokValDataset][/bold yellow] Warning: missing video file {video_fp}. Resampling another video."
                    )
                    return self.__getitem__(np.random.choice(self.__len__()))
                else:
                    raise ValueError(f"Video file {video_fp} is missing, skip_missing_files={self.skip_missing_files}.")
        except Exception as e:
            # if the video exists, but loading failed
            if self.is_strict_loading:
                raise ValueError(
                    f"Video loading failed for {video_fp}, is_strict_loading={self.is_strict_loading}."
                ) from e
            else:
                print0(
                    "[bold yellow]\[vidtok.data.vidtok][VidTokValDataset][/bold yellow] Warning: using the pure black image as the frame sample"
                )
                imgs = Image.new(
                    "RGB", (self.video_params["input_width"], self.video_params["input_height"]), (0, 0, 0)
                )
                imgs = v2.ToTensor()(imgs).unsqueeze(0)

        if self.transforms is not None:
            imgs = self.transforms(imgs)

        if imgs.shape[0] < self.video_params["sample_num_frames"]:
            print0(
                f"[bold yellow]\[vidtok.data.vidtok][VidTokValDataset][/bold yellow] Warning: video {video_fp} has less frames {imgs.shape[0]} than sample_num_frames {self.video_params['sample_num_frames']}."
            )
            imgs = torch.cat(
                [imgs, imgs[-1].unsqueeze(0).repeat(self.video_params["sample_num_frames"] - imgs.shape[0], 1, 1, 1)],
                dim=0,
            )

        imgs = imgs.permute(1, 0, 2, 3)  # (C, T, H, W)

        return {
            "jpg": imgs,
            "path": video_fp,
        }
