import os
import random
import decord
import numpy as np
import torch

from vidtok.modules.util import print0

decord.bridge.set_bridge("torch")


def sample_frames_with_fps(
    total_frames,
    video_fps,
    sample_num_frames,
    sample_fps,
    start_index=None
):
    """sample frames proportional to the length of the frames in one second
    e.g., 1s video has 30 frames, when 'fps'=3, we sample frames with spacing of 30/3=10
    return the frame indices

    Parameters
    ----------
    total_frames : length of the video
    video_fps : original fps of the video
    sample_num_frames : number of frames to sample
    sample_fps : the fps to sample frames
    start_index : the starting frame index. If it is not None, it will be used as the starting frame index  

    Returns
    -------
    frame indices
    """
    sample_num_frames = min(sample_num_frames, total_frames)
    interval = round(video_fps / sample_fps)
    frames_range = (sample_num_frames - 1) * interval + 1

    if start_index is not None:
        start = start_index
    elif total_frames - frames_range - 1 < 0:
        start = 0
    else:
        start = random.randint(0, total_frames - frames_range - 1)

    frame_idxs = np.linspace(
        start=start, stop=min(total_frames - 1, start + frames_range), num=sample_num_frames
    ).astype(int)

    return frame_idxs


def read_frames_with_decord(
    video_path,
    sample_num_frames,
    sample_fps,
    start_index=None
) -> tuple[torch.Tensor, list[int]]:
    """read frames from video path using decord

    Parameters
    ----------
    video_path : path to video
    sample_num_frames : number of frames to sample
    sample_fps : the fps to sample frames
    start_index : the starting frame index. If it is not None, it will be used as the starting frame index  

    Returns
    -------
    frames (tensor 0~1), frame indices
    """
    video_reader = decord.VideoReader(video_path, num_threads=0)
    total_frames = len(video_reader)
    video_fps = video_reader.get_avg_fps()  # note that the fps here is float.
    frame_idxs = sample_frames_with_fps(
        total_frames=total_frames,
        video_fps=video_fps,
        sample_num_frames=sample_num_frames,
        sample_fps=sample_fps,
        start_index=start_index
    )
    frames = video_reader.get_batch(frame_idxs)
    frames = frames.float() / 255
    frames = frames.permute(0, 3, 1, 2)
    if (frames.shape[0] != sample_num_frames) or (len(frame_idxs) != sample_num_frames):
        print0(f"[bold yellow]\[vidtok.data.video_read][read_frames_with_decord][/bold yellow] Warning: need {sample_num_frames} frames, "
               f"but got {frames.shape[0]} frames, {len(frame_idxs)} frame indices, video_path={video_path}.")
    return frames, frame_idxs
