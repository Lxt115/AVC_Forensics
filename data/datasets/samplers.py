"""Samplers for dataloaders"""

import torch
from torch.utils.data import Sampler, RandomSampler


class ConsecutiveClipSampler(RandomSampler):
    """Sampler for consecutive non-overlapping clips in a video"""
    # 视频中连续不重叠剪辑的采样器
    def __init__(self, clips_per_video):
        """ "
        Parameters
        ----------
        clips_per_video : list
            Number of clips in each video
        """
        self.clips_per_video = clips_per_video

    def __iter__(self):
        """Sampler for consecutive non-overlapping clips in a video"""
        idxs = []
        s = 0
        for num_clips in self.clips_per_video:
            sampled = torch.arange(num_clips)[:num_clips] + s
            s += num_clips
            idxs.append(sampled)
        idxs = torch.cat(idxs).tolist()
        return iter(idxs)

    def __len__(self):
        return sum(num_clips for num_clips in self.clips_per_video)
