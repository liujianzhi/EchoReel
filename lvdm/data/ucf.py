import torch
import json
import os
import cv2
import numpy as np
import pandas as pd
import random
from decord import VideoReader, cpu
from torch.utils.data import Dataset

class ucf(Dataset):
    def __init__(self, dataset, for_train, mode):
        self.datasetroot = os.path.dirname(dataset)
        if for_train:
            self.dataset = json.loads(open(os.path.join(dataset)).read())
        if not for_train:
            self.dataset = json.loads(open(os.path.join(self.datasetroot, 'webvid.json')).read())
            if mode == 'train':
                self.dataset = self.dataset[::len(self.dataset)//20]
        self.video_length = 16
        self.frame_stride = 4
        self.resolution = [256, 256]
        pass

    def _make_dataset(self):
        return
    
    def video(self, video_path, index):
        while True:
            try:
                video_reader = VideoReader(video_path, ctx=cpu(0), width=self.resolution[1], height=self.resolution[0])
                if len(video_reader) < self.video_length:
                    index += 1
                    continue
                else:
                    break
            except:
                if index < len(self.dataset):
                    index += 1
                else:
                    index = 0
                print(f"Load video failed! path = {video_path}")
    
        all_frames = list(range(0, len(video_reader), self.frame_stride))
        if len(all_frames) < self.video_length:
            all_frames = list(range(0, len(video_reader), 1))

        # select random clip
        rand_idx = random.randint(0, len(all_frames) - self.video_length)
        frame_indices = list(range(rand_idx, rand_idx+self.video_length))
        frames = video_reader.get_batch(frame_indices)
        assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'

        frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
        assert(frames.shape[2] == self.resolution[0] and frames.shape[3] == self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        frames = (frames / 255 - 0.5) * 2
        return frames


    def __getitem__(self, index):
        data = self.dataset[index]

        input_text = data['input_text']
        gt_video_path = os.path.join(self.datasetroot, data['gt_video_path'])
        inject_text = data['inject_text']
        inject_video_path = os.path.join(self.datasetroot, data['inject_video_path'])
        gt_video = self.video(gt_video_path, index)
        inject_video = self.video(inject_video_path, index)
        return {'input_text': input_text,
                'gt_video': gt_video,
                'inject_text': inject_text,
                'inject_video': inject_video,
            }

    def __len__(self):
        return len(self.dataset)