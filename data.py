import torch
import json
import os
import cv2
import numpy as np
import pandas as pd
import random
from decord import VideoReader, cpu
from torch.utils.data import Dataset

def load_video_batch(filepath, frame_stride, video_size=(256,256), video_frames=16):
        '''
        Notice about some special cases:
        1. video_frames=-1 means to take all the frames (with fs=1)
        2. when the total video frames is less than required, padding strategy will be used (repreated last frame)
        '''
        fps_list = []
        batch_tensor = []
        assert frame_stride > 0, "valid frame stride should be a positive interge!"
        padding_num = 0
        vidreader = VideoReader(filepath, ctx=cpu(0), width=video_size[1], height=video_size[0])
        fps = vidreader.get_avg_fps()
        total_frames = len(vidreader)
        max_valid_frames = (total_frames-1) // frame_stride + 1
        if video_frames < 0:
            ## all frames are collected: fs=1 is a must
            required_frames = total_frames
            frame_stride = 1
        else:
            required_frames = video_frames
        query_frames = min(required_frames, max_valid_frames)
        frame_indices = [frame_stride*i for i in range(query_frames)]

        ## [t,h,w,c] -> [c,t,h,w]
        frames = vidreader.get_batch(frame_indices)
        frame_tensor = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()
        frame_tensor = (frame_tensor / 255. - 0.5) * 2
        if max_valid_frames < required_frames:
            padding_num = required_frames - max_valid_frames
            frame_tensor = torch.cat([frame_tensor, *([frame_tensor[:,-1:,:,:]]*padding_num)], dim=1)
            print(f'{os.path.split(filepath)[1]} is not long enough: {padding_num} frames padded.')
        batch_tensor.append(frame_tensor)
        sample_fps = int(fps/frame_stride)
        fps_list.append(sample_fps)
        
        return torch.stack(batch_tensor, dim=0).squeeze(0)

class create_dataset(Dataset):
    def __init__(self, dataset, lim):
        self.datasetroot = os.path.dirname(dataset)
        self.dataset = json.loads(open(os.path.join(dataset)).read())
        if lim:
            self.dataset = self.dataset[::len(self.dataset)//20]
        self.video_length = 16
        self.frame_stride = 4
        self.resolution = [512, 512]
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
        while True:
            try:
                data = self.dataset[index]
            except:
                index = 0
                data = self.dataset[index]
            input_text = data['input_text']
            gt_video_path = os.path.join(self.datasetroot, data['gt_video_path'])
            inject_text = data['inject_text']
            inject_video_path = os.path.join(self.datasetroot, data['inject_video_path'])
            try:
                video_reader = VideoReader(gt_video_path, ctx=cpu(0), width=self.resolution[1], height=self.resolution[0])
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
                print(f"Load video failed! path = {gt_video_path}")
        

        #     gt_video_cv2 = cv2.VideoCapture(gt_video_path)
        #     gt_video_frame_count = gt_video_cv2.get(7) # frame count
        #     inject_video_cv2 = cv2.VideoCapture(inject_video_path)
        #     inject_video_frame_count = inject_video_cv2.get(7) # frame count
        #     if index < len(self.dataset):
        #         index += 1
        #     else:
        #         index = 0

        # gt_video_start_frame = random.randint(0, gt_video_frame_count - 17)
        # inject_video_start_frame = random.randint(0, inject_video_frame_count - 17)
        # gt_video = None
        # inject_video = None

        
        # for _ in range(gt_video_start_frame):
        #     ret, frame = gt_video_cv2.read()
        # for _ in range(16):
        #     ret, frame = gt_video_cv2.read()
        #     if not ret:
        #         frame = old_frame
        #     frame = cv2.resize(frame, (256, 256))
        #     if gt_video is None:
        #         gt_video = np.expand_dims(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), axis=0)
        #     else:
        #         gt_video = np.append(gt_video, np.expand_dims(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), axis=0), axis=0)
        #     old_frame = frame
        # for _ in range(inject_video_start_frame):
        #     ret, frame = inject_video_cv2.read()
        # for _ in range(16):
        #     ret, frame = inject_video_cv2.read()
        #     if not ret:
        #         frame = old_frame
        #     frame = cv2.resize(frame, (256, 256))
        #     if inject_video is None:
        #         inject_video = np.expand_dims(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), axis=0)
        #     else:
        #         inject_video = np.append(inject_video, np.expand_dims(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), axis=0), axis=0)
        #     old_frame = frame
        
        # gt_video = torch.tensor(gt_video).permute(3, 0, 1, 2).float()
        # inject_video = torch.tensor(inject_video).permute(3, 0, 1, 2).float()
        # gt_video = (gt_video / 255 - 0.5) * 2
        # inject_video = (inject_video / 255 - 0.5) * 2
        # print(input_text, data['gt_video_path'], inject_text, data['inject_video_path'])
        gt_video = load_video_batch(gt_video_path, 4)
        inject_video = load_video_batch(inject_video_path, 4)
        return {'input_text': input_text,
                'gt_video': gt_video,
                'caption': gt_video[0],
                'inject_text': inject_text,
                'inject_video': inject_video,
            }

    def __len__(self):
        return len(self.dataset)