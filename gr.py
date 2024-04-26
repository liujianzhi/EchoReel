import os
from omegaconf import OmegaConf
import torch
from collections import OrderedDict
import gradio as gr
import random
import numpy as np
import wget
import pandas as pd
import importlib
import torchvision
from decord import VideoReader, cpu
import pytorch_lightning
import time
path = "dataset/results_10M_val.csv"
dataset_root = './cache'
if not os.path.exists(dataset_root):
    os.makedirs(dataset_root)

datas = np.array(pd.read_csv(path))

text = datas[:, 4] 


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

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")

    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def check(path):
    import cv2
    vid = cv2.VideoCapture(path)
    ret, frame = vid.read()
    frame = vid.get(7)
    if frame < 16:
        return False
    else:
        return True

def load_model_checkpoint(model, ckpt, full_strict):
    def load_checkpoint(model, ckpt, full_strict):
        state_dict = torch.load(ckpt, map_location="cpu")
        try:
            new_pl_sd = OrderedDict()
            for key in state_dict['module'].keys():
                new_pl_sd[key[16:]]=state_dict['module'][key]
            model.load_state_dict(new_pl_sd, strict=full_strict)
        except:
            if "state_dict" in list(state_dict.keys()):
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict, strict=full_strict)
        return model
    load_checkpoint(model, ckpt, full_strict=full_strict)
    print('>>> model checkpoint loaded.')
    return model


config = OmegaConf.load("configs/gr.yaml")
model_config = config.pop("model", OmegaConf.create())
model = instantiate_from_config(model_config)



d = torch.load('checkpoint/checkpoint.pt', map_location='cpu')
d_con = {}
for t in d['module']:
    d_con[t[t.find('.') + 1:]] = d['module'][t]
    
model.load_state_dict(d_con, strict=True)
model = model.cuda().half()

def processsd(prompt, word):
    global preprocessor
    print("processing")

    indexs = []
    for i, t in enumerate(text):
        if word in t:
            indexs.append(i)
    if len(indexs) == 0:
        batch = {'input_text': [prompt],
                'gt_video': torch.zeros([1, 4, 16, 256, 256]).cuda().half(),
                'inject_text': [prompt],
                'inject_video': torch.zeros([1, 4, 16, 256, 256]).cuda().half(),
            }
        video = model.video_create(batch)
        vid_tensor = video['samples']
        video = vid_tensor.detach().cpu()
        video = torch.clamp(video.float(), -1.0, 1.0)
        video = video.permute(2, 0, 1, 3, 4)  # t,n,c,h,w

        frame_grids = [
            torchvision.utils.make_grid(framesheet, nrow=1)
            for framesheet in video
        ]  # [3, 1*h, n*w]
        grid = torch.stack(frame_grids, dim=0)  # stack in temporal dim [t, 3, n*h, w]
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        torchvision.io.write_video(
            'output_video.mp4',
            grid,
            fps=4,
            video_codec="h264",
            options={"crf": "10"},
        )
        return ['output_video.mp4', f"No video searched, using off-the-shell LVDM"]
    index = random.choice(range(len(indexs)))
    videoid, contentUrl, duration, page_dir, name = datas[index]
    if not check(os.path.join(dataset_root, str(videoid) + '.mp4')):
        try:
            wget.download(contentUrl, os.path.join(dataset_root, str(videoid) + '.mp4'))
        except:
            return [None, f"Download error"]
    with torch.no_grad():
        inject_text = name
        inject_video_path = os.path.join(dataset_root, str(videoid) + '.mp4')
        
        inject_video = load_video_batch(inject_video_path, 4)

        batch = {'input_text': [prompt],
                 'gt_video': inject_video.cuda().half().unsqueeze(0),
                'inject_text': [inject_text],
                'inject_video': inject_video.cuda().half().unsqueeze(0),
            }
        video = model.video_create(batch)
        vid_tensor = video['samples']
        video = vid_tensor.detach().cpu()
        video = torch.clamp(video.float(), -1.0, 1.0)
        video = video.permute(2, 0, 1, 3, 4)  # t,n,c,h,w

        frame_grids = [
            torchvision.utils.make_grid(framesheet, nrow=1)
            for framesheet in video
        ]  # [3, 1*h, n*w]
        grid = torch.stack(frame_grids, dim=0)  # stack in temporal dim [t, 3, n*h, w]
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        torchvision.io.write_video(
            'output_video.mp4',
            grid,
            fps=4,
            video_codec="h264",
            options={"crf": "10"},
        )
    
    return ['output_video.mp4', f"Find {len(indexs)} videos in dataset, and random select one."]


pytorch_lightning.seed_everything(time.time())

block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## LVDM with EchoReel generation")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                prompt = gr.Textbox(label="Prompt")
            with gr.Row():
                word = gr.Textbox(label="Auto search word, please input only one word as the action of prompt")
            with gr.Row():
                run_button = gr.Button(label="Search and Generate")
        with gr.Column():
            result_video = gr.Video(label='Output Video', elem_id="video")
            result_text = gr.Textbox()
    ips = [prompt, word]
    run_button.click(fn=processsd, inputs=ips, outputs=[result_video, result_text])

block.launch(server_name='0.0.0.0', share=True)