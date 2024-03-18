<div align="center">

<h2> <img src="assets/favicon.ico" style="vertical-align: middle;" width="30" height="30"> EchoReel: <span style="font-size:12px">Enhancing Action Generation of Existing Video Diffusion Models </span> </h2>

<a href='https://arxiv.org/abs/xx.xx'><img src='https://img.shields.io/badge/ArXiv-xx.xx-red'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='https://liujianzhi.github.io/EchoReel/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>


<div>
    <a href='https://github.com/liujianzhi' target='_blank'> Jianzhi Liu</a>&emsp;
    <a href='https://scholar.google.com/citations?user=J0qJuYAAAAAJ&hl=zh-CN' target='_blank'> Junchen Zhu</a>&emsp;
    <a href='https://scholar.google.com.au/citations?user=zsm2dpYAAAAJ&hl=en' target='_blank'> Lianli Gao</a>&emsp;
    <a href='https://scholar.google.com.au/citations?user=F5Zy9V4AAAAJ&hl=en' target='_blank'> Jingkuan Song</a>&emsp;
</div>
<div>
    University of Electronic Science and Technology of China
</div>
<br>


<b>An innovative method designed to augment the capabilities of existing video diffusion models that can:</b>  
1️⃣ utilize multiple reference videos to achieve a broader spectrum of action imitation and generate novel actions without fine-tuning;  
2️⃣ distill effective and related visual motion features instead of replicating the referred content.


<div align="left">



<div style="text-align: center;">
  <i>"Imitation is the sincerest form of flattery that mediocrity can pay to greatness."</i> — Oscar Wilde 
</div>


## ✌️ Results
<table class="center">
  <tr>
  <td style="text-align:center;" width="20%">input text</td>
  <td style="text-align:center;" width="40%">Original VideoCrafter2</td>
  <td style="text-align:center;" width="40%">+ EchoReel</td>
  <tr>
  <td style="text-align:center;">"A man is studying in the library"</td>
  <td><img src=assets/1.gif></td>
  <td><img src=assets/2.gif></td>
  <tr>
  <td style="text-align:center;">"A man is skiing"</td>
  <td><img src=assets/3.gif></td>
  <td><img src=assets/4.gif></td>
  <tr>
  <td style="text-align:center;">"A man is running"</td>
  <td><img src=assets/5.gif></td>
  <td><img src=assets/6.gif></td>
  <tr>
  <td style="text-align:center;">"Couple walking on the beach"</td>
  <td><img src=assets/7.gif></td>
  <td><img src=assets/8.gif></td>
  <tr>
  <td style="text-align:center;">"A man is carving a stone statue"</td>
  <td><img src=assets/9.gif></td>
  <td><img src=assets/10.gif></td>
</tr>
</table > 

## ⏳ TODO
- [x] Release code of LVDM text-to-video with EchoReel
- [x] Release training code
- [ ] Release pretrained weight
- [ ] Release image-to-video VideoCrafter code with EchoReel

## ⚙️ Setup

Please prepare .json data in the following format:

```
[
	{
		"input_text": ...,
		"gt_video_path": ...,
		"reference_text": ...,
		"reference_video_path": ...
	},
    ...
]
```

Install Environment via Anaconda
```
conda create -n EchoReel python=3.10.13
conda activate EchoReel
pip install -r requirements.txt
```

## 💫 For Train

```bash
% use original LVDM pretrain weight to initialize model
wget -O models/t2v/model.ckpt https://huggingface.co/Yingqing/LVDM/resolve/main/lvdm_short/t2v.ckpt
bash train_EchoReel.sh
```

## 💫 For Sample

```bash
bash sample_EchoReel.sh
```

## 🔮 Pipeline
<p align="center">
    <img src=assets/overview.jpg />
</p>

## 😉 Citation

```
@article{Liu2024EchoReel,
      title={EchoReel: Enhancing Action Generation of Existing Video Diffusion Models}, 
      author={Jianzhi Liu, Junchen Zhu, Lianli Gao, Jingkuan Song},
      year={2024},
      eprint={--},
      archivePrefix={arXiv},
}
```

## 🤗 Acknowledgements

We built our code partially based on [latent video diffusion models](https://github.com/CompVis/latent-diffusion). Thanks for their wonderful work!