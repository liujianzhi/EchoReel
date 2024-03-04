<div align="center">

<h2> <img src="assets/favicon.ico" style="vertical-align: middle;" width="30" height="30"> EchoReel: <span style="font-size:12px">Enhancing Action Generation of Existing Video Diffusion Models </span> </h2>

<div>
    Jianzhi Liu &emsp; Junchen Zhu &emsp; Lianli Gao &emsp; Jingkuan Song <sup>*</sup>
</div>
<div>
    University of Electronic Science and Technology of China
</div>
<br>
<div align="left">

<p align="center">
    <img src=assets/banner.jpg />
</p>

## ✌️ Results
<table class="center">
  <!-- <td style="text-align:center;" width="50">Input Text</td> -->
  <tr>
  <td style="text-align:center;" width="20%">input text</td>
  <td style="text-align:center;" width="40%">Original LVDM</td>
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
		"inject_text": ...,
		"inject_video_path": ...
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
@article{LiuZGS2024EchoReel,
      title={EchoReel: Enhancing Action Generation of Existing Video Diffusion Models}, 
      author={Jianzhi Liu and Junchen Zhu and Lianli Gao and Jingkuan Song},
      year={2024}
}
```

## 🤗 Acknowledgements

We built our code partially based on [latent video diffusion models](https://github.com/CompVis/latent-diffusion). Thanks for their wonderful work!