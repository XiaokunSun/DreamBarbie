<!-- <p align="center">
  <h2 align="center">Barbie: Text to Barbie-Style 3D Avatars</h2>
  <p align="center">
    <p align="center">
        <a href="https://xiaokunsun.github.io"><strong>Xiaokun Sun<sup>1</sup></strong></a>,
        <a href="https://jessezhang92.github.io"><strong>Zhenyu Zhang<sup>1*</sup></strong></a>,
        <a href="https://tyshiwo.github.io/index.html"><strong>Ying Tai<sup>1</sup></strong></a>,
        <a href="https://ha0tang.github.io"><strong>Hao Tang<sup>2</sup></strong></a>,
        <a href="https://zili-yi.github.io"><strong>Zili Yi<sup>1</sup></strong></a>,
        <a href="https://scholar.google.com.hk/citations?user=6CIDtZQAAAAJ"><strong>Jian Yang<sup>1</sup></strong></a>
        <br>
        <b><sup>1</sup>Nanjing University</b>,
        <b><sup>2</sup>Peking University</b>,
        <b><sup>*</sup>Corresponding Author</b>
    </p>
  </p>
<div align="center">
<a href="https://arxiv.org/pdf/2408.09126">
    <img src="https://img.shields.io/badge/arXiv-2408.09126-b31b1b.svg" alt="ArXiv">
</a> &nbsp;
<a href="https://xiaokunsun.github.io/Barbie.github.io">
    <img src="https://img.shields.io/badge/Project%20Page-Barbie-pink" alt="Project Page">
</a> &nbsp;
<a href="https://drive.google.com/drive/folders/1FXDROWXrnsSQiOZ4vBgA_Yzib3irLNBc?usp=sharing">
    <img src="https://img.shields.io/badge/Gallery-blue" alt="Gallery">
</a>

<img src="assets/teaser.png" alt="Teaser" width="80%">
</div> -->
<div align="center">

# Barbie: Text to Barbie-Style 3D Avatars

<div>
    <a href="https://xiaokunsun.github.io"><strong>Xiaokun Sun</strong></a><sup>1</sup>,
    <a href="https://jessezhang92.github.io"><strong>Zhenyu Zhang</strong></a><sup>1*</sup>,
    <a href="https://tyshiwo.github.io/index.html"><strong>Ying Tai</strong></a><sup>1</sup>,
    <a href="https://ha0tang.github.io"><strong>Hao Tang</strong></a><sup>2</sup>,
    <a href="https://zili-yi.github.io"><strong>Zili Yi</strong></a><sup>1</sup>,
    <a href="https://scholar.google.com.hk/citations?user=6CIDtZQAAAAJ"><strong>Jian Yang</strong></a><sup>1</sup>
</div>

<div>
    <sup>1</sup><strong>Nanjing University</strong> &nbsp;&nbsp;
    <sup>2</sup><strong>Peking University</strong>
</div>

<div>
    <sup>*</sup><strong>Corresponding Author</strong>
</div>

<br>

[![ArXiv](https://img.shields.io/badge/ArXiv-2408.09126-b31b1b.svg)](https://arxiv.org/pdf/2408.09126)
[![Project Page](https://img.shields.io/badge/Project%20Page-Barbie-ff69b4.svg)](https://xiaokunsun.github.io/Barbie.github.io)
[![Gallery](https://img.shields.io/badge/Gallery-View-blue.svg)](https://drive.google.com/drive/folders/1FXDROWXrnsSQiOZ4vBgA_Yzib3irLNBc?usp=sharing)

<br>
<img src="assets/teaser.png" alt="Teaser" width="85%">
</div>

## 🔨 Installation
Tested on **Ubuntu 20.04**, **Python 3.8**, **NVIDIA A6000**, **CUDA 11.7**, and **PyTorch 2.0.0**. Follow the steps below to set up the environment.

1. Clone the repo:
```bash
git clone https://github.com/XiaokunSun/Barbie.git
cd Barbie
```
2. Create a conda environment:
```bash
conda create -n barbie python=3.8 -y
conda activate barbie
```

3. Install dependencies:
```bash
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/ashawkey/envlight.git
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install git+https://github.com/KAIR-BAIR/nerfacc.git@v0.5.2
pip install git+https://github.com/ashawkey/cubvh --no-build-isolation
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.5/download/linux-64/pytorch3d-0.7.5-py38_cu117_pyt200.tar.bz2 # Note: Please ensure the pytorch3d version matches your CUDA and Torch versions
pip install git+https://github.com/bellockk/alphashape.git
```

4. Download models:
```bash
mkdir ./pretrained_models
bash ./scripts/download_humannorm_models.sh
python ./scripts/download_richdreamer_models.py
cd ./pretrained_models && ln -s ~/.cache/huggingface ./
cd ../
```
5. Download other models (eg., SMPLX, Tets) from [GoogleDrive](https://drive.google.com/drive/folders/1c8ouintJ1xqnlx2logEHfJSn4HD3EHAu?usp=drive_link).
Make sure you have the following models:
```bash
Barbie
|-- load
    |-- barbie
        |-- data_dict.json
        |-- overall_data_dict.json
    |-- smplx_models
        |-- smplx
            |-- smplx_cloth_mask.pkl
            |-- smplx_face_ears_noeyeballs_idx.npy
            |-- SMPLX_NEUTRAL.npz
            |-- smplx_watertight.pkl
    |-- tets
        |-- 256_tets.npz
    |-- prompt_library.json
|-- pretrained_models
    |-- controlnet-normal-sd1.5
    |-- depth-adapted-sd1.5
    |-- normal-adapted-sd1.5
    |-- normal-aligned-sd1.5
    |-- Damo_XR_Lab
        |-- Normal-Depth-Diffusion-Model
            |-- nd_mv_ema.ckpt
            |-- albedo_mv_ema.ckpt
    |-- huggingface
        |-- hub
            |-- models--runwayml--stable-diffusion-v1-5
            |-- models--openai--clip-vit-large-patch14
            |-- models--stabilityai--stable-diffusion-2-1-base
            |-- models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K
```

## 🕺 Inference
```bash
# Generate naked human
python ./scripts/generate_naked_human.py --dict_path ./load/barbie/data_dict.json --naked_human_exp_root_dir ./outputs/naked_human --naked_human_idx 0:1:1 --gpu_idx 0
# Generate clothed human
python ./scripts/generate_clothed_human.py --dict_path ./load/barbie/data_dict.json --naked_human_exp_root_dir ./outputs/naked_human --clothed_human_exp_root_dir ./outputs/clothed_human --naked_human_idx 0:1:1 --cloth_idx 0:1:1 --gpu_idx 0
# Generate human wearing overall
python ./scripts/generate_naked_human.py --dict_path ./load/barbie/overall_data_dict.json --naked_human_exp_root_dir ./outputs/naked_human --naked_human_idx 0:1:1 --gpu_idx 0
python ./scripts/generate_clothed_overall_human.py --dict_path ./load/barbie/overall_data_dict.json --naked_human_exp_root_dir ./outputs/naked_human --clothed_human_exp_root_dir ./outputs/clothed_overall_human --naked_human_idx 0:1:1 --cloth_idx 0:1:1 --gpu_idx 0
```

## 🪄 Application
```bash
# Apparel Combination
python ./scripts/apparel_combination.py
# Apparel Editing
python ./scripts/apparel_editing.py
# Animation
python ./scripts/animation.py
```
If you want to customize clothing templates, please see ```./scripts/smplx_cloth_mask.py```

## ⭐ Acknowledgements
This repository is based on many amazing research works and open-source projects: [ThreeStudio](https://github.com/threestudio-project/threestudio), [HumanNorm](https://github.com/xhuangcv/humannorm), [RichDreamer](https://github.com/modelscope/richdreamer), [G-Shell](https://github.com/lzzcd001/GShell), etc. Thanks all the authors for their selfless contributions to the community!

## 📚 Citation
If you find this repository helpful for your work, please consider citing it as follows:
```bibtex
@article{sun2024barbie,
  title={Barbie: Text to Barbie-Style 3D Avatars},
  author={Sun, Xiaokun and Zhang, Zhenyu and Tai, Ying and Tang, Hao and Yi, Zili and Yang, Jian},
  journal={arXiv preprint arXiv:2408.09126},
  year={2024}
}
```
