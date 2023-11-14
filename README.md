## AANet Version 1.0

AANet: Adaptive Attention Networks for Semantic Segmentation of High-Resolution Remote Sensing Imagery

## Introduction

**AANet** is an open-source semantic segmentation toolbox based on PyTorch, [pytorch lightning](https://www.pytorchlightning.ai/)
and [timm](https://github.com/rwightman/pytorch-image-models),

## Folder Structure

Prepare the following folders to organize this repo:

```none
airs
├── AANet (code)
├── pretrain_weights (pretrained weights of backbones, such as vit, swin, etc)
├── model_weights (save the model weights trained on ISPRS vaihingen, LoveDA, etc)
├── fig_results (save the masks predicted by models)
├── lightning_logs (CSV format training logs)
├── data
│   ├── vaihingen
│   │   ├── train_images (original)
│   │   ├── train_masks (original)
│   │   ├── test_images (original)
│   │   ├── test_masks (original)
│   │   ├── test_masks_eroded (original)
│   │   ├── train (processed)
│   │   ├── test (processed)
│   ├── potsdam (the same with vaihingen)
```

## Install

Open the folder **airs** using **Linux Terminal** and create python environment:

```
conda create -n AANet python=3.8
conda activate AANet
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r AANet/requirements.txt
```

## Data Preprocessing

Download the datasets from the official website and split them yourself.
The potsdam and vaihingen datasets can be downloaded at https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx
**Vaihingen**
Generate the training set.

```
python AANet/tools/vaihingen_patch_split.py --img-dir "data/vaihingen/train_images" --mask-dir "data/vaihingen/train_masks" --output-img-dir "data/vaihingen/train/images_1024" --output-mask-dir "data/vaihingen/train/masks_1024" --mode "train" --split-size 1024 --stride 512 
```

Generate the testing set.

```
python AANet/tools/vaihingen_patch_split.py --img-dir "data/vaihingen/test_images" --mask-dir "data/vaihingen/test_masks_eroded" --output-img-dir "data/vaihingen/test/images_1024" --output-mask-dir "data/vaihingen/test/masks_1024" --mode "val" --split-size 1024 --stride 1024 --eroded
```

Generate the masks_1024_rgb (RGB format ground truth labels) for visualization.

```
python AANet/tools/vaihingen_patch_split.py --img-dir "data/vaihingen/test_images" --mask-dir "data/vaihingen/test_masks" --output-img-dir "data/vaihingen/test/images_1024" --output-mask-dir "data/vaihingen/test/masks_1024_rgb" --mode "val" --split-size 1024 --stride 1024 --gt
```

As for the validation set, you can select some images from the training set to build it.
**Potsdam**

```
python AANet/tools/potsdam_patch_split.py --img-dir "data/potsdam/train_images" --mask-dir "data/potsdam/train_masks" --output-img-dir "data/potsdam/train/images_1024" --output-mask-dir "data/potsdam/train/masks_1024" --mode "train" --split-size 1024 --stride 1024 --rgb-image 
```

```
python AANet/tools/potsdam_patch_split.py --img-dir "data/potsdam/test_images" --mask-dir "data/potsdam/test_masks_eroded" --output-img-dir "data/potsdam/test/images_1024" --output-mask-dir "data/potsdam/test/masks_1024" --mode "val" --split-size 1024 --stride 1024 --eroded --rgb-image
```

```
python AANet/tools/potsdam_patch_split.py --img-dir "data/potsdam/test_images" --mask-dir "data/potsdam/test_masks" --output-img-dir "data/potsdam/test/images_1024" --output-mask-dir "data/potsdam/test/masks_1024_rgb" --mode "val" --split-size 1024 --stride 1024 --gt --rgb-image
```

## Training

"-c" means the path of the config, use different **config** to train different models.

```
vaihingen:
python AANet/train_supervision.py -c AANet/config/vaihingen/aanet.py

potsdam:
python AANet/train_supervision.py -c AANet/config/potsdam/aanet.py
```

## Testing

"-c" denotes the path of the config, Use different **config** to test different models.
"-o" denotes the output path
"--rgb" denotes whether to output masks in RGB format

**Vaihingen**

```
python AANet/vaihingen_test.py -c AANet/config/vaihingen/aanet.py -o fig_results/vaihingen/aanet --rgb
```

**Potsdam**

```
python AANet/potsdam_test.py -c AANet/config/potsdam/aanet.py -o fig_results/potsdam/aanet --rgb
```
