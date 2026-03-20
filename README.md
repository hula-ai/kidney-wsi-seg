# Kidney WSI segmentation of diagnostic tissue compartments

Minimal steps to train, run inference for renal compartment segmentation on whole slide images.

## Contents

- [Overview](#overview)
- [Setup](#setup)
- [Run](#run)
  - [Training](#training)
  - [Inference](#inference)
  - [Evaluation](#evaluation)
- [Resources](#resources)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Overview

- Three-stage pipeline: tissue segmentation, instance segmentation with MMDetection, and inference-time post-processing.
- Key folders: `train` (MMDetection configs and scripts), `inference` (checkpointed models and runners), `eval` (metrics and plots).

## Setup

1) Create and activate the conda environment

```sh
conda create --name milxseg python=3.8.15 -y
conda activate milxseg
```

2) Install PyTorch

```sh
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
```

3) Install Python requirements and MMDetection stack

```sh
pip install -r requirements.txt
pip install -U openmim
mim install mmcv-full==1.5.0   # or: mim install mmcv-full==1.7.1
mim install mmdet==2.20.0
mim install mmengine           # quick install verification
```

> Tips
> - If `mmcv-full` install fails, try another compatible `cudatoolkit` version.
> - Error: `AssertionError: MMCV==1.7.1 is used but incompatible. Please install mmcv>=1.3.17, <=1.5.0.`
>   - Edit `./train/mmdet/__init__.py` and bump `mmcv_maximum_version`, or patch your site-packages copy of `mmdet/__init__.py`.
> - Error: `ValueError: no valid qupath installation found`
>   - Install QuPath and export `PAQUO_QUPATH_DIR` as suggested by `paquo get_qupath`.

## Run

### Training

If you prefer to skip training, jump to [Inference](#inference).

1) Move into the training workspace

```sh
cd ./train
```

2) Prepare COCO JSONs

```sh
./scripts/create_tma_json_tissue_detection_and_subtract_smaller_compartments.sh
```

3) Review config guidance at [docs/train_configs.md](./docs/train_configs.md).

4) Launch distributed training (example)

```sh
./tools/dist_train.sh ./configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_HULA_compartment.py 4 --work-dir /data/hqvo3/mmdet/run2
```

Where `4` sets the GPU count and `--work-dir` sets the checkpoint output directory.

Troubleshooting: see [docs/qa.md](./docs/qa.md). Reference MMDetection docs for deeper details: https://mmdetection.readthedocs.io/en/latest/.

### Inference

1) Download checkpoints from this [🤗 link](https://huggingface.co/HuyVoHF304/kidney-wsi-seg/tree/main) and place them in `./inference/checkpoints`.

2) Use the sample command in `inference/scripts/inf.sh` as a template for your data.

### Evaluation

- Evaluation utilities live in `eval`; more details at [docs/eval.md](./docs/eval.md).

## Resources

- Training config how-to: [docs/train_configs.md](./docs/train_configs.md)
- Troubleshooting Q&A: [docs/qa.md](./docs/qa.md)
- Evaluation guide: [docs/eval.md](./docs/eval.md)

## Citation

If you use this work, please cite:

```
@article{vo2023segmentation,
  title={Segmentation of diagnostic tissue compartments on whole slide images with renal thrombotic microangiopathies (TMAs)},
  author={Vo, Huy Q and Cicalese, Pietro A and Seshan, Surya and Rizvi, Syed A and Vathul, Aneesh and Bueno, Gloria and Dorado, Anibal Pedraza and Grabe, Niels and Stolle, Katharina and Pesce, Francesco and others},
  journal={arXiv preprint arXiv:2311.14971},
  year={2023}
}
```

## Acknowledgement

[MMDetection Toolbox](https://github.com/open-mmlab/mmdetection)
