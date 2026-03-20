# Kidney WSI segmentation of diagnostic tissue compartments


# Overview

## 3-stage model

- Tissue segmentation: tree structure
- Instance segmentation: mmdet
- Post-processing step: only in inference stage

## 3 folders

- `train`
- `inference`
- Produce results

# Setup

Create conda environment:

```sh
conda create --name milxseg python=3.8.15 -y
conda activate milxseg
```

Install pytorch:

```sh
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
```

Install required packages:

```sh
pip install -r requirements.txt
pip install -U openmim
mim install mmcv-full==1.5.0 (or mim install mmcv-full==1.7.1)
mim install mmdet==2.20.0
mim install mmengine (for installation verification)
```

_Note:_

- When installing `mim install mmcv-full==1.5.0`, you may have an issue with mismatched cuda version. If so, please use or install another `cudatoolkit` version.

- When there is a problem like this "AssertionError: MMCV==1.7.1 is used but incompatible. Please install mmcv>=1.3.17, <=1.5.0.
":
  - Go to `./train/mmdet/__init__.py`: change the variable `mmcv_maximum_version` to a newer version
  - or Go to `/data/<user>/miniconda3/envs/milxseg/lib/python3.8/site-packages/mmdet/__init__.py`: change the variable `mmcv_maximum_version` to a newer version

- "ValueError: no valid qupath installation found": 
  - when getting this error, it means the package `paquo` cannot find the QuPath folder on your machine (mostly because QuPath has not been installed).
  - To solve this problem, read the `Getting QuPath` section in the [Paquo document](https://pypi.org/project/paquo/).
  - Please make sure to do the export according to the output of the command `paquo get_qupath`, for example: `export PAQUO_QUPATH_DIR=/home/<user>/QuPath/QuPath-0.4.1`.

# Run

## Training

If you prefer to skip training, jump to the [Inference](#inference) section below.

Training segmentation model with the MMDetection Toolbox.

```sh
cd ./train
```

Create coco json files:

```sh
./scripts/create_tma_json_tissue_detection_and_subtract_smaller_compartments.sh
```

Before training, let's look at the [instructions](./docs/train_configs.md) for modifying config files.

Use the distributed training command in the script file `./scripts/train.sh`

```sh
./tools/dist_train.sh ./configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_HULA_compartment.py 4 --work-dir /data/hqvo3/mmdet/run2
```

where
`4` means we use 4 gpus for distributed training, and
`--work-dir` to specify where to save our checkpoint files

If there are some issues with training, please try having a look at [Q&A](./docs/qa.md).

We suggest looking at the original document of MMDetection for more information [link](https://mmdetection.readthedocs.io/en/latest/).

## Inference

You download the checkpoint files [here](), and put them into `./inference/checkpoints`.

Then, have a look at the `inference/scripts/inf.sh` for a sample inference command.

## Evaluation

Evaluation codes are in the folder `eval`. More information at [doc](./docs/eval.md).

# Citation

```
@article{vo2023segmentation,
  title={Segmentation of diagnostic tissue compartments on whole slide images with renal thrombotic microangiopathies (TMAs)},
  author={Vo, Huy Q and Cicalese, Pietro A and Seshan, Surya and Rizvi, Syed A and Vathul, Aneesh and Bueno, Gloria and Dorado, Anibal Pedraza and Grabe, Niels and Stolle, Katharina and Pesce, Francesco and others},
  journal={arXiv preprint arXiv:2311.14971},
  year={2023}
}
```

# Acknowledgement

[MMDetection Toolbox](https://github.com/open-mmlab/mmdetection)
