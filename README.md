[FADE: Few-shot/zero-shot Anomaly Detection Engine using Large Vision-Language Model](https://arxiv.org/abs/2409.00556)
---
This repository contains the official PyTorch implementation of 
[FADE: Few-shot/zero-shot Anomaly Detection Engine using Large 
Vision-Language Model](https://arxiv.org/abs/2409.00556), BMVC 2024.

<p align="center">
    <img src="media/mvtec_shots_study.png" alt="study" width="70%">
    <br>
    <em>Zero-/one-shot anomaly segmentation results of FADE on the MVTec dataset.</em>
</p>

## Prerequisites

For dependencies see `poetry.yaml`.

### Install python dependencies
```
poetry install
```

### Download evaluation datasets
* **MVTec-AD** dataset can be downloaded from https://www.mvtec.com/company/research/datasets/mvtec-ad
* **VisA** dataset can be downloaded from https://paperswithcode.com/dataset/visa

## Running FADE
The algorithm does not require any funetuning nor auxiliary training datasets.
`scripts/run_fade.py` allows to run evaluation on MVTec or VisA and:
1) compute text and visual features, the type of features, CLIP or GEM, is specified through command line arguments.
2) build a memory bank for few-shot anomaly detection if the non-zero number of shots is chosen.

To evaluate on a custom dataset please use `datasets.BaseDataset` as a base class. 
For the full set of command line arguments please refer to the click descriptions in `scripts/run_fade.py`.

### Zero-shot settings
Here are examples of running **zero-shot** FADE with language- and vision-guided anomaly classification (AC) and segmentation (AS).

#### Zero-shot language-guided AC and AS
```
DATASET=mvtec
DATASET_PATH=../mvtec
SEG_FEATURE=gem
CLASS_FEATURE=clip
SHOTS=0

python scripts/run_fade.py \
    --dataset-name $DATASET \
    --dataset-path $DATASET_PATH
    --experiment-name $DATASET/zeroshot/cm_language_sm_language_${CLASS_FEATURE}_${SEG_FEATURE}/img_size_$SZ \
    --classification-mode language \
    --segmentation-mode language \
    --shots ${SHOTS} \
    --language-classification-feature $CLASS_FEATURE \
    --language-segmentation-feature $SEG_FEATURE
```
#### Zero-shot vision-guided AC and AS
```
DATASET=mvtec
DATASET_PATH=../mvtec
VIS_FEATURE=gem
CLASS_FEATURE=clip
SHOTS=0

python scripts/run_fade.py \
  --dataset-name $DATASET \
  --dataset-path $DATASET_PATH
  --experiment-name $DATASET/zeroshot/cm_vision_sm_vision_${CLASS_FEATURE}_${VIS_FEATURE}/img_size_${SZ} \
  --classification-mode both \
  --segmentation-mode both \
  --shots $SHOTS
  --language-classification-feature $CLASS_FEATURE \
  --vision-feature $VIS_FEATURE \
  --use-query-img-in-vision-memory-bank
```

#### Zero-shot language- and vision-guided AC and AS
```
DATASET=mvtec
DATASET_PATH=../mvtec
VIS_FEATURE=gem
SEG_FEATURE=gem
CLASS_FEATURE=clip
SHOTS=0

python scripts/run_fade.py \
  --dataset-name $DATASET \
  --dataset-path $DATASET_PATH
  --experiment-name $DATASET/zeroshot/cm_both_sm_both_${CLASS_FEATURE}_${SEG_FEATURE}_${VIS_FEATURE}/img_size_${SZ} \
  --classification-mode both \
  --segmentation-mode both \
  --shots $SHOTS
  --language-classification-feature $CLASS_FEATURE \
  --language-segmentation-feature $SEG_FEATURE \
  --vision-feature $VIS_FEATURE \
  --vision-segmentation-multiplier 80.0 \
  --vision-segmentation-weight 0.5 \
  --use-query-img-in-vision-memory-bank
```

### Few-shot settings
Below you can find examples of running **few-shot** FADE 
with language- and vision-guided anomaly classification and segmentation. 

#### Few-shot language- and vision-guided AC and AS
```
DATASET=mvtec
DATASET_PATH=../mvtec
VIS_FEATURE=clip
SEG_FEATURE=gem
CLASS_FEATURE=clip
SHOTS=1
SEED=0

python scripts/run_fade.py \
  --dataset-name $DATASET \
  --dataset-path $DATASET_PATH
  --experiment-name $DATASET/fewshot/cm_both_sm_both_${CLASS_FEATURE}_${SEG_FEATURE}_${VIS_FEATURE}/img_size_${SZ}/shot_${SHOTS}/seed_${SEED} \
  --classification-mode both \
  --segmentation-mode both \
  --shots $SHOTS \
  --seed $SEED \
  --language-classification-feature $CLASS_FEATURE \
  --language-segmentation-feature $SEG_FEATURE \
  --vision-feature $VIS_FEATURE \
  --vision-segmentation-multiplier 3.5 \
  --vision-segmentation-weight 0.85
```

To perform language- or vision-guided only AC and AS please set `classification-mode` and `segmentation-mode` 
to `language` or `vision` respectively.

## Citation
If our work is helpful for your research please consider citing:

```
@misc{li2024fade,
      title={FADE: Few-shot/zero-shot Anomaly Detection Engine using Large Vision-Language Model}, 
      author={Yuanwei Li and Elizaveta Ivanova and Martins Bruveris},
      year={2024},
      eprint={2409.00556},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.00556}, 
}
```