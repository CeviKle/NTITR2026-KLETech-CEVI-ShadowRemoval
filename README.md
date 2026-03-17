# NTIRE 2026 Shadow Removal

## 1. Introduction

This repository contains our implementation for the **NTIRE 2026 Shadow Removal Challenge**. The task focuses on restoring clean images from shadow-affected scenes while preserving natural colors, illumination, and fine details.

Our main implementation is built around the **final_net** pipeline.

---

## 2. Method Overview

We use a two-stage deep learning framework for shadow removal:

* Backbone: `final_net`
* Stage 1: `fusion_net`
* Stage 2: `Restormer`
* Base training: supervised paired learning
* Fine-tuning: PSNR-focused optimization

Main loss functions:

* Base training: **L1 loss**
* Fine-tuning: **PSNR-focused mixed loss**

---

## 3. Repository Structure

```text
Restormer/                 Restormer backbone
saicinpainting/            FFC-based modules
oath-main/                 OATH codebase
weights/                   Pretrained and trained checkpoints
train.py                   Base training
train_psnr_boost.py        PSNR-focused fine-tuning
fine_tune_oath.py          OATH fine-tuning
test.py                    final_net inference
run_epoch200_boost_gpu.sh  Fine-tuning launcher
run_oath_finetune.sh       OATH launcher
```

---

## 4. Installation and Initial Setup

### 4.1 Clone the Repository

```bash
cd /NTIRE2026/runs/C4_ShdwRem
git clone http://10.9.0.115:3000/nikhil_akalwadi/NTIRE_ShadowRemoval_2025.git
cd NTIRE_ShadowRemoval_2025
```

### 4.2 Create the Environment

```bash
conda create --name watformer python=3.8
conda activate watformer
conda install pytorch=1.11 torchvision cudatoolkit=11.3 -c pytorch
pip install numpy matplotlib scikit-learn scikit-image opencv-python timm kornia einops pytorch_lightning
```

### 4.3 Prepare Weights

```bash
mkdir -p weights
```

Required initial files:

* `weights/shadowremoval.pkl`
* `weights/refinement.pkl`

Optional full checkpoints:

* `weights/epoch_150.pth`
* `weights/epoch_160.pth`
* `weights/epoch_170.pth`
* `weights/epoch_180.pth`
* `weights/epoch_190.pth`
* `weights/epoch_200.pth`

---

## 5. GPU Usage

Use the GPU visible inside the active PyTorch environment. In our runs, the working device was:

* **GPU 0 -> Training / Testing**

Verify before running:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

---

## 6. Training

### 6.1 Base Training from the `.pkl` Pair

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --train_inp /NTIRE2026/C4_ShdwRem/train_inp \
  --train_gt /NTIRE2026/C4_ShdwRem/train_gt \
  --valid_dir /NTIRE2026/C4_ShdwRem/valid \
  --paired_eval_root /NTIRE2026/C4_ShdwRem/ntire26_shadow_removal_train \
  --epochs 200 \
  --patch_size 320 \
  --lr 2e-5 \
  --eval_every 2 \
  --device cuda:0 \
  --use_grad_checkpointing \
  --init_full_ckpt "" \
  --out_dir weights
```

### 6.2 Continue Training from an Existing Checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --train_inp /NTIRE2026/C4_ShdwRem/train_inp \
  --train_gt /NTIRE2026/C4_ShdwRem/train_gt \
  --valid_dir /NTIRE2026/C4_ShdwRem/valid \
  --paired_eval_root /NTIRE2026/C4_ShdwRem/ntire26_shadow_removal_train \
  --epochs 40 \
  --patch_size 320 \
  --lr 2e-5 \
  --eval_every 2 \
  --device cuda:0 \
  --use_grad_checkpointing \
  --init_full_ckpt weights/epoch_200.pth \
  --out_dir weights
```

---

## 7. Fine-Tuning

### 7.1 PSNR-Focused Fine-Tuning for `final_net`

```bash
CUDA_VISIBLE_DEVICES=0 python train_psnr_boost.py \
  --train_inp /NTIRE2026/C4_ShdwRem/train_inp \
  --train_gt /NTIRE2026/C4_ShdwRem/train_gt \
  --init_ckpt weights/epoch_200.pth \
  --out_dir weights/boost_epoch200 \
  --epochs 20 \
  --patch_size 256 \
  --lr 1e-5 \
  --loss_preset psnr_focus \
  --val_every 1 \
  --val_max_images 40 \
  --device cuda:0
```


## 8. Validation and Test Inference

### 8.1 `final_net` Validation

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --device cuda:0 \
  --test_dir /NTIRE2026/C4_ShdwRem/valid \
  --full_ckpt weights/epoch_200.pth \
  --tta \
  --tile 512 \
  --tile_overlap 64 \
  --output_dir results_valid
```

### 8.2 `final_net` Test Inference

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --device cuda:0 \
  --test_dir /NTIRE2026/runs/C4_ShdwRem/NTIRE_ShadowRemoval_2025/test_inputs/ntire26_shadow_test_in \
  --full_ckpt weights/epoch_200.pth \
  --tta \
  --tile 512 \
  --tile_overlap 64 \
  --output_dir test_run_b/cand_epoch_200_strong
```


## 9. Submission

Create the submission zip from the output folder:

```bash
cd test_run_b/cand_epoch_200_strong
zip -r ../../C4_submit_test_run_b.zip . -x 'Thumbs.db'
```

Before submission, check:

* correct image count
* no extra folders
* no missing files
* original resolution preserved

---

## 10. Notes

* Base `final_net` training uses **L1 loss**
* Fine-tuning uses **PSNR-focused mixed loss**
* Training patches do **not** change final output size
* Final outputs remain full resolution, e.g. `960x720`
* Do **not** use `sudo` for this project

---

## 11. Current Best Result

Best known hidden-set result so far:

* **`test_run_b` -> `24.68 PSNR`**

---

## 12. Acknowledgement

This repository is based on the project’s original shadow-removal pipeline with integrated `Restormer`, `saicinpainting` modules, and an optional OATH branch for additional experiments.
