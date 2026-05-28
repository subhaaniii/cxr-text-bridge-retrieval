# GPU Training Protocol

This document describes how the CXR-text bridge retrieval experiments were run.

## Purpose

The purpose of the GPU training protocol is to evaluate whether an image-text contrastive model can form a reliable retrieval bridge under controlled difficulty settings.

The experiment tracks both retrieval quality and compute behavior.

## Compute Setup

The main experiments were run using CUDA-enabled PyTorch on:

```text
NVIDIA GeForce RTX 4090
CUDA 12.8
Mixed precision enabled
```

The training script records:

- CUDA device name
- CUDA availability
- CUDA version
- total GPU memory
- peak CUDA memory allocation
- batch size
- epoch time
- training loss
- retrieval metrics

## Data Generation

Generate a controlled CXR-text dataset:

```powershell
python src/make_demo_data.py --mode shifted --n-samples 10000 --n-groups 200 --image-size 64
```

Available modes:

```text
easy
shifted
noisy
```

The generated images are synthetic CXR-like patterns, not clinical images.

## Training

Example training command:

```powershell
python src/train.py --epochs 10 --batch-size 512 --num-workers 0 --output-dir outputs/shifted_demo --checkpoint-dir checkpoints/shifted_demo
```

For the RTX 4090 environment, the experiments were run through the `mmembed` conda environment:

```powershell
& "C:\ProgramData\anaconda3\Scripts\conda.exe" run --no-capture-output -n mmembed python src/train.py --epochs 10 --batch-size 512 --num-workers 0 --output-dir outputs/shifted_demo --checkpoint-dir checkpoints/shifted_demo
```

## Larger Experiment Example

Generate 30k shifted data with image size 96:

```powershell
& "C:\ProgramData\anaconda3\Scripts\conda.exe" run -n mmembed python src/make_demo_data.py --mode shifted --n-samples 30000 --n-groups 300 --image-size 96
```

Train with batch size 512:

```powershell
& "C:\ProgramData\anaconda3\Scripts\conda.exe" run --no-capture-output -n mmembed python src/train.py --epochs 10 --batch-size 512 --image-size 96 --num-workers 0 --output-dir outputs/shifted_n30000_bs512_d256_img96 --checkpoint-dir checkpoints/shifted_n30000_bs512_d256_img96
```

## Result Collection

After running experiments:

```powershell
python src/collect_results.py
```

This writes:

```text
experiments/results_table.csv
```

## Generated Files

The following folders are generated locally and ignored by Git:

```text
data_generated/
outputs/
checkpoints/
```

These folders contain synthetic images, run logs, and model checkpoints. They are not committed to keep the repository lightweight.

The repository tracks source code, documentation, and aggregate result tables.