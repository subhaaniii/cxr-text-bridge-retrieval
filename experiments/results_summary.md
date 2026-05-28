# CXR-Text Bridge Retrieval: GPU Experiment Summary

## Overview

This experiment studies whether CXR-like image representations and report/text embeddings can form a reliable retrieval bridge through contrastive pretraining.

The focus is not only whether the model learns, but when the bridge forms, weakens, or collapses under different data difficulty settings.

The experiment was run with CUDA-enabled PyTorch on an NVIDIA GeForce RTX 4090.

---

## Research Question

When does CXR-text contrastive pretraining form a reliable image-report retrieval bridge, and which training configurations cause the visual branch to align or fail?

---

## Compute Setup

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 |
| CUDA available | True |
| CUDA version | 12.8 |
| GPU memory | 23.988 GB |
| Mixed precision | Enabled |
| Framework | PyTorch |

The training script logs epoch time, peak CUDA memory, batch size, retrieval metrics, and positive-pair similarity.

---

## Experimental Setup

The benchmark uses synthetic CXR-like images paired with synthetic report/text embeddings.

The generated images are not medical images. They are controlled CXR-like patterns used to test whether the training pipeline can align an image branch and a text branch under known difficulty levels.

Three data modes were tested:

| Mode | Meaning |
|---|---|
| Easy | Strong image-text alignment signal |
| Shifted | Cross-modal relation is harder due to representation shift |
| Noisy | Image and text signals are heavily corrupted |

---

## Model

The model uses a dual-encoder architecture:

| Branch | Encoder |
|---|---|
| Image branch | Small CNN encoder |
| Text branch | MLP projection over report/text embeddings |
| Shared space | L2-normalized embedding space |
| Loss | Symmetric InfoNCE |

---

## Runs Completed

| Run | Samples | Image size | Batch size | Val pool | R@1 | R@10 | R@50 | Lift@50 | Pos Sim | Epoch Time | Peak GPU Mem |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| easy 10k | 10000 | 64 | 512 | 2000 | 0.3035 | 0.8185 | 0.9810 | 39.24x | 0.8992 | 12.1s | 0.23 GB |
| shifted 10k | 10000 | 64 | 512 | 2000 | 0.0300 | 0.1885 | 0.4705 | 18.82x | 0.6164 | 12.2s | 0.23 GB |
| noisy 10k | 10000 | 64 | 512 | 2000 | 0.0015 | 0.0065 | 0.0290 | 1.16x | 0.0109 | 12.1s | 0.23 GB |
| easy 30k | 30000 | 96 | 512 | 6000 | 0.2002 | 0.6972 | 0.9505 | 114.06x | 0.9050 | 41.3s | 0.47 GB |
| shifted 30k | 30000 | 96 | 512 | 6000 | 0.0253 | 0.1420 | 0.3740 | 44.88x | 0.7263 | 41.1s | 0.47 GB |
| shifted 30k | 30000 | 96 | 128 | 6000 | 0.0218 | 0.1412 | 0.3673 | 44.08x | 0.7582 | 54.5s | 0.15 GB |
| noisy 30k | 30000 | 96 | 512 | 6000 | 0.0008 | 0.0072 | 0.0303 | 3.64x | 0.4833 | 41.0s | 0.47 GB |

---

## Main Findings

### 1. The bridge forms strongly in easy mode

The easy setting produced high retrieval performance.

At 10k samples, the model reached:

```text
Recall@50 = 0.9810
Lift@50   = 39.24x
```

At 30k samples, the validation pool became larger, but the model still reached:

```text
Recall@50 = 0.9505
Lift@50   = 114.06x
```

This confirms that the training pipeline can learn a strong image-text retrieval bridge when the cross-modal signal is clear.

---

### 2. Shifted mode creates a useful middle difficulty

The shifted setting is the most informative condition.

At 10k samples:

```text
Recall@50 = 0.4705
Lift@50   = 18.82x
```

At 30k samples:

```text
Recall@50 = 0.3740
Lift@50   = 44.88x
```

The absolute Recall@50 is lower at 30k because the retrieval pool is larger. However, the lift over random becomes much higher, meaning the model remains far above chance even as the candidate pool grows.

This suggests that the bridge partially forms under cross-modal shift.

---

### 3. Noisy mode exposes the failure boundary

The noisy setting caused the bridge to nearly collapse.

At 10k samples:

```text
Recall@50 = 0.0290
Lift@50   = 1.16x
```

At 30k samples:

```text
Recall@50 = 0.0303
Lift@50   = 3.64x
```

The model still learns some similarity structure at 30k, but retrieval remains weak compared with easy and shifted modes.

This shows that contrastive training cannot recover a reliable bridge when both visual and text signals are heavily corrupted.

---

### 4. Batch size affects retrieval and compute behavior

For shifted 30k with image size 96:

| Batch size | R@50 | Lift@50 | Pos Sim | Epoch Time | Peak GPU Mem |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.3740 | 44.88x | 0.7263 | 41.1s | 0.47 GB |
| 128 | 0.3673 | 44.08x | 0.7582 | 54.5s | 0.15 GB |

Batch size 512 gave slightly better retrieval and faster epoch time. Batch size 128 produced higher positive-pair similarity, but this did not translate into stronger ranking performance.

This supports an important interpretation:

```text
Higher positive-pair similarity does not always mean better retrieval ranking.
```

---

## Key Interpretation

The experiment shows a clear alignment ladder:

```text
easy    -> bridge forms strongly
shifted -> bridge forms partially
noisy   -> bridge mostly collapses
```

This makes the repo useful as a diagnostic study of CXR-text bridge formation.

The results also show why retrieval metrics are necessary. Training loss and positive similarity alone do not fully explain whether the model can rank the correct report among many candidates.

---

## Compute Observation

The RTX 4090 enabled larger batch contrastive training with mixed precision.

The 30k image-size-96 runs completed at roughly:

```text
~41 seconds per epoch with batch size 512
```

Peak memory remained below 1 GB because the current CNN is compact and the images are 96x96. This leaves room for future experiments with deeper image encoders, larger image resolution, real CXR images, or larger batch sizes.

---

## Conclusion

CXR-text contrastive bridge retrieval is highly sensitive to data difficulty.

The bridge forms reliably when image and text signals are aligned, remains partially useful under representation shift, and collapses under high noise.

The main lesson is:

```text
CXR-text bridge quality should be evaluated with retrieval metrics, not only contrastive loss or positive-pair similarity.
```

Future extensions should test stronger CXR encoders, real report embeddings, frozen versus partially unfrozen visual backbones, and authorized real chest X-ray datasets.