# Project Framing: CXR-Text Bridge Retrieval

## Research Direction

This repository studies chest X-ray and radiology-report contrastive retrieval as a bridge between visual medical imaging representations and report/text representations.

The focus is not only whether image-report contrastive learning works, but when it forms a stable retrieval bridge under different training configurations.

## Research Question

When does CXR-text contrastive pretraining form a reliable image-report retrieval bridge, and which training configurations cause the visual branch to align or fail?

## Motivation

Medical vision-language learning often relies on paired images and reports. In chest X-ray datasets, each image or study may be associated with a free-text radiology report, making contrastive image-text learning a natural way to build cross-modal representations.

However, image-report alignment can be unstable. A model may reduce training loss without learning useful retrieval structure, or one branch may dominate while the other branch fails to align.

This repository is designed as a diagnostic study of that alignment behavior.

## Related Work

This project is motivated by medical image-text contrastive learning and chest X-ray vision-language pretraining.

ConVIRT introduced contrastive learning from paired medical images and reports, using a bidirectional contrastive objective to learn medical visual representations from naturally paired descriptive text.

MedCLIP studied contrastive learning from unpaired medical images and text and highlighted a key issue in medical contrastive learning: semantically similar medical samples may be treated as false negatives.

BioViL showed the value of radiology-specific text semantics for biomedical vision-language processing, including the use of report-aware language representations for chest radiography tasks.

CXR-CLIP extended chest X-ray language-image pretraining by using image-label and report-section information to support large-scale CXR vision-language learning.

## Planned Experiment Variables

| Variable | Values |
|---|---|
| Dataset size | 10000, 30000, 60000 pairs |
| Batch size | 128, 512 |
| Visual encoder setting | frozen, partially unfrozen |
| Projection dimension | 128, 256 |
| Loss | symmetric InfoNCE |

The first implementation can start with a smaller smoke-test configuration, then scale to larger runs after the training loop is verified.

## Metrics

The benchmark will report:

- train loss
- Recall@1
- Recall@5
- Recall@10
- Recall@50
- Lift@K over random retrieval
- positive-pair similarity
- epoch time
- CUDA device name
- peak CUDA memory allocation where available

## Compute Reporting

The research question should focus on alignment behavior, not hardware.

Hardware details should be reported separately under a compute setup section.

Example:

```text
Experiments were executed with CUDA-enabled PyTorch on an NVIDIA GeForce RTX 4090. The training script records batch size, epoch time, peak CUDA memory allocation, retrieval metrics, and positive-pair similarity.
```

## References

- **ConVIRT**: Yuhao Zhang, Hang Jiang, Yasuhide Miura, Christopher D. Manning, and Curtis P. Langlotz. *Contrastive Learning of Medical Visual Representations from Paired Images and Text*. 2020.

- **MedCLIP**: Zifeng Wang, Zhenbang Wu, Dinesh Agarwal, and Jimeng Sun. *MedCLIP: Contrastive Learning from Unpaired Medical Images and Text*. 2022.

- **BioViL / CXR-BERT**: Benedikt Boecking et al. *Making the Most of Text Semantics to Improve Biomedical Vision-Language Processing*. ECCV, 2022.

- **CXR-CLIP**: Kihyun You et al. *CXR-CLIP: Toward Large Scale Chest X-ray Language-Image Pre-training*. 2023.