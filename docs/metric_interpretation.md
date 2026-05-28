# Metric Interpretation

This document explains how to read the metrics used in the CXR-text bridge retrieval experiments.

## Training loss

The model is trained with symmetric InfoNCE loss.

Lower training loss means the model is optimizing the contrastive objective better, but loss alone does not prove good retrieval.

A model can reduce loss while still failing to rank the correct report highly among many candidates.

For this reason, retrieval metrics are required.

## Recall@K

Recall@K measures whether the correct text/report embedding appears in the top K retrieved candidates for a given image query.

| Metric | Meaning |
|---|---|
| Recall@1 | Correct report is ranked first |
| Recall@5 | Correct report appears in the top 5 |
| Recall@10 | Correct report appears in the top 10 |
| Recall@50 | Correct report appears in the top 50 |

Higher Recall@K means stronger image-to-text retrieval.

## Lift@K

Lift@K compares retrieval performance against random retrieval.

For example, if the validation pool contains 6000 candidates, random Recall@50 is:

```text
50 / 6000 = 0.00833
```

If the model reaches Recall@50 = 0.3740, then:

```text
Lift@50 = 0.3740 / 0.00833 = 44.88x
```

This means the model retrieves the correct report within the top 50 about 44.88 times better than random chance.

## Positive-pair similarity

Positive-pair similarity is the average cosine similarity between the correct image-text pairs in the learned embedding space.

A higher value usually means the model is pulling paired samples closer together.

However, positive-pair similarity should not be interpreted alone. A model can increase positive similarity but still fail to rank the correct pair above competing candidates.

This happened in the batch-size comparison: batch size 128 produced higher positive-pair similarity than batch size 512, but batch size 512 gave slightly better retrieval ranking.

## Epoch time

Epoch time measures how long one full training epoch takes.

This is useful for comparing training configurations such as batch size or image resolution.

## Peak CUDA memory

Peak CUDA memory reports the maximum GPU memory allocated during training.

This helps estimate how much room remains for larger models, larger images, larger batches, or deeper encoders.

## Practical reading rule

Use the metrics together:

```text
training loss shows optimization behavior
Recall@K shows retrieval behavior
Lift@K shows improvement over chance
positive similarity shows pair closeness
epoch time shows compute cost
peak CUDA memory shows hardware usage
```

The strongest result is not necessarily the one with the lowest loss or highest positive similarity. For retrieval, Recall@K and Lift@K matter most.