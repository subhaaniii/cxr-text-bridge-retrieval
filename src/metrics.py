from __future__ import annotations

import math

import numpy as np
import torch


@torch.no_grad()
def retrieval_metrics(
    image_z: torch.Tensor,
    text_z: torch.Tensor,
    sample_ids: np.ndarray,
    k_values: tuple[int, ...] = (1, 5, 10, 50),
) -> dict[str, float]:
    """
    Image-to-text retrieval.

    Assumption:
    image_z[i] and text_z[i] are the true matched pair after sorting/collecting
    by the same sample order.
    """
    image_z = image_z.float().cpu()
    text_z = text_z.float().cpu()

    sim = (image_z @ text_z.T).numpy()

    n_queries, n_candidates = sim.shape
    if n_queries != n_candidates:
        raise ValueError("Expected same number of image and text embeddings.")

    max_k = min(max(k_values), n_candidates)

    top_part = np.argpartition(sim, -max_k, axis=1)[:, -max_k:]
    top_scores = sim[np.arange(n_queries)[:, None], top_part]
    order = np.argsort(top_scores, axis=1)[:, ::-1]
    top_sorted = top_part[np.arange(n_queries)[:, None], order]

    true_indices = np.arange(n_queries)

    metrics: dict[str, float] = {}

    for k_req in k_values:
        k = min(k_req, n_candidates)
        hit = (top_sorted[:, :k] == true_indices[:, None]).any(axis=1)

        recall = float(hit.mean())
        random_recall = float(min(k / n_candidates, 1.0))

        metrics[f"recall@{k_req}"] = recall
        metrics[f"lift@{k_req}"] = (
            float(recall / random_recall) if random_recall > 0 else math.inf
        )

    pos_sim = sim[np.arange(n_queries), true_indices]

    metrics["pos_sim_mean"] = float(pos_sim.mean())
    metrics["pos_sim_median"] = float(np.median(pos_sim))
    metrics["n_pool"] = float(n_candidates)

    return metrics


def symmetric_infonce_loss(
    image_z: torch.Tensor,
    text_z: torch.Tensor,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Symmetric image-text InfoNCE.
    """
    batch_size = image_z.size(0)
    labels = torch.arange(batch_size, device=image_z.device)

    logits = image_z @ text_z.T / temperature

    loss_i2t = torch.nn.functional.cross_entropy(logits, labels)
    loss_t2i = torch.nn.functional.cross_entropy(logits.T, labels)

    loss = 0.5 * (loss_i2t + loss_t2i)

    with torch.no_grad():
        sim = image_z @ text_z.T
        eye = torch.eye(batch_size, dtype=torch.bool, device=image_z.device)

        pos_sim = sim.diagonal().mean().item()
        neg_sim = sim[~eye].mean().item()

    return loss, {
        "batch_pos_sim": pos_sim,
        "batch_neg_sim": neg_sim,
    }