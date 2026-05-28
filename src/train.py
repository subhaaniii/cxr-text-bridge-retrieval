from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import CXRTextBridgeDataset, collate_batch
from metrics import retrieval_metrics, symmetric_infonce_loss
from model import CXRTextBridgeModel


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 10
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    temperature: float = 0.07
    embed_dim: int = 256
    text_hidden_dim: int = 512
    dropout: float = 0.10
    image_size: int = 64
    val_fraction: float = 0.20
    num_workers: int = 0
    use_amp: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CXR-text bridge contrastive retrieval model."
    )

    parser.add_argument("--manifest-csv", type=Path, default=Path("data_generated/demo/manifest.csv"))
    parser.add_argument("--metadata-json", type=Path, default=Path("data_generated/demo/metadata.json"))

    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--temperature", type=float, default=TrainConfig.temperature)
    parser.add_argument("--embed-dim", type=int, default=TrainConfig.embed_dim)
    parser.add_argument("--text-hidden-dim", type=int, default=TrainConfig.text_hidden_dim)
    parser.add_argument("--dropout", type=float, default=TrainConfig.dropout)
    parser.add_argument("--image-size", type=int, default=TrainConfig.image_size)
    parser.add_argument("--val-fraction", type=float, default=TrainConfig.val_fraction)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)

    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo_run"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/demo_run"))

    return parser.parse_args()


def resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device_info(device: torch.device) -> dict[str, str | int | float | bool]:
    info: dict[str, str | int | float | bool] = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(idx),
                "cuda_device_index": idx,
                "cuda_total_memory_gb": round(props.total_memory / (1024 ** 3), 3),
                "torch_cuda_version": torch.version.cuda or "unknown",
            }
        )
    else:
        info.update(
            {
                "cuda_device_name": "cpu",
                "cuda_device_index": -1,
                "cuda_total_memory_gb": 0.0,
                "torch_cuda_version": "none",
            }
        )

    return info


def split_dataset(dataset: CXRTextBridgeDataset, seed: int, val_fraction: float):
    n_total = len(dataset)
    n_val = max(1, int(round(n_total * val_fraction)))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    return train_ds, val_ds


def infer_text_dim(dataset: CXRTextBridgeDataset) -> int:
    return len(dataset.text_cols)


def make_loaders(
    dataset: CXRTextBridgeDataset,
    cfg: TrainConfig,
):
    train_ds, val_ds = split_dataset(dataset, seed=cfg.seed, val_fraction=cfg.val_fraction)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )

    return train_loader, val_loader, len(train_ds), len(val_ds)


def train_one_epoch(
    model: CXRTextBridgeModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    cfg: TrainConfig,
) -> dict[str, float]:
    model.train()

    total_loss = 0.0
    total_pos_sim = 0.0
    total_neg_sim = 0.0
    steps = 0

    use_amp = cfg.use_amp and device.type == "cuda"

    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        texts = batch["text"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            image_z, text_z = model(images, texts)
            loss, loss_metrics = symmetric_infonce_loss(
                image_z=image_z,
                text_z=text_z,
                temperature=cfg.temperature,
            )

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_pos_sim += loss_metrics["batch_pos_sim"]
        total_neg_sim += loss_metrics["batch_neg_sim"]
        steps += 1

    return {
        "train_loss": total_loss / max(steps, 1),
        "train_batch_pos_sim": total_pos_sim / max(steps, 1),
        "train_batch_neg_sim": total_neg_sim / max(steps, 1),
    }


@torch.no_grad()
def evaluate(
    model: CXRTextBridgeModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    all_image_z = []
    all_text_z = []
    all_ids = []

    for batch in tqdm(loader, desc="eval", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        texts = batch["text"].to(device, non_blocking=True)

        image_z, text_z = model(images, texts)

        all_image_z.append(image_z.cpu())
        all_text_z.append(text_z.cpu())
        all_ids.extend(batch["sample_id"].cpu().numpy().tolist())

    image_z = torch.cat(all_image_z, dim=0)
    text_z = torch.cat(all_text_z, dim=0)
    sample_ids = np.asarray(all_ids, dtype=np.int64)

    return retrieval_metrics(
        image_z=image_z,
        text_z=text_z,
        sample_ids=sample_ids,
        k_values=(1, 5, 10, 50),
    )


def save_checkpoint(
    path: Path,
    model: CXRTextBridgeModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    row: dict,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": row,
            "config": config,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    manifest_path = resolve_path(args.manifest_csv, repo_root)
    metadata_path = resolve_path(args.metadata_json, repo_root)
    output_dir = resolve_path(args.output_dir, repo_root)
    checkpoint_dir = resolve_path(args.checkpoint_dir, repo_root)

    cfg = TrainConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        embed_dim=args.embed_dim,
        text_hidden_dim=args.text_hidden_dim,
        dropout=args.dropout,
        image_size=args.image_size,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
    )

    set_seed(cfg.seed)

    dataset = CXRTextBridgeDataset(
        manifest_csv=manifest_path,
        image_size=cfg.image_size,
    )

    text_dim = infer_text_dim(dataset)
    train_loader, val_loader, n_train, n_val = make_loaders(dataset, cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_info = get_device_info(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model = CXRTextBridgeModel(
        text_dim=text_dim,
        embed_dim=cfg.embed_dim,
        text_hidden_dim=cfg.text_hidden_dim,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scaler = GradScaler(enabled=cfg.use_amp and device.type == "cuda")

    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    run_config = {
        **asdict(cfg),
        **device_info,
        "manifest_csv": str(manifest_path),
        "metadata_json": str(metadata_path),
        "n_total": int(len(dataset)),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "text_dim": int(text_dim),
        "data_mode": metadata.get("mode", "unknown"),
        "n_groups": metadata.get("n_groups", "unknown"),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print("CXR-text bridge training")
    print(f"Manifest     : {manifest_path}")
    print(f"Samples      : {len(dataset)}")
    print(f"Train / Val  : {n_train} / {n_val}")
    print(f"Text dim     : {text_dim}")
    print(f"Device       : {device_info['cuda_device_name']}")
    print(f"CUDA memory  : {device_info['cuda_total_memory_gb']} GB")
    print(f"Batch size   : {cfg.batch_size}")
    print(f"AMP enabled  : {cfg.use_amp and device.type == 'cuda'}")
    print(f"Output dir   : {output_dir}")

    history = []
    metrics_path = output_dir / "metrics.csv"
    best_lift50 = -math.inf

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            cfg=cfg,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
        )

        epoch_time_sec = time.perf_counter() - epoch_start

        peak_memory_gb = 0.0
        if device.type == "cuda":
            peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        row = {
            "epoch": float(epoch),
            **train_metrics,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "epoch_time_sec": float(epoch_time_sec),
            "peak_cuda_memory_gb": float(peak_memory_gb),
            "batch_size": float(cfg.batch_size),
            "embed_dim": float(cfg.embed_dim),
            "n_train": float(n_train),
            "n_val": float(n_val),
        }

        history.append(row)
        pd.DataFrame(history).to_csv(metrics_path, index=False)

        print(
            f"Epoch {epoch:03d}/{cfg.epochs} "
            f"loss={row['train_loss']:.4f} "
            f"R@10={row['val_recall@10']:.4f} "
            f"R@50={row['val_recall@50']:.4f} "
            f"Lift@50={row['val_lift@50']:.2f}x "
            f"pos_sim={row['val_pos_sim_mean']:.4f} "
            f"time={epoch_time_sec:.1f}s "
            f"peak_mem={peak_memory_gb:.2f}GB"
        )

        save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, epoch, row, run_config)

        if row["val_lift@50"] > best_lift50:
            best_lift50 = row["val_lift@50"]
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, epoch, row, run_config)

    print("\nDone.")
    print(f"Best Lift@50 : {best_lift50:.2f}x")
    print(f"Metrics      : {metrics_path}")
    print(f"Best ckpt    : {checkpoint_dir / 'best.pt'}")


if __name__ == "__main__":
    main()