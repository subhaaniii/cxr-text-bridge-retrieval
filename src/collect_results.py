from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect CXR-text bridge training results.")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--out-csv", type=Path, default=Path("experiments/results_table.csv"))
    return parser.parse_args()


def parse_run_name(run_name: str) -> dict[str, str]:
    parts = run_name.split("_")

    info = {
        "run_name": run_name,
        "mode": "unknown",
        "n_samples": "unknown",
        "batch_size": "unknown",
        "embed_dim": "unknown",
    }

    for part in parts:
        if part in {"easy", "shifted", "noisy"}:
            info["mode"] = part

        if part.startswith("n") and part[1:].isdigit():
            info["n_samples"] = part[1:]

        if part.startswith("bs") and part[2:].isdigit():
            info["batch_size"] = part[2:]

        if part.startswith("d") and part[1:].isdigit():
            info["embed_dim"] = part[1:]

    return info


def main() -> None:
    args = parse_args()

    rows = []

    for metrics_path in sorted(args.outputs_dir.glob("*/metrics.csv")):
        run_dir = metrics_path.parent
        metrics_df = pd.read_csv(metrics_path)

        if metrics_df.empty:
            continue

        final = metrics_df.iloc[-1].to_dict()
        parsed = parse_run_name(run_dir.name)

        config_path = run_dir / "config.json"
        config = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

        row = {
            **parsed,
            "epoch": final.get("epoch"),
            "train_loss": final.get("train_loss"),
            "train_batch_pos_sim": final.get("train_batch_pos_sim"),
            "train_batch_neg_sim": final.get("train_batch_neg_sim"),
            "val_recall@1": final.get("val_recall@1"),
            "val_recall@5": final.get("val_recall@5"),
            "val_recall@10": final.get("val_recall@10"),
            "val_recall@50": final.get("val_recall@50"),
            "val_lift@1": final.get("val_lift@1"),
            "val_lift@5": final.get("val_lift@5"),
            "val_lift@10": final.get("val_lift@10"),
            "val_lift@50": final.get("val_lift@50"),
            "val_pos_sim_mean": final.get("val_pos_sim_mean"),
            "val_pos_sim_median": final.get("val_pos_sim_median"),
            "val_n_pool": final.get("val_n_pool"),
            "epoch_time_sec": final.get("epoch_time_sec"),
            "peak_cuda_memory_gb": final.get("peak_cuda_memory_gb"),
            "config_batch_size": config.get("batch_size"),
            "config_embed_dim": config.get("embed_dim"),
            "config_n_total": config.get("n_total"),
            "config_n_train": config.get("n_train"),
            "config_n_val": config.get("n_val"),
            "data_mode": config.get("data_mode"),
            "cuda_device_name": config.get("cuda_device_name"),
            "cuda_total_memory_gb": config.get("cuda_total_memory_gb"),
            "torch_cuda_version": config.get("torch_cuda_version"),
            "use_amp": config.get("use_amp"),
        }

        rows.append(row)

    if not rows:
        raise RuntimeError(f"No metrics.csv files found under {args.outputs_dir}")

    out_df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    print(f"Wrote {len(out_df)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()