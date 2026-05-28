from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create synthetic CXR-like image/report-embedding pairs for GPU pipeline testing."
    )
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--n-groups", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--text-dim", type=int, default=384)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument(
        "--mode",
        choices=["easy", "shifted", "noisy"],
        default="easy",
        help="Controls the strength of image-text alignment.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("data_generated/demo"))
    return parser.parse_args()


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def settings_for_mode(mode: str) -> dict[str, float]:
    if mode == "easy":
        return {
            "latent_noise": 0.10,
            "image_noise": 0.08,
            "text_noise": 0.08,
            "shift_strength": 0.05,
        }
    if mode == "shifted":
        return {
            "latent_noise": 0.18,
            "image_noise": 0.14,
            "text_noise": 0.18,
            "shift_strength": 0.35,
        }
    if mode == "noisy":
        return {
            "latent_noise": 0.32,
            "image_noise": 0.28,
            "text_noise": 0.30,
            "shift_strength": 0.55,
        }
    raise ValueError(f"Unknown mode: {mode}")


def make_projection(rng: np.random.Generator, in_dim: int, out_dim: int) -> np.ndarray:
    mat = rng.normal(0, 1, size=(in_dim, out_dim))
    return mat / np.sqrt(in_dim)


def make_cxr_like_image(
    rng: np.random.Generator,
    latent_vec: np.ndarray,
    group_id: int,
    image_size: int,
    noise: float,
) -> np.ndarray:
    """
    Synthetic CXR-like grayscale image.

    This is not a medical image. It creates smooth lung-like blobs and
    group-dependent intensity patterns so the image branch has real signal to learn.
    """
    h = w = image_size
    y, x = np.mgrid[-1:1:complex(0, h), -1:1:complex(0, w)]

    left_lung = np.exp(-(((x + 0.35) / 0.35) ** 2 + ((y + 0.02) / 0.70) ** 2))
    right_lung = np.exp(-(((x - 0.35) / 0.35) ** 2 + ((y + 0.02) / 0.70) ** 2))
    lungs = left_lung + right_lung

    rib_pattern = 0.10 * np.sin((y + 1.2) * np.pi * (5 + group_id % 4))
    vertical_gradient = 0.15 * (1 - y)

    latent_signal = (
        0.22 * latent_vec[0] * left_lung
        + 0.22 * latent_vec[1] * right_lung
        + 0.12 * latent_vec[2] * np.exp(-((x / 0.20) ** 2 + ((y - 0.2) / 0.45) ** 2))
        + 0.08 * latent_vec[3] * np.sin((x + y) * np.pi * 3)
    )

    image = 0.45 + 0.35 * lungs + rib_pattern + vertical_gradient + latent_signal
    image += rng.normal(0, noise, size=(h, w))

    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    image = (image * 255).clip(0, 255).astype(np.uint8)
    return image


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    settings = settings_for_mode(args.mode)

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = args.out_dir if args.out_dir.is_absolute() else repo_root / args.out_dir
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    if args.n_samples < args.n_groups:
        raise ValueError("--n-samples must be >= --n-groups")

    sample_ids = np.arange(args.n_samples)

    group_ids = np.arange(args.n_samples) % args.n_groups
    rng.shuffle(group_ids)

    group_centers = rng.normal(0, 1, size=(args.n_groups, args.latent_dim))
    group_centers = normalize_rows(group_centers)

    latent = group_centers[group_ids] + rng.normal(
        0,
        settings["latent_noise"],
        size=(args.n_samples, args.latent_dim),
    )
    latent = normalize_rows(latent)

    text_proj_base = make_projection(rng, args.latent_dim, args.text_dim)
    text_proj_shift = make_projection(rng, args.latent_dim, args.text_dim)

    text_proj = (
        (1.0 - settings["shift_strength"]) * text_proj_base
        + settings["shift_strength"] * text_proj_shift
    )

    text_embeddings = latent @ text_proj + rng.normal(
        0,
        settings["text_noise"],
        size=(args.n_samples, args.text_dim),
    )
    text_embeddings = normalize_rows(text_embeddings)

    rows = []

    for i in range(args.n_samples):
        image = make_cxr_like_image(
            rng=rng,
            latent_vec=latent[i],
            group_id=int(group_ids[i]),
            image_size=args.image_size,
            noise=settings["image_noise"],
        )

        image_path = image_dir / f"cxr_demo_{i:06d}.png"
        Image.fromarray(image, mode="L").convert("RGB").save(image_path)

        row = {
            "sample_id": int(sample_ids[i]),
            "image_path": str(image_path),
            "group_id": int(group_ids[i]),
        }

        for j in range(args.text_dim):
            row[f"text_emb_{j:03d}"] = float(text_embeddings[i, j])

        for j in range(args.latent_dim):
            row[f"latent_{j:03d}"] = float(latent[i, j])

        rows.append(row)

    manifest = pd.DataFrame(rows)
    manifest_path = out_dir / "manifest.csv"
    metadata_path = out_dir / "metadata.json"

    manifest.to_csv(manifest_path, index=False)

    metadata = {
        "mode": args.mode,
        "n_samples": args.n_samples,
        "n_groups": args.n_groups,
        "image_size": args.image_size,
        "text_dim": args.text_dim,
        "latent_dim": args.latent_dim,
        "seed": args.seed,
        "settings": settings,
        "note": "Synthetic CXR-like images and synthetic report embeddings. Not real medical data.",
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Mode       : {args.mode}")
    print(f"Samples    : {args.n_samples}")
    print(f"Groups     : {args.n_groups}")
    print(f"Manifest   : {manifest_path}")
    print(f"Images     : {image_dir}")
    print(f"Metadata   : {metadata_path}")
    print("Note       : synthetic images and embeddings, not real medical data.")


if __name__ == "__main__":
    main()