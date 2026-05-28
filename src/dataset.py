from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CXRTextBridgeDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str | Path,
        image_size: int = 64,
    ):
        self.manifest_csv = Path(manifest_csv)
        self.df = pd.read_csv(self.manifest_csv)

        self.text_cols = sorted([c for c in self.df.columns if c.startswith("text_emb_")])

        if len(self.text_cols) == 0:
            raise RuntimeError("No text_emb_* columns found in manifest.")

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        image_path = Path(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)

        text_embedding = row[self.text_cols].to_numpy(dtype=np.float32)
        text_tensor = torch.from_numpy(text_embedding)

        sample_id = int(row["sample_id"])

        return {
            "image": image_tensor,
            "text": text_tensor,
            "sample_id": sample_id,
        }


def collate_batch(batch):
    images = torch.stack([item["image"] for item in batch])
    texts = torch.stack([item["text"] for item in batch])
    sample_ids = torch.tensor([item["sample_id"] for item in batch], dtype=torch.long)

    return {
        "image": images,
        "text": texts,
        "sample_id": sample_ids,
    }