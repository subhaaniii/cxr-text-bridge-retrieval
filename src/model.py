from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCXRImageEncoder(nn.Module):
    """
    Compact CNN encoder for synthetic CXR-like images.

    This is intentionally lightweight but still GPU-relevant:
    convolutional layers, batch normalization, pooling, and projection.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        dropout: float = 0.10,
    ):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.backbone(images)
        x = self.projector(x)
        return F.normalize(x, dim=-1)


class TextEmbeddingEncoder(nn.Module):
    """
    MLP projection head for synthetic report/text embeddings.
    """

    def __init__(
        self,
        text_dim: int = 384,
        hidden_dim: int = 512,
        embed_dim: int = 256,
        dropout: float = 0.10,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        x = self.net(text_embeddings)
        return F.normalize(x, dim=-1)


class CXRTextBridgeModel(nn.Module):
    """
    Dual encoder for image-report contrastive retrieval.
    """

    def __init__(
        self,
        text_dim: int = 384,
        embed_dim: int = 256,
        text_hidden_dim: int = 512,
        dropout: float = 0.10,
    ):
        super().__init__()

        self.image_encoder = SmallCXRImageEncoder(
            embed_dim=embed_dim,
            dropout=dropout,
        )

        self.text_encoder = TextEmbeddingEncoder(
            text_dim=text_dim,
            hidden_dim=text_hidden_dim,
            embed_dim=embed_dim,
            dropout=dropout,
        )

    def forward(
        self,
        images: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_z = self.image_encoder(images)
        text_z = self.text_encoder(text_embeddings)
        return image_z, text_z

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(images)

    def encode_text(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(text_embeddings)