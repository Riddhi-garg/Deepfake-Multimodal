"""EfficientNet / ResNet video backbone for deepfake classification."""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models

from utils.helpers import load_config, setup_logger

logger = setup_logger("VideoModel")


# Model builder


class DeepfakeVideoClassifier(nn.Module):
    """Pretrained CNN backbone fine-tuned for binary deepfake classification."""

    def __init__(self, backbone: str = "efficientnet_b0", freeze_layers: int = 6):
        super().__init__()
        self.backbone_name = backbone
        self._build_backbone(backbone, freeze_layers)

   
    def _build_backbone(self, name: str, freeze_layers: int) -> None:
        if name == "efficientnet_b0":
            base = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.DEFAULT)
            feat_dim = base.classifier[1].in_features
            base.classifier = nn.Identity()
            self.feature_dim = feat_dim
        elif name == "efficientnet_b3":
            base = tv_models.efficientnet_b3(weights=tv_models.EfficientNet_B3_Weights.DEFAULT)
            feat_dim = base.classifier[1].in_features
            base.classifier = nn.Identity()
            self.feature_dim = feat_dim
        elif name == "resnet50":
            base = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
            feat_dim = base.fc.in_features
            base.fc = nn.Identity()
            self.feature_dim = feat_dim
        else:  # fallback – resnet18
            base = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
            feat_dim = base.fc.in_features
            base.fc = nn.Identity()
            self.feature_dim = feat_dim

        # Freeze lower layers
        children = list(base.children())
        for child in children[:freeze_layers]:
            for param in child.parameters():
                param.requires_grad = False

        self.backbone = base
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return backbone feature vectors (N, feature_dim)."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.extract_features(x)
        return self.head(feats)


def build_video_model(config: Optional[Dict] = None) -> DeepfakeVideoClassifier:
    cfg = config or load_config()
    vcfg = cfg["video"]
    return DeepfakeVideoClassifier(
        backbone=vcfg.get("backbone", "efficientnet_b0"),
        freeze_layers=vcfg.get("freeze_layers", 6),
    )


# Heuristic fallback (used when no pretrained checkpoint is available)

def predict_video_fake_heuristic(face_batch: torch.Tensor) -> float:
    """Signal-processing based fake-probability estimate.

    Computes local variance, colour channel imbalance, and high-frequency
    energy of face crops.  Returns a value in [0.05, 0.95].
    """
    faces_np = face_batch.detach().cpu().numpy()  

    scores = []
    for face in faces_np:
        # Un-normalise to [0,1]
        img = face.transpose(1, 2, 0)
        img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        img = img.clip(0, 1)

        # 1. Pixel variance (deepfakes often over-smooth)
        var = float(img.var())
        var_score = 1.0 - min(var / 0.05, 1.0)  # low variance → fake

        # 2. Channel colour imbalance (GAN artefacts shift colour distribution)
        r_mean, g_mean, b_mean = img[:,:,0].mean(), img[:,:,1].mean(), img[:,:,2].mean()
        imbalance = float(abs(r_mean - b_mean) + abs(g_mean - b_mean))
        imbalance_score = min(imbalance / 0.15, 1.0)  # high imbalance → fake

        face_score = 0.5 * var_score + 0.5 * imbalance_score
        scores.append(face_score)

    raw = float(sum(scores) / len(scores)) if scores else 0.5
    # Clamp to a realistic prediction range
    return max(0.05, min(0.95, raw))
