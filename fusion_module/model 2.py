"""Fusion module: Attention-based feature-level fusion + late fusion."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from utils.helpers import load_config, setup_logger

logger = setup_logger("FusionModel")


# ─────────────────────────────────────────────────────────────────────────────
# Attention-based feature fusion
# ─────────────────────────────────────────────────────────────────────────────

class AttentionFusionClassifier(nn.Module):
    """Projects video and audio features into a shared space, applies multi-head
    self-attention, then classifies through an MLP.

    Architecture:
        Video feats → Linear(video_dim, proj_dim)
        Audio feats → Linear(audio_dim, proj_dim)
        Concat       → (1, 2 * proj_dim)
        Attention    → MultiheadAttention
        BN → Dropout → FC → ReLU → FC → Softmax
    """

    def __init__(
        self,
        video_dim: int = 1280,   # EfficientNet-B0 output
        audio_dim: int = 956,    # 188 handcrafted + 768 Wav2Vec2
        proj_dim: int = 256,
        n_heads: int = 4,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.video_proj = nn.Linear(video_dim, proj_dim)
        self.audio_proj = nn.Linear(audio_dim, proj_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=proj_dim * 2,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(proj_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(proj_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
        )

    def forward(self, video_feats: torch.Tensor, audio_feats: torch.Tensor) -> torch.Tensor:
        """Args:
            video_feats: (B, video_dim)
            audio_feats: (B, audio_dim)
        Returns:
            logits: (B, 2)
        """
        # Pad / truncate to expected dims
        video_feats = self._adapt(video_feats, self.video_proj.in_features)
        audio_feats = self._adapt(audio_feats, self.audio_proj.in_features)

        v = self.video_proj(video_feats)   # (B, proj_dim)
        a = self.audio_proj(audio_feats)   # (B, proj_dim)
        fused = torch.cat([v, a], dim=-1)  # (B, 2*proj_dim)

        # Attention expects (B, seq_len, embed_dim) – treat each sample as seq_len=1
        fused_3d = fused.unsqueeze(1)       # (B, 1, 2*proj_dim)
        attended, _ = self.attention(fused_3d, fused_3d, fused_3d)
        attended = attended.squeeze(1)      # (B, 2*proj_dim)

        return self.classifier(attended)

    @staticmethod
    def _adapt(x: torch.Tensor, target_dim: int) -> torch.Tensor:
        if x.shape[-1] == target_dim:
            return x
        if x.shape[-1] < target_dim:
            pad = torch.zeros(*x.shape[:-1], target_dim - x.shape[-1], device=x.device)
            return torch.cat([x, pad], dim=-1)
        return x[..., :target_dim]


def build_fusion_model(config: Optional[Dict] = None) -> AttentionFusionClassifier:
    cfg = config or load_config()
    fcfg = cfg.get("fusion", {})
    proj_dim = fcfg.get("projection_dim", 256)
    n_heads = fcfg.get("attention_heads", 4)
    dropout = fcfg.get("dropout", 0.4)
    return AttentionFusionClassifier(
        proj_dim=proj_dim, n_heads=n_heads, dropout=dropout
    )


# ─────────────────────────────────────────────────────────────────────────────
# Late fusion helper
# ─────────────────────────────────────────────────────────────────────────────

def run_late_fusion(
    video_prob: float,
    audio_prob: float,
    weight_video: float = 0.6,
    weight_audio: float = 0.4,
) -> Dict[str, object]:
    """Weighted late fusion of video and audio probabilities.

    Returns:
        dict with keys 'probability', 'label', 'confidence'
    """
    final = weight_video * video_prob + weight_audio * audio_prob
    final = float(max(0.0, min(1.0, final)))
    label = "Fake" if final >= 0.5 else "Real"
    return {"probability": final, "label": label, "confidence": final}
