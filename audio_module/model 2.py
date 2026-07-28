"""Audio model: Wav2Vec2 feature extractor + handcrafted features → MLP classifier."""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from audio_module.processor import compute_handcrafted_features, get_handcrafted_feature_dim
from utils.helpers import get_device, load_config, setup_logger

logger = setup_logger("AudioModel")

# Feature extractor (Wav2Vec2 + handcrafted)


class AudioFeatureExtractor:
    """Combines Wav2Vec2 embeddings with handcrafted features.

    If the Wav2Vec2 model cannot be loaded (e.g. offline), it falls back to
    handcrafted-only features.
    """

    def __init__(self, device: Optional[torch.device] = None, model_id: str = "facebook/wav2vec2-base-960h"):
        self.device = device or get_device()
        self.wav2vec = None
        self.processor = None
        self._load_wav2vec(model_id)

    def _load_wav2vec(self, model_id: str) -> None:
        try:
            from transformers import Wav2Vec2Model, Wav2Vec2Processor
            logger.info(f"Loading Wav2Vec2: {model_id}")
            self.processor = Wav2Vec2Processor.from_pretrained(model_id)
            self.wav2vec = Wav2Vec2Model.from_pretrained(model_id).to(self.device)
            self.wav2vec.eval()
            logger.info("Wav2Vec2 loaded successfully.")
        except Exception as e:
            logger.warning(f"Wav2Vec2 unavailable ({e}). Using handcrafted features only.")

    @torch.no_grad()
    def extract(self, y: np.ndarray, sr: int) -> torch.Tensor:
        """Return a single feature vector concatenating Wav2Vec2 + handcrafted."""
        # Handcrafted features
        hc = compute_handcrafted_features(y, sr)
        hc_tensor = torch.from_numpy(hc).float()

        # Wav2Vec2 embeddings
        if self.wav2vec is not None and len(y) > 0:
            try:
                inputs = self.processor(
                    y, sampling_rate=sr, return_tensors="pt", padding=True
                )
                input_values = inputs["input_values"].to(self.device)
                outputs = self.wav2vec(input_values)
                # Mean-pool over time → (768,)
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu()
                return torch.cat([hc_tensor, embeddings])
            except Exception as e:
                logger.warning(f"Wav2Vec2 inference error ({e}). Using handcrafted only.")

        return hc_tensor


# Audio classifier model


class DeepfakeAudioClassifier(nn.Module):
    """MLP that classifies combined audio features as Real (0) or Fake (1)."""

    def __init__(self, handcrafted_dim: int = 188, wav2vec_dim: int = 768):
        super().__init__()
        in_dim = handcrafted_dim + wav2vec_dim
        # Also handle handcrafted-only mode
        self._in_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle both handcrafted-only and full feature vectors
        if x.shape[-1] != self._in_dim:
            # Pad / truncate to expected dimension
            if x.shape[-1] < self._in_dim:
                pad = torch.zeros(*x.shape[:-1], self._in_dim - x.shape[-1], device=x.device)
                x = torch.cat([x, pad], dim=-1)
            else:
                x = x[..., :self._in_dim]
        return self.net(x)


def build_audio_model(config: Optional[Dict] = None) -> DeepfakeAudioClassifier:
    cfg = config or load_config()
    hc_dim = get_handcrafted_feature_dim(cfg["audio"].get("n_mfcc", 40))
    return DeepfakeAudioClassifier(handcrafted_dim=hc_dim)


# Heuristic fallback (signal-processing based)


def predict_audio_fake_heuristic(y: np.ndarray, sr: int) -> float:
    """Spectral analysis based fake-probability estimate.

    Analyses:
    - Spectral flatness (GAN speech → unnaturally flat spectrum)
    - Pitch continuity (synthetic speech → erratic pitch)
    - Spectral roll-off (over-compressed deepfakes → reduced high-freq content)

    Returns a value in [0.05, 0.95].
    """
    import librosa as _lb

    if len(y) < 512:
        return 0.5

    scores = []

    try:
        # 1. Spectral flatness – high flatness → noise-like (often fake)
        flatness = _lb.feature.spectral_flatness(y=y)
        flat_mean = float(flatness.mean())
        scores.append(min(flat_mean * 50, 1.0))  # normalise to [0,1]
    except Exception:
        pass

    try:
        # 2. Pitch continuity – erratic f0 jumps → synthetic
        f0, voiced, _ = _lb.pyin(y, fmin=50, fmax=500, sr=sr)
        f0_valid = f0[voiced & ~np.isnan(f0)]
        if len(f0_valid) > 1:
            continuity = float(np.abs(np.diff(f0_valid)).mean())
            scores.append(min(continuity / 100.0, 1.0))
    except Exception:
        pass

    try:
        # 3. Spectral roll-off (low roll-off → high-freq absent → possibly fake)
        rolloff = _lb.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = float(rolloff.mean()) / (sr / 2)
        low_rolloff_score = 1.0 - rolloff_mean
        scores.append(low_rolloff_score)
    except Exception:
        pass

    if not scores:
        return 0.5

    raw = float(sum(scores) / len(scores))
    return max(0.05, min(0.95, raw))
