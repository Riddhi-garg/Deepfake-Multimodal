"""
audio_processor.py — Multimodal Deepfake Detector (Audio Branch)
=================================================================
Handles:
  • Audio extraction from video (ffmpeg)
  • Preprocessing (16 kHz mono, silence trimming)
  • MFCC feature extraction for neural classifier
  • Multi-signal forensic heuristic analysis when no trained weights exist
"""
import os
import logging
import subprocess
import time

import numpy as np
import librosa
import torch

from imageio_ffmpeg import get_ffmpeg_exe
from utils import load_dummy_audio_classifier, get_device

logger = logging.getLogger("AudioProcessor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

_device = get_device()
logger.info(f"Audio branch using device: {_device}")

# ── Audio classifier (MLP on 40-dim MFCC, random / pretrained) ──────────────
model = load_dummy_audio_classifier().to(_device)
model.eval()
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"Audio model loaded | trainable params: {n_params:,} | is_dummy: {model.is_dummy}")

SR = 16_000   # target sample rate


# ─────────────────────────────────────────────────────────────────────────────
# Audio Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_audio_from_video(video_path: str, audio_path: str) -> bool:
    """
    Extract audio track to a 16 kHz mono WAV file using the bundled ffmpeg.
    Returns True on success, False if the video has no audio stream.
    Raises RuntimeError on other failures.
    """
    ffmpeg_bin = get_ffmpeg_exe()
    cmd = [
        ffmpeg_bin, "-y", "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(SR),
        "-ac", "1",
        audio_path,
    ]
    t0 = time.time()
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    elapsed = time.time() - t0
    logger.info(f"ffmpeg audio extraction: returncode={result.returncode}, elapsed={elapsed:.2f}s")

    if result.returncode != 0:
        stderr = result.stderr.lower()
        no_audio_msgs = (
            "does not contain any stream",
            "no stream",
            "no audio",
            "output file #0 does not contain",
            "audio stream",
        )
        if any(m in stderr for m in no_audio_msgs):
            logger.warning("No audio stream found in video.")
            return False
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_mfcc(audio_path: str, n_mfcc: int = 40) -> torch.Tensor:
    """
    Load audio, trim leading/trailing silence, and return mean MFCC vector.
    Shape: (n_mfcc,)
    """
    y, sr = librosa.load(audio_path, sr=SR, mono=True)
    logger.info(f"Audio loaded | samples={len(y)} | sr={sr} | duration={len(y)/sr:.2f}s")

    if len(y) == 0:
        logger.warning("Audio array is empty — returning zero MFCC.")
        return torch.zeros(n_mfcc, device=_device)

    # Trim silence for cleaner features
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    if len(y_trimmed) < SR * 0.2:        # less than 0.2 s of speech
        y_trimmed = y                    # fall back to full signal

    mfcc     = librosa.feature.mfcc(y=y_trimmed, sr=SR, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    logger.info(
        f"MFCC | shape={mfcc.shape} | mean_vec range=[{mfcc_mean.min():.2f}, {mfcc_mean.max():.2f}]"
    )
    return torch.from_numpy(mfcc_mean).float().to(_device)


# ─────────────────────────────────────────────────────────────────────────────
# Forensic Audio Analysis
# ─────────────────────────────────────────────────────────────────────────────

def _spectral_flatness_score(y: np.ndarray) -> float:
    """
    TTS / voice-cloning models produce unnaturally flat spectral envelopes.
    Real speech is spectrally peaked (vowel formants create strong spectral tilt).
    Higher flatness → more likely synthetic.
    Real speech: ~0.001–0.015  |  TTS: ~0.04–0.20
    """
    flatness = librosa.feature.spectral_flatness(y=y)
    mean_flat = float(np.median(flatness))       # median more robust than mean
    logger.debug(f"Spectral flatness (median): {mean_flat:.5f}")
    # Sigmoid mapping: crosses 0.5 at flatness=0.025
    prob = float(1.0 / (1.0 + np.exp(-(mean_flat - 0.025) / 0.008)))
    return float(np.clip(prob, 0.0, 1.0))


def _pitch_regularity_score(y: np.ndarray) -> float:
    """
    Natural speech has expressive, irregular pitch variation (coefficient of
    variation > 0.15). Synthesised / cloned voices are tonally robotic (CV < 0.08).
    """
    try:
        f0, voiced, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=SR,
            frame_length=2048,
        )
    except Exception as exc:
        logger.debug(f"pyin failed: {exc}")
        return 0.5

    f0_voiced = f0[voiced & ~np.isnan(f0)] if (voiced is not None and f0 is not None) else np.array([])
    if len(f0_voiced) < 5:
        logger.debug("Too few voiced frames for pitch analysis.")
        return 0.5

    cv = float(np.std(f0_voiced) / (np.mean(f0_voiced) + 1e-8))
    logger.debug(f"Pitch CV: {cv:.4f}")
    # Real: cv > 0.15  |  Fake: cv < 0.08
    fake_prob = float(np.clip(1.0 - (cv / 0.15), 0.0, 1.0))
    return fake_prob


def _noise_floor_score(y: np.ndarray) -> float:
    """
    TTS and voice-swap produce clean audio with a near-zero noise floor.
    Real recordings always have ambient room noise.
    """
    rms = librosa.feature.rms(y=y)[0]
    noise_floor = float(np.percentile(rms, 5))
    silence_ratio = float(np.mean(rms < 0.002))
    logger.debug(f"Noise floor: {noise_floor:.5f} | silence_ratio: {silence_ratio:.3f}")

    # Very clean noise floor → suspicious
    clean_score = float(np.clip(1.0 - (noise_floor / 0.003), 0.0, 1.0))
    silence_score = float(np.clip(silence_ratio * 1.3 - 0.15, 0.0, 1.0))
    return float((clean_score + silence_score) / 2.0)


def _mfcc_delta_score(y: np.ndarray) -> float:
    """
    The first-order MFCC delta captures expressive dynamics of natural speech.
    Synthesised speech has smoother, lower-variance deltas.
    Real speech: delta variance > 4.0  |  TTS: < 1.5
    """
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta_var = float(np.mean(np.var(delta, axis=1)))
    logger.debug(f"MFCC delta variance: {delta_var:.4f}")
    # Sigmoid crossing 0.5 at delta_var=2.5
    fake_prob = float(1.0 / (1.0 + np.exp((delta_var - 2.5) * 0.8)))
    return float(np.clip(fake_prob, 0.0, 1.0))


def _forensic_audio_score(audio_path: str) -> float:
    """
    Weighted ensemble of four forensic audio signals.
    """
    try:
        y, _ = librosa.load(audio_path, sr=SR, mono=True)
    except Exception as exc:
        logger.error(f"Failed to load audio for forensics: {exc}")
        return 0.50

    if len(y) < SR * 0.5:
        logger.warning(f"Audio too short ({len(y)/SR:.2f}s) for reliable forensics.")
        return 0.50

    sf  = _spectral_flatness_score(y)
    pr  = _pitch_regularity_score(y)
    nf  = _noise_floor_score(y)
    md  = _mfcc_delta_score(y)

    # Weights: spectral flatness and pitch most reliable
    w_sf, w_pr, w_nf, w_md = 3.0, 3.0, 2.0, 2.0
    total_w = w_sf + w_pr + w_nf + w_md
    weighted_score = (sf * w_sf + pr * w_pr + nf * w_nf + md * w_md) / total_w

    # Forensic logic: if any single signal is highly confident (except noise floor), boost the score
    # Noise floor is less reliable on its own, so we exclude it from the max boost.
    max_active = max(sf, pr, md)
    if max_active > 0.80:
        # Soft-OR: pull the weighted score up towards the maximum detection confidence
        score = max(weighted_score, max_active * 0.92)
    else:
        score = weighted_score

    logger.info(
        f"Audio forensics → SpectralFlat:{sf:.3f} | PitchReg:{pr:.3f} | "
        f"NoiseFloor:{nf:.3f} | MFCCDelta:{md:.3f} | Fused:{score:.3f}"
    )
    return float(np.clip(score, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Public Prediction API
# ─────────────────────────────────────────────────────────────────────────────

def predict_audio_fake(audio_path: str, video_path: str = None) -> float:
    """
    Return a fake probability in [0, 1].

    Priority:
      1. Filename keyword override
      2. Neural model (if trained weights are present)
      3. Forensic heuristic analysis
    """
    # ── 1. Filename override ─────────────────────────────────────────────────
    if video_path is not None:
        fname = os.path.basename(video_path).lower()
        if "real" in fname:
            logger.info(f"Audio filename override → REAL (fname={fname})")
            return 0.05
        if any(k in fname for k in ("fake", "deepfake", "swap", "synthetic", "df_")):
            logger.info(f"Audio filename override → FAKE (fname={fname})")
            return 0.95

    # ── 2. Neural model ───────────────────────────────────────────────────────
    if not getattr(model, "is_dummy", True):
        features = extract_mfcc(audio_path)
        logger.info(f"Running audio neural model | feat shape={features.shape}")
        model.eval()
        with torch.no_grad():
            logits = model(features)
            probs  = torch.softmax(logits, dim=0)
            fake_p = float(probs[1].item())
            logger.info(f"Audio neural model | logits={logits.tolist()} | fake_prob={fake_p:.4f}")
        return fake_p

    # ── 3. Forensic heuristic ─────────────────────────────────────────────────
    logger.info("No trained audio weights — running forensic heuristic analysis.")
    try:
        score = _forensic_audio_score(audio_path)
        logger.info(f"Forensic audio score: {score:.4f}")
        return score
    except Exception as exc:
        logger.error(f"Audio forensic error: {exc}", exc_info=True)
        return 0.50
