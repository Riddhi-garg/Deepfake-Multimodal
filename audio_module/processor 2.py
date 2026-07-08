"""Audio preprocessing: extraction, resampling, silence removal, feature computation."""
from __future__ import annotations

import subprocess
import tempfile
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

from utils.helpers import load_config, setup_logger

logger = setup_logger("AudioProcessor")


# ─────────────────────────────────────────────────────────────────────────────
# FFmpeg-based audio extraction (robust – handles QuickTime / HEVC / HEIF)
# ─────────────────────────────────────────────────────────────────────────────

def extract_audio_from_video(video_path: str, audio_path: str) -> bool:
    """Extract the audio track from *video_path* and write to *audio_path* (WAV).

    Returns:
        True  – audio was extracted successfully.
        False – video has no audio track.
    Raises:
        RuntimeError – unexpected ffmpeg failure.
    """
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg_bin = get_ffmpeg_exe()
    except Exception:
        ffmpeg_bin = "ffmpeg"

    cmd = [
        ffmpeg_bin, "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        stderr_lower = result.stderr.lower()
        no_audio_signs = [
            "does not contain any stream",
            "no stream",
            "no audio",
            "output file #0 does not contain",
        ]
        if any(s in stderr_lower for s in no_audio_signs):
            return False
        raise RuntimeError(f"FFmpeg extraction failed:\n{result.stderr}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Load & preprocess
# ─────────────────────────────────────────────────────────────────────────────

def load_and_preprocess_audio(
    audio_path: str,
    target_sr: int = 16000,
    max_seconds: int = 30,
) -> Tuple[np.ndarray, int]:
    """Load an audio file, downsample to *target_sr*, remove silence, and normalise.

    Returns:
        (waveform_array, sample_rate)
    """
    try:
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    except Exception as e:
        logger.warning(f"librosa.load failed ({e}), falling back to soundfile.")
        data, sr = sf.read(audio_path, always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        y = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Trim silence
    try:
        intervals = librosa.effects.split(y, top_db=30)
        y = np.concatenate([y[s:e] for s, e in intervals]) if len(intervals) > 0 else y
    except Exception:
        pass

    # Truncate to max_seconds
    max_samples = max_seconds * sr
    y = y[:max_samples]

    # Amplitude normalisation
    max_amp = np.abs(y).max()
    if max_amp > 1e-6:
        y = y / max_amp

    return y.astype(np.float32), sr


# ─────────────────────────────────────────────────────────────────────────────
# Hand-crafted feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def compute_handcrafted_features(y: np.ndarray, sr: int, n_mfcc: int = 40) -> np.ndarray:
    """Compute and concatenate all handcrafted audio features.

    Features (all mean-aggregated over time):
        MFCC  (n_mfcc dims)
        Mel Spectrogram (128 dims)
        Chroma (12 dims)
        Spectral Contrast (7 dims)
        Zero Crossing Rate (1 dim)

    Returns:
        np.ndarray of shape (n_mfcc + 128 + 12 + 7 + 1,)
    """
    if len(y) == 0:
        return np.zeros(n_mfcc + 128 + 12 + 7 + 1, dtype=np.float32)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = mfcc.mean(axis=1)

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_mean = mel.mean(axis=1)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)

    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = contrast.mean(axis=1)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=y)
    zcr_mean = zcr.mean(axis=1)

    return np.concatenate([mfcc_mean, mel_mean, chroma_mean, contrast_mean, zcr_mean]).astype(np.float32)


def get_handcrafted_feature_dim(n_mfcc: int = 40) -> int:
    return n_mfcc + 128 + 12 + 7 + 1
