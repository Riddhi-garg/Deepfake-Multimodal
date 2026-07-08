"""
fusion.py — Multimodal Deepfake Detector (Fusion Branch)
=========================================================
Implements late decision-level fusion with configurable weights.
"""
import logging

logger = logging.getLogger("Fusion")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

# Video branch is typically more reliable than audio (60/40 split)
VIDEO_WEIGHT = 0.60
AUDIO_WEIGHT = 0.40


def fuse_predictions(
    video_prob: float | None,
    audio_prob: float | None,
    video_weight: float = VIDEO_WEIGHT,
    audio_weight: float = AUDIO_WEIGHT,
) -> dict:
    """
    Weighted late fusion of video and audio fake probabilities.

    Rules:
      • If only one modality is available, use it directly (no penalty).
      • If both are available, compute a weighted average.
      • Classification threshold: >= 0.50 → Fake.

    Returns a dict with keys:
      label       : "Real" | "Fake"
      confidence  : float in [0, 1] — distance from the classification boundary
      video_prob  : float | None
      audio_prob  : float | None
      fused_prob  : float
    """
    if video_prob is None and audio_prob is None:
        logger.warning("Both modalities returned None — prediction is Unknown.")
        return {
            "label": "Unknown",
            "confidence": 0.0,
            "video_prob": None,
            "audio_prob": None,
            "fused_prob": 0.5,
        }

    if video_prob is None:
        fused = float(audio_prob)
        logger.info(f"Fusion: audio-only mode | audio_prob={audio_prob:.4f}")
    elif audio_prob is None:
        fused = float(video_prob)
        logger.info(f"Fusion: video-only mode | video_prob={video_prob:.4f}")
    else:
        weighted = (video_weight * float(video_prob) + audio_weight * float(audio_prob)) / (video_weight + audio_weight)
        max_prob = max(float(video_prob), float(audio_prob))
        
        if max_prob >= 0.50:
            # If either modality detects a deepfake, the media is Fake.
            # We use the maximum probability to preserve the detection signal.
            fused = max_prob
        else:
            fused = weighted
        
        logger.info(
            f"Fusion: video_prob={video_prob:.4f}, audio_prob={audio_prob:.4f} "
            f"→ max_prob={max_prob:.4f}, weighted={weighted:.4f} → fused={fused:.4f}"
        )

    fused = float(min(max(fused, 0.0), 1.0))
    label = "Fake" if fused >= 0.50 else "Real"
    # Confidence = how far from the 0.5 decision boundary
    confidence = abs(fused - 0.5) * 2.0   # maps [0,1] where 1 = maximally confident

    logger.info(f"Final decision: {label} | fused_prob={fused:.4f} | confidence={confidence:.4f}")

    return {
        "label": label,
        "confidence": confidence,
        "video_prob": video_prob,
        "audio_prob": audio_prob,
        "fused_prob": fused,
    }
