"""Face detection and frame sampling for the video analysis pipeline."""
from __future__ import annotations

import subprocess
import tempfile
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision import transforms

from utils.helpers import get_device, load_config, setup_logger

logger = setup_logger("VideoDetector")

# ─────────────────────────────────────────────────────────────────────────────
# Singleton face detector
# ─────────────────────────────────────────────────────────────────────────────

_mtcnn: Optional[MTCNN] = None


def _get_mtcnn() -> MTCNN:
    global _mtcnn
    if _mtcnn is None:
        _mtcnn = MTCNN(keep_all=True, device=get_device(), post_process=False)
    return _mtcnn


# ─────────────────────────────────────────────────────────────────────────────
# Pre-processing transform
# ─────────────────────────────────────────────────────────────────────────────

def _build_transform(face_size: int, mean: List[float], std: List[float]) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((face_size, face_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Core extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_face_tensors(
    video_path: str,
    config: Optional[dict] = None,
) -> Optional[torch.Tensor]:
    """Extract faces from video, returning a tensor of shape (N, 3, H, W).

    Samples one frame every ``frame_sample_every`` frames and picks the largest
    detected face per frame.  Returns ``None`` when no faces are found.
    """
    cfg = config or load_config()
    vcfg = cfg["video"]
    sample_every: int = vcfg.get("frame_sample_every", 10)
    max_frames: int = vcfg.get("max_frames", 20)
    face_size: int = vcfg.get("face_size", 224)
    mean: List[float] = vcfg.get("imagenet_mean", [0.485, 0.456, 0.406])
    std: List[float] = vcfg.get("imagenet_std", [0.229, 0.224, 0.225])

    transform = _build_transform(face_size, mean, std)
    detector = _get_mtcnn()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return None

    face_tensors: List[torch.Tensor] = []
    raw_tensors: List[torch.Tensor] = []   # fallback when no face found
    frame_idx = 0

    while len(face_tensors) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every != 0:
            frame_idx += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Keep a raw centre-crop as fallback regardless of face detection
        if len(raw_tensors) < max_frames:
            raw_tensors.append(transform(pil_img))

        try:
            boxes, probs = detector.detect(pil_img)
        except Exception:
            frame_idx += 1
            continue

        if boxes is None or len(boxes) == 0:
            frame_idx += 1
            continue

        # Pick largest face by bounding-box area
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        best = int(np.argmax(areas))
        x1, y1, x2, y2 = [max(0, int(v)) for v in boxes[best]]
        face_crop = rgb[y1:y2, x1:x2]
        if face_crop.size == 0:
            frame_idx += 1
            continue

        face_pil = Image.fromarray(face_crop)
        face_tensors.append(transform(face_pil))
        frame_idx += 1

    cap.release()

    if not face_tensors:
        if raw_tensors:
            logger.warning(f"No faces detected in {video_path} – using raw frame crops as fallback.")
            return torch.stack(raw_tensors)
        logger.warning(f"No faces and no frames extracted from {video_path}")
        return None

    return torch.stack(face_tensors)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience alias used by the old video_processor.py shim
# ─────────────────────────────────────────────────────────────────────────────

__all__ = ["extract_face_tensors"]
