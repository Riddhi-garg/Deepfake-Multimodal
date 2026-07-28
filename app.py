"""
app.py — Multimodal Deepfake Detector
======================================
Streamlit UI integrating video and audio forensic analysis pipelines.
"""
import os
import logging
import time
import tempfile
from pathlib import Path

import torch
import streamlit as st

from video_processor import extract_face_tensors, predict_video_fake, model as video_model
from audio_processor import extract_audio_from_video, predict_audio_fake, model as audio_model
from fusion import fuse_predictions

logger = logging.getLogger("App")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Multimodal Deepfake Detector",
    page_icon="🎭",
    layout="wide",
)

# Background color
st.markdown("""
<style>
.stApp {
    background: linear-gradient(
        135deg,
        #000000 0%,
        #1A1A1A 30%,
        #5C1A3A 70%,
        #E75480 100%
    );
}
</style>
""", unsafe_allow_html=True)

st.title("🎭 Multimodal Deepfake Detection System")
st.markdown(
    "Upload a video and the system analyses both **visual face forensics** and "
    "**audio speech forensics** to determine whether the media is real or AI-generated."
)

# Default settings (sidebar removed)
video_weight = 0.60
audio_weight = 0.40
show_debug = False

# ─────────────────────────────────────────────────────────────────────────────
# File upload
# ─────────────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Choose a video file (MP4, AVI, MOV, MKV)",
    type=["mp4", "avi", "mov", "mkv"],
)

if uploaded_file is None:
    st.info("👆 Upload a video to begin analysis.")
    st.stop()

# Save to temp file
with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
    tmp.write(uploaded_file.getbuffer())
    video_path = tmp.name

st.success(f"Video received: `{uploaded_file.name}` ({uploaded_file.size / 1e6:.1f} MB)")

t_start = time.time()

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Visual Analysis
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🎞️ Step 1 · Visual Analysis")

video_prob   = None
raw_faces    = []
face_batch   = None

with st.spinner("Extracting faces from video frames…"):
    try:
        face_batch, raw_faces = extract_face_tensors(video_path, max_frames=20)
    except Exception as exc:
        st.error(f"Face extraction failed: {exc}")
        logger.error("Face extraction exception", exc_info=True)

if face_batch is None or len(raw_faces) == 0:
    st.warning(
        "⚠️ No faces detected in the video.  \n"
        "Visual branch will be excluded from the final decision."
    )
else:
    n_faces = len(raw_faces)
    st.success(f"✅ {n_faces} face crop(s) extracted successfully.")

    # Show a sample of face crops
# Show a sample of face crops
cols = st.columns(min(n_faces, 6))

for i, (col, face) in enumerate(zip(cols, raw_faces[:6])):
    col.image(
        face,
        caption=f"Frame {i+1}",
        width="stretch"
    )

# Run visual analysis ONLY ONCE
with st.spinner("Running visual forensic analysis…"):
    try:
        video_prob = predict_video_fake(
            face_batch,
            video_path=video_path,
            raw_faces=raw_faces
        )
    except Exception as exc:
        st.error(f"Visual analysis error: {exc}")
        logger.error(
            "Visual analysis exception",
            exc_info=True
        )
        video_prob = None

# Display result ONLY ONCE
if video_prob is not None:
    v_label = "🔴 Fake" if video_prob >= 0.5 else "🟢 Real"

    st.metric(
        "Visual Fake Probability",
        f"{video_prob:.1%}"
    )

    st.progress(
        float(video_prob),
        text=f"Video: {v_label} ({video_prob:.1%})"
    )

    if show_debug:
        with st.expander("🔍 Visual Debug Info"):
            st.write(f"- Face crops analysed: `{n_faces}`")
            st.write(f"- Tensor shape: `{tuple(face_batch.shape)}`")
            st.write(
                f"- Model is_dummy: "
                f"`{getattr(video_model, 'is_dummy', True)}`"
            )
            st.write(
                f"- Raw fake probability: `{video_prob:.6f}`"
            )

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Acoustic Analysis
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🎙️ Step 2 · Acoustic Analysis")

audio_prob = None

with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
    audio_path = tmp_audio.name

try:
    with st.spinner("Extracting audio track…"):
        has_audio = extract_audio_from_video(video_path, audio_path)

    if not has_audio:
        st.warning(
            "⚠️ No audio track found in the video.  \n"
            "Acoustic branch will be excluded from the final decision."
        )
    else:
        audio_size = os.path.getsize(audio_path)
        st.success(f"✅ Audio extracted ({audio_size / 1e3:.0f} KB, 16 kHz mono WAV).")

        with st.spinner("Running acoustic forensic analysis…"):
            try:
                audio_prob = predict_audio_fake(audio_path, video_path=video_path)
            except Exception as exc:
                st.error(f"Acoustic analysis error: {exc}")
                logger.error("Acoustic analysis exception", exc_info=True)
                audio_prob = None

        if audio_prob is not None:
            a_label = "🔴 Fake" if audio_prob >= 0.5 else "🟢 Real"
            st.metric("Acoustic Fake Probability", f"{audio_prob:.1%}", delta=None)
            st.progress(float(audio_prob), text=f"Audio: {a_label}  ({audio_prob:.1%})")

            if show_debug:
                with st.expander("🔍 Acoustic Debug Info"):
                    st.write(f"- Audio path: `{audio_path}`")
                    st.write(f"- Audio size: `{audio_size:,}` bytes")
                    st.write(f"- Model is_dummy: `{getattr(audio_model, 'is_dummy', True)}`")
                    st.write(f"- Raw fake probability: `{audio_prob:.6f}`")

except Exception as exc:
    st.error(f"Audio pipeline error: {exc}")
    logger.error("Audio pipeline exception", exc_info=True)

finally:
    if os.path.exists(audio_path):
        os.remove(audio_path)

# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Fusion & Final Decision
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🧩 Step 3 · Multimodal Fusion & Final Decision")

result = fuse_predictions(video_prob, audio_prob, video_weight, audio_weight)
fused_prob  = result["fused_prob"]
label       = result["label"]
confidence  = result["confidence"]

t_elapsed = time.time() - t_start

# Large result card
if label == "Fake":
    st.error(f"## 🔴 FAKE  —  {fused_prob:.1%} fake probability")
elif label == "Real":
    st.success(f"## 🟢 REAL  —  {fused_prob:.1%} fake probability")
else:
    st.warning("## ❓ UNKNOWN — not enough data to classify")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Fused Probability",   f"{fused_prob:.1%}")
col2.metric("Decision Confidence", f"{confidence:.1%}")
col3.metric("Video Probability",   f"{video_prob:.1%}" if video_prob is not None else "N/A")
col4.metric("Audio Probability",   f"{audio_prob:.1%}" if audio_prob is not None else "N/A")

st.progress(float(fused_prob), text=f"Fused fake score: {fused_prob:.1%}")
st.caption(f"⏱️ Total processing time: {t_elapsed:.2f} seconds")

if show_debug:
    with st.expander("🔍 Fusion Debug Info"):
        st.json({
            "video_prob":    round(video_prob, 6) if video_prob is not None else None,
            "audio_prob":    round(audio_prob, 6) if audio_prob is not None else None,
            "video_weight":  round(video_weight, 2),
            "audio_weight":  round(audio_weight, 2),
            "fused_prob":    round(fused_prob, 6),
            "label":         label,
            "confidence":    round(confidence, 6),
            "elapsed_s":     round(t_elapsed, 2),
        })

# ─────────────────────────────────────────────────────────────────────────────
# Video playback
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📽️ Uploaded Video")
st.video(video_path)

# Cleanup
try:
    os.remove(video_path)
except Exception:
    pass
