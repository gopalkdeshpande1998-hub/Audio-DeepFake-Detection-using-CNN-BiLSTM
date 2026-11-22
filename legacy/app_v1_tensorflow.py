import io
import logging
import os
from pathlib import Path

# Quiet TensorFlow before it is imported: hide C++ INFO notices (oneDNN, CPU
# instruction set) and the tf.compat.v1 deprecation warnings that Keras 2.15
# raises from its own internals. Real errors are still shown.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
import tensorflow as tf  # noqa: E402

# Feature settings: must match the training notebook (Model_Training.ipynb).
SAMPLE_RATE = 16000
N_MFCC = 40
MAX_FRAMES = 500
HOP_LENGTH = 512  # librosa default, used during training
MAX_SECONDS = MAX_FRAMES * HOP_LENGTH / SAMPLE_RATE  # ~16 s analysed per clip

# Input sanity limits
MIN_SECONDS = 0.5
SILENCE_PEAK = 1e-4

MODEL_PATH = Path(__file__).resolve().parent / "savedmodels" / "updated_model.keras"

# Map uploaded file extensions to MIME types for the audio player
AUDIO_MIME = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "ogg": "audio/ogg",
}


def load_audio(audio_bytes, sr=SAMPLE_RATE):
    """Decode audio bytes into a mono float waveform at the given sample rate."""
    audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr)
    return audio


def extract_features(audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC, max_length=MAX_FRAMES):
    """Compute MFCCs, pad or trim to a fixed frame count, and shape for the model."""
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    if mfccs.shape[1] < max_length:
        mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode="constant")
    else:
        mfccs = mfccs[:, :max_length]
    return mfccs.reshape(1, n_mfcc, max_length, 1)


def validate_audio(audio, sr=SAMPLE_RATE):
    """Return (errors, warnings) describing problems with the decoded audio."""
    errors, warnings = [], []
    duration = len(audio) / sr
    if duration < MIN_SECONDS:
        errors.append(f"Audio is too short ({duration:.2f} s). Please upload at least {MIN_SECONDS} s.")
    elif np.max(np.abs(audio)) < SILENCE_PEAK:
        errors.append("Audio appears to be silent.")
    if duration > MAX_SECONDS:
        warnings.append(
            f"Audio is {duration:.1f} s long; only the first {MAX_SECONDS:.0f} s are analysed."
        )
    return errors, warnings


@st.cache_resource
def load_model(path=MODEL_PATH):
    try:
        # compile=False: inference only, so skip restoring optimizer state
        # (avoids the "Skipping variable loading for optimizer" warning).
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        st.error(f"Error loading model from {path}: {e}")
        return None


def main():
    st.title("Audio Deepfake Detection")
    st.write("Upload an audio file to detect if it is real or deepfake")

    model = load_model()

    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=list(AUDIO_MIME),
        help="Upload an audio file to check if it's a deepfake",
    )

    if uploaded_file is not None:
        ext = Path(uploaded_file.name).suffix.lower().lstrip(".")
        st.audio(uploaded_file, format=AUDIO_MIME.get(ext, uploaded_file.type))

        if st.button("Detect Deepfake"):
            if model is None:
                st.error("Model could not be loaded. Please check the model file.")
                return

            try:
                audio = load_audio(uploaded_file.getvalue())
            except Exception as e:
                st.error(f"Error decoding audio: {e}")
                return

            errors, warnings = validate_audio(audio)
            for msg in errors:
                st.error(msg)
            if errors:
                return
            for msg in warnings:
                st.warning(msg)

            features = extract_features(audio)
            fake_score = float(model.predict(features, verbose=0)[0][0])
            is_deepfake = fake_score > 0.5

            st.subheader("Detection Results")
            if is_deepfake:
                st.error(f"🚨 Deepfake Detected (model score: {fake_score * 100:.2f}%)")
            else:
                st.success(f"✅ Real Audio (model score: {(1 - fake_score) * 100:.2f}%)")

            st.progress(fake_score, text=f"Deepfake score: {fake_score:.3f}")
            st.caption(
                "The score is the model's raw output, not a calibrated probability. "
                "The model was trained on one English speaker (LJSpeech) and one vocoder "
                "(MelGAN), so results on other voices or generators may be unreliable."
            )

    st.sidebar.header("About")
    st.sidebar.info(
        "This app uses a deep learning model to detect audio deepfakes. "
        "It analyzes the audio file's MFCC features to classify if the audio is real or synthetic."
    )

    st.sidebar.header("How it Works")
    st.sidebar.write(
        "1. Upload an audio file\n"
        "2. Click 'Detect Deepfake'\n"
        "3. Get the result with the model's score"
    )


if __name__ == "__main__":
    main()
