"""
av_pipeline.py
Audio-Video analysis pipeline for depression assessment.
Records video + audio, extracts features, and returns a depression risk score.
"""

import cv2
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
import numpy as np
import librosa
import torch
import time
import os
import av as pyav          # PyAV — uses libav C bindings, no ffmpeg CLI needed

from faster_whisper import WhisperModel
from safetensors.torch import load_file as load_safetensors
from transformers import AutoConfig, AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
from deepface import DeepFace

# Optional: Praat/Parselmouth for advanced voice analysis
try:
    import parselmouth
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False
    print("[AV Pipeline] WARNING: parselmouth not installed. Voice feature extraction will be limited.")


# ─────────────────────────────────────────────
# MODEL LOADING (done once at import time)
# ─────────────────────────────────────────────

def _resolve_cached_model_path(model_cache_dir: str, snapshot_id: str, fallback_name: str):
    snapshot_path = os.path.join(
        os.path.expanduser("~"),
        ".cache",
        "huggingface",
        "hub",
        model_cache_dir,
        "snapshots",
        snapshot_id,
    )
    if os.path.isdir(snapshot_path):
        return snapshot_path
    return fallback_name


print("[AV Pipeline] Loading Whisper model...")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
_WHISPER_MODEL_PATH = _resolve_cached_model_path(
    "models--Systran--faster-whisper-base",
    "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66",
    "base",
)
whisper_model = WhisperModel(
    _WHISPER_MODEL_PATH,
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE_TYPE,
)

print("[AV Pipeline] Loading audio emotion model...")
_EMOTION_MODEL_NAME = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
_AUDIO_MODEL_PATH = _resolve_cached_model_path(
    "models--ehcalabres--wav2vec2-lg-xlsr-en-speech-emotion-recognition",
    "b520c9c46a719e36e1b9a91cad2cb5d0668757d8",
    _EMOTION_MODEL_NAME,
)
feature_extractor = AutoFeatureExtractor.from_pretrained(
    _AUDIO_MODEL_PATH,
    local_files_only=True,
)


def _load_audio_emotion_model(model_name: str):
    config = AutoConfig.from_pretrained(_AUDIO_MODEL_PATH, local_files_only=True)
    if hasattr(config, "classifier_proj_size"):
        config.classifier_proj_size = config.hidden_size
    model = Wav2Vec2ForSequenceClassification(config)

    checkpoint_path = os.path.join(_AUDIO_MODEL_PATH, "model.safetensors")
    state_dict = load_safetensors(checkpoint_path)

    key_map = {
        "classifier.dense.weight": "projector.weight",
        "classifier.dense.bias": "projector.bias",
        "classifier.output.weight": "classifier.weight",
        "classifier.output.bias": "classifier.bias",
    }
    for old_key, new_key in key_map.items():
        if old_key in state_dict:
            state_dict[new_key] = state_dict.pop(old_key)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"[AV Pipeline] Warning: unexpected audio checkpoint keys: {unexpected}")
    if missing:
        print(f"[AV Pipeline] Warning: missing audio checkpoint keys: {missing}")

    model.eval()
    return model


emotion_model = _load_audio_emotion_model(_EMOTION_MODEL_NAME)

print("[AV Pipeline] All models loaded.")


# ─────────────────────────────────────────────
# NEGATIVE EMOTION WEIGHTS  (used for scoring)
# ─────────────────────────────────────────────

# Higher value → contributes more to depression risk score
EMOTION_WEIGHTS = {
    "sad":      1.0,
    "angry":    0.7,
    "fear":     0.8,
    "disgust":  0.6,
    "neutral":  0.3,
    "surprise": 0.1,
    "happy":    0.0,
}

AUDIO_EMOTION_WEIGHTS = {
    "sad":      1.0,
    "angry":    0.7,
    "fear":     0.8,
    "disgust":  0.6,
    "neutral":  0.3,
    "surprise": 0.1,
    "happy":    0.0,
}


# ─────────────────────────────────────────────
# RECORD VIDEO + AUDIO
# ─────────────────────────────────────────────

def record_video_audio(duration: int = 60,
                        video_path: str = "av_output/video.avi",
                        audio_path: str = "av_output/audio.wav") -> None:
    """Record webcam video and microphone audio simultaneously."""

    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    fs = 44100
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1)

    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))

    start = time.time()
    frame_count = 0
    current_emotion = "Detecting..."

    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % 10 == 0:
            try:
                result = DeepFace.analyze(
                    frame, actions=["emotion"], enforce_detection=False
                )
                emotion_data = result[0]["emotion"]
                current_emotion = max(emotion_data, key=emotion_data.get)

                region = result[0]["region"]
                x, y, w, h = region["x"], region["y"], region["w"], region["h"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, current_emotion, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception:
                pass

        # Countdown overlay
        elapsed = int(time.time() - start)
        remaining = duration - elapsed
        cv2.putText(frame, f"Recording: {remaining}s left", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        out.write(frame)
        cv2.imshow("AV Assessment - Press Q to stop early", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    sd.wait()
    wav_write(audio_path, fs, audio_data)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

TARGET_SR = 16000   # Whisper, librosa, and wav2vec2 all expect 16 kHz
VIDEO_EMOTION_SAMPLE_EVERY = int(os.environ.get("VIDEO_EMOTION_SAMPLE_EVERY", "60"))
VIDEO_EMOTION_MAX_WIDTH = int(os.environ.get("VIDEO_EMOTION_MAX_WIDTH", "160"))

def extract_audio_from_video(video_path: str, audio_path: str) -> None:
    """
    Extract audio from a video file and write a 16 kHz mono WAV.

    Uses PyAV (libav C bindings — no ffmpeg CLI needed).
    Output matches what sounddevice produces in the working reference pipeline,
    so Whisper, librosa, and wav2vec2 all work correctly.
    """
    os.makedirs(os.path.dirname(audio_path) or ".", exist_ok=True)

    container = pyav.open(video_path)
    audio_stream = next(
        (s for s in container.streams if s.type == "audio"), None
    )
    if audio_stream is None:
        raise ValueError(f"No audio stream found in {video_path}")

    # Resample to 16 kHz mono s16 in one step — identical format to
    # scipy.io.wavfile.write(int16) used in the working reference code.
    resampler = pyav.AudioResampler(
        format="s16",
        layout="mono",
        rate=TARGET_SR,
    )

    pcm_chunks: list[np.ndarray] = []

    for packet in container.demux(audio_stream):
        for frame in packet.decode():
            for rf in resampler.resample(frame):
                # s16 mono → shape (1, N); flatten to 1-D int16
                pcm_chunks.append(rf.to_ndarray().flatten())

    # Flush resampler
    for rf in resampler.resample(None):
        pcm_chunks.append(rf.to_ndarray().flatten())

    container.close()

    if not pcm_chunks:
        raise RuntimeError(
            f"Audio extraction produced no samples from {video_path}. "
            "Check that the file contains an audio track."
        )

    audio_data = np.concatenate(pcm_chunks)   # int16, 1-D, 16 kHz
    wav_write(audio_path, TARGET_SR, audio_data)
    print(f"[AV Pipeline] Audio extracted -> {audio_path} "
          f"({len(audio_data) / TARGET_SR:.1f}s @ {TARGET_SR}Hz mono)")


def run_av_pipeline_on_file(file_path: str):
    """Run full AV analysis on an existing video file (no recording needed)."""
    video_path = file_path
    audio_path = "temp/audio.wav"

    os.makedirs("temp", exist_ok=True)
    extract_audio_from_video(video_path, audio_path)   # ← PyAV, no ffmpeg CLI

    video_emotion, avg_video_emotions = analyze_video_emotion(video_path)
    transcript = speech_to_text(audio_path)
    features = extract_features(audio_path)
    audio_emotion, audio_confidence = predict_audio_emotion(audio_path)

    score = compute_av_score(
        video_emotion or "neutral",
        avg_video_emotions,
        audio_emotion,
        audio_confidence,
        features,
    )

    level = (
        "Low" if score <= 3 else
        "Moderate" if score <= 6 else
        "High" if score <= 8 else
        "Critical"
    )

    return {
        "score": score,
        "level": level,
        "transcript": transcript,
        "video_emotion": video_emotion,
        "audio_emotion": audio_emotion,
        "audio_confidence": audio_confidence,
    }


# ─────────────────────────────────────────────
# VIDEO EMOTION ANALYSIS
# ─────────────────────────────────────────────

def analyze_video_emotion(video_path: str):
    """Returns (dominant_emotion, avg_emotion_dict) from a recorded video."""

    cap = cv2.VideoCapture(video_path)
    emotions_sum = {k: 0.0 for k in EMOTION_WEIGHTS}
    frame_count = 0
    sample_every = max(1, VIDEO_EMOTION_SAMPLE_EVERY)

    processed = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed += 1
        if processed % sample_every != 0:
            continue

        try:
            height, width = frame.shape[:2]
            if width > VIDEO_EMOTION_MAX_WIDTH:
                scale = VIDEO_EMOTION_MAX_WIDTH / float(width)
                frame = cv2.resize(frame, (VIDEO_EMOTION_MAX_WIDTH, int(height * scale)))
            result = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="opencv",
            )
            emotion_data = result[0]["emotion"]
            for e in emotions_sum:
                emotions_sum[e] += emotion_data.get(e, 0)
            frame_count += 1
        except Exception:
            pass

    cap.release()

    if frame_count == 0:
        return None, {}

    avg_emotions = {k: round(v / frame_count, 4) for k, v in emotions_sum.items()}
    dominant_emotion = max(avg_emotions, key=avg_emotions.get)
    return dominant_emotion, avg_emotions,


# ─────────────────────────────────────────────
# SPEECH TO TEXT
# ─────────────────────────────────────────────

def speech_to_text(audio_path: str) -> str:
    segments, _ = whisper_model.transcribe(
        audio_path,
        beam_size=1,
        best_of=1,
        vad_filter=True,
        temperature=0.0,
    )
    return "".join(seg.text for seg in segments).strip()


# ─────────────────────────────────────────────
# VOICE FEATURE EXTRACTION
# ─────────────────────────────────────────────

def extract_features(audio_path: str) -> dict:
    y, sr = librosa.load(audio_path)
    
    energy = float(np.mean(librosa.feature.rms(y=y)))
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    mfcc = float(np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)))
    
    duration = librosa.get_duration(y=y, sr=sr)
    segments = librosa.effects.split(y)
    speaking_rate = len(segments) / duration if duration > 0 else 0
    pause_duration = duration - np.sum([(e - s) / sr for s, e in segments])

    features = {
        "energy":           round(float(energy), 6),
        "spectral_centroid":round(float(spectral_centroid), 4),
        "mfcc":             round(float(mfcc), 4),
        "speaking_rate":    round(float(speaking_rate), 4),
        "pause_duration":   round(float(pause_duration), 4),
    }
    
    # Parselmouth-based features (if available)
    if PARSELMOUTH_AVAILABLE:
        try:
            sound = parselmouth.Sound(audio_path)

            pitch = sound.to_pitch()
            mean_pitch = parselmouth.praat.call(pitch, "Get mean", 0, 0, "Hertz")

            point_process = parselmouth.praat.call(
                sound, "To PointProcess (periodic, cc)", 75, 500
            )
            jitter = parselmouth.praat.call(
                point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
            )
            shimmer = parselmouth.praat.call(
                [sound, point_process],
                "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
            )
            hnr_obj = parselmouth.praat.call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr_value = parselmouth.praat.call(hnr_obj, "Get mean", 0, 0)

            formant = sound.to_formant_burg()
            f1 = formant.get_value_at_time(1, 0.5)
            f2 = formant.get_value_at_time(2, 0.5)

            features.update({
                "jitter":           round(float(jitter), 6),
                "shimmer":          round(float(shimmer), 6),
                "hnr":              round(float(hnr_value), 4),
                "pitch":            round(float(mean_pitch), 4),
                "formant1":         round(float(f1), 4) if f1 else 0.0,
                "formant2":         round(float(f2), 4) if f2 else 0.0,
            })
        except Exception as e:
            print(f"[AV Pipeline] Warning: Parselmouth analysis failed: {e}")
    else:
        # Fallback features when parselmouth is unavailable
        features.update({
            "jitter":           0.0,
            "shimmer":          0.0,
            "hnr":              0.0,
            "pitch":            0.0,
            "formant1":         0.0,
            "formant2":         0.0,
        })

    return features


# ─────────────────────────────────────────────
# AUDIO EMOTION PREDICTION
# ─────────────────────────────────────────────

def predict_audio_emotion(audio_path: str):
    audio, _ = librosa.load(audio_path, sr=16000)
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        logits = emotion_model(**inputs).logits

    probs = torch.nn.functional.softmax(logits, dim=1)
    predicted_id = torch.argmax(probs).item()
    emotion = emotion_model.config.id2label[predicted_id]
    emotion = {
        "fearful": "fear",
        "surprised": "surprise",
        "calm": "neutral",
    }.get(emotion.lower(), emotion.lower())
    confidence = round(float(probs[0][predicted_id].item()), 4)

    return emotion, confidence


# ─────────────────────────────────────────────
# DEPRESSION RISK SCORER
# ─────────────────────────────────────────────

def compute_av_score(
    video_emotion: str,
    avg_video_emotions: dict,
    audio_emotion: str,
    audio_confidence: float,
    features: dict,
) -> float:
    """
    Heuristic depression risk score (0–10) from AV signals.

    Components
    ----------
    1. Video emotion score   — weighted sum of avg facial emotion percentages
    2. Audio emotion score   — emotion label mapped to risk weight × confidence
    3. Voice feature score   — elevated jitter/shimmer, low HNR, slow speaking rate
    """

    # 1. Video emotion score (0–10)
    video_score = sum(
        EMOTION_WEIGHTS.get(e, 0) * v
        for e, v in avg_video_emotions.items()
    )
    # avg_video_emotions values are percentages (0–100); normalise to 0-10
    video_score = min(video_score / 10, 10)

    # 2. Audio emotion score (0–10)
    audio_risk_weight = AUDIO_EMOTION_WEIGHTS.get(audio_emotion.lower(), 0.3)
    audio_score = audio_risk_weight * audio_confidence * 10

    # 3. Voice feature score (0–10)
    # Thresholds based on clinical literature
    feature_score = 0.0
    if features["jitter"] > 0.01:          # abnormal jitter > 1%
        feature_score += 2.0
    if features["shimmer"] > 0.05:         # abnormal shimmer > 5%
        feature_score += 2.0
    if features["hnr"] < 15:               # low HNR → noisy voice
        feature_score += 2.0
    if features["speaking_rate"] < 2:      # slow speech
        feature_score += 2.0
    if features["pause_duration"] > 10:    # long silences
        feature_score += 2.0
    feature_score = min(feature_score, 10)

    # Weighted combination
    final_score = (
        0.40 * video_score +
        0.35 * audio_score +
        0.25 * feature_score
    )

    return round(min(max(final_score, 0), 10), 2)


# ─────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────

def run_av_pipeline(
    duration: int = 60,
    video_path: str = "av_output/video.avi",
    audio_path: str = "av_output/audio.wav",
) -> dict:
    """
    Run the complete AV pipeline.
    Returns a dict with score, level, transcript, emotions, and voice features.
    """

    record_video_audio(duration, video_path, audio_path)

    video_emotion, avg_video_emotions = analyze_video_emotion(video_path)
    transcript = speech_to_text(audio_path)
    features = extract_features(audio_path)
    audio_emotion, audio_confidence = predict_audio_emotion(audio_path)

    score = compute_av_score(
        video_emotion or "neutral",
        avg_video_emotions,
        audio_emotion,
        audio_confidence,
        features,
    )

    level = (
        "Low" if score <= 3 else
        "Moderate" if score <= 6 else
        "High" if score <= 8 else
        "Critical"
    )

    return {
        "score":             score,
        "level":             level,
        "transcript":        transcript,
        "video_emotion":     video_emotion,
        "avg_video_emotions": avg_video_emotions,
        "audio_emotion":     audio_emotion,
        "audio_confidence":  audio_confidence,
        "voice_features":    features,
    }
