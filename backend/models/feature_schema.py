from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List

import torch

AUDIO_FEATURE_KEYS: List[str] = [
    "energy",
    "spectral_centroid",
    "mfcc",
    "speaking_rate",
    "pause_duration",
    "jitter",
    "shimmer",
    "hnr",
    "pitch",
    "formant1",
    "formant2",
    "transcript_length",
    "audio_emotion_confidence",
    "audio_sad",
    "audio_fear",
    "audio_angry",
]

VIDEO_FEATURE_KEYS: List[str] = [
    "happy",
    "sad",
    "angry",
    "fear",
    "surprise",
    "disgust",
    "neutral",
]

AUDIO_EMOTION_INDEX = {
    "sad": "audio_sad",
    "fear": "audio_fear",
    "angry": "audio_angry",
}


@dataclass
class AudioFeatureVector:
    energy: float = 0.0
    spectral_centroid: float = 0.0
    mfcc: float = 0.0
    speaking_rate: float = 0.0
    pause_duration: float = 0.0
    jitter: float = 0.0
    shimmer: float = 0.0
    hnr: float = 0.0
    pitch: float = 0.0
    formant1: float = 0.0
    formant2: float = 0.0
    transcript_length: float = 0.0
    audio_emotion_confidence: float = 0.0
    audio_sad: float = 0.0
    audio_fear: float = 0.0
    audio_angry: float = 0.0

    def to_tensor(self, device: str = "cpu") -> torch.Tensor:
        return torch.tensor(
            [getattr(self, key) for key in AUDIO_FEATURE_KEYS],
            dtype=torch.float32,
            device=device,
        )

    @classmethod
    def from_pipeline(
        cls,
        features: Dict[str, float],
        transcript: str,
        audio_emotion: str,
        confidence: float,
    ):
        payload = {key: float(features.get(key, 0.0)) for key in AUDIO_FEATURE_KEYS}
        payload["transcript_length"] = float(len((transcript or "").split()))
        payload["audio_emotion_confidence"] = float(confidence)
        emotion_key = AUDIO_EMOTION_INDEX.get((audio_emotion or "").lower())
        if emotion_key:
            payload[emotion_key] = float(confidence)
        return cls(**payload)

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class VideoFeatureVector:
    happy: float = 0.0
    sad: float = 0.0
    angry: float = 0.0
    fear: float = 0.0
    surprise: float = 0.0
    disgust: float = 0.0
    neutral: float = 0.0

    def to_tensor(self, device: str = "cpu") -> torch.Tensor:
        return torch.tensor(
            [getattr(self, key) for key in VIDEO_FEATURE_KEYS],
            dtype=torch.float32,
            device=device,
        )

    @classmethod
    def from_emotions(cls, emotion_scores: Dict[str, float] | None):
        payload = {key: 0.0 for key in VIDEO_FEATURE_KEYS}
        if emotion_scores:
            for key in VIDEO_FEATURE_KEYS:
                payload[key] = float(emotion_scores.get(key, 0.0))
        return cls(**payload)

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def batch_feature_tensors(items: Iterable[AudioFeatureVector], device: str = "cpu") -> torch.Tensor:
    return torch.stack([item.to_tensor(device=device) for item in items], dim=0)


def batch_video_tensors(items: Iterable[VideoFeatureVector], device: str = "cpu") -> torch.Tensor:
    return torch.stack([item.to_tensor(device=device) for item in items], dim=0)
