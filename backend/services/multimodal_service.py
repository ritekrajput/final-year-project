from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from backend.av_pipeline import (
    analyze_video_emotion,
    compute_av_score,
    extract_audio_from_video,
    extract_features,
    predict_audio_emotion,
    speech_to_text,
)
from backend.models.feature_schema import (
    AudioFeatureVector,
    VideoFeatureVector,
)
from backend.models.fusion_model import CrossAttentionFusionModel
from backend.models.text_model import TextRegressor
from backend.utils.preprocess import preprocess_text

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "models"
TEXT_CHECKPOINT = CHECKPOINT_DIR / "best_regression_model.pt"
FUSION_CHECKPOINT = CHECKPOINT_DIR / "best_multimodal_model.pt"

_text_model = None
_fusion_model = None


@dataclass
class MultimodalInferenceResult:
    text_score: float
    av_score: float
    fused_score: float
    transcript: str
    video_emotion: str
    avg_video_emotions: dict
    audio_emotion: str
    audio_confidence: float
    voice_features: dict
    audio_vector: dict
    video_vector: dict
    model_source: str

    def to_dict(self) -> dict:
        return asdict(self)


def clamp_score(score: float) -> float:
    return round(max(0.0, min(float(score), 10.0)), 2)


def get_text_model() -> TextRegressor | None:
    global _text_model
    if _text_model is not None:
        return _text_model
    if not TEXT_CHECKPOINT.exists():
        return None
    model = TextRegressor()
    model.load_state_dict(torch.load(TEXT_CHECKPOINT, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    _text_model = model
    return _text_model


def get_fusion_model() -> CrossAttentionFusionModel | None:
    global _fusion_model
    if _fusion_model is not None:
        return _fusion_model
    if not FUSION_CHECKPOINT.exists():
        return None
    text_model = get_text_model() or TextRegressor()
    model = CrossAttentionFusionModel(text_model=text_model)
    model.load_state_dict(torch.load(FUSION_CHECKPOINT, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    _fusion_model = model
    return _fusion_model


def score_text_with_model(patient_text: str, relative_text: str = "") -> float:
    text_model = get_text_model()
    inputs = preprocess_text(patient_text, relative_text, DEVICE)
    if text_model is None or inputs is None:
        from backend.inference import predict_text

        return clamp_score(predict_text(patient_text, relative_text))
    with torch.no_grad():
        score = text_model(inputs["input_ids"], inputs["attention_mask"]).item()
    return clamp_score(score)


def run_multimodal_inference(
    patient_text: str,
    relative_text: str,
    video_path: str,
    audio_path: str | None = None,
) -> MultimodalInferenceResult:
    video_file = Path(video_path)
    if audio_path:
        audio_file = Path(audio_path)
    else:
        audio_file = video_file.with_suffix(".wav")
        extract_audio_from_video(str(video_file), str(audio_file))

    transcript = speech_to_text(str(audio_file))
    voice_features = extract_features(str(audio_file))
    audio_emotion, audio_confidence = predict_audio_emotion(str(audio_file))
    video_emotion, avg_video_emotions = analyze_video_emotion(str(video_file))

    av_score = compute_av_score(
        video_emotion or "neutral",
        avg_video_emotions,
        audio_emotion,
        audio_confidence,
        voice_features,
    )
    text_score = score_text_with_model(patient_text, relative_text)

    audio_vector = AudioFeatureVector.from_pipeline(
        voice_features,
        transcript=transcript,
        audio_emotion=audio_emotion,
        confidence=audio_confidence,
    )
    video_vector = VideoFeatureVector.from_emotions(avg_video_emotions)

    fusion_model = get_fusion_model()
    inputs = preprocess_text(patient_text, relative_text, DEVICE)
    if fusion_model is None or inputs is None:
        fused_score = clamp_score((0.45 * text_score) + (0.55 * av_score))
        model_source = "heuristic_fusion"
    else:
        with torch.no_grad():
            output = fusion_model(
                inputs["input_ids"],
                inputs["attention_mask"],
                audio_vector.to_tensor(device=DEVICE).unsqueeze(0),
                video_vector.to_tensor(device=DEVICE).unsqueeze(0),
            )
            fused_score = clamp_score(output.score.item())
        model_source = "cross_attention_checkpoint"

    return MultimodalInferenceResult(
        text_score=text_score,
        av_score=clamp_score(av_score),
        fused_score=fused_score,
        transcript=transcript,
        video_emotion=video_emotion or "neutral",
        avg_video_emotions=avg_video_emotions,
        audio_emotion=audio_emotion,
        audio_confidence=audio_confidence,
        voice_features=voice_features,
        audio_vector=audio_vector.to_dict(),
        video_vector=video_vector.to_dict(),
        model_source=model_source,
    )
