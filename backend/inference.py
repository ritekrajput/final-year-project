from pathlib import Path

import torch

from backend.utils.preprocess import preprocess_text

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "best_regression_model.pt"

_model = None
_model_error = None

NEGATIVE_WEIGHTS = {
    "depressed": 2.0,
    "hopeless": 2.0,
    "sad": 1.2,
    "empty": 1.3,
    "anxious": 1.0,
    "panic": 1.6,
    "worthless": 1.8,
    "tired": 0.7,
    "alone": 1.0,
    "cry": 1.0,
    "suicide": 3.0,
    "kill myself": 4.0,
    "end my life": 4.0,
    "not worth living": 4.0,
}

POSITIVE_WEIGHTS = {
    "better": 0.8,
    "improving": 1.0,
    "hopeful": 1.2,
    "supported": 0.8,
    "calm": 0.5,
    "stable": 0.7,
    "good": 0.4,
}


def _clamp(score: float) -> float:
    return max(1.0, min(10.0, round(score, 2)))


def _estimate_with_keywords(patient_text: str, relative_text: str) -> float:
    text = f"{patient_text} {relative_text}".lower()
    score = 3.5

    for phrase, weight in NEGATIVE_WEIGHTS.items():
        if phrase in text:
            score += weight

    for phrase, weight in POSITIVE_WEIGHTS.items():
        if phrase in text:
            score -= weight

    if len(text.split()) > 80:
        score += 0.3

    return _clamp(score)


def load_model():
    global _model, _model_error

    if _model is not None:
        return _model
    if _model_error is not None:
        return None
    if not MODEL_PATH.exists():
        _model_error = f"Missing model weights: {MODEL_PATH}"
        return None

    try:
        from backend.models.text_model import TextRegressor

        model = TextRegressor()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        _model = model
        return _model
    except Exception as exc:
        _model_error = str(exc)
        return None


def predict(input_ids, attention_mask):
    model = load_model()
    if model is None:
        return 5.0

    with torch.no_grad():
        output = model(input_ids, attention_mask)

    severity = float(output.cpu().item())
    return _clamp(severity)


def predict_text(patient_text: str, relative_text: str = ""):
    model = load_model()
    if model is None:
        return _estimate_with_keywords(patient_text, relative_text)

    inputs = preprocess_text(patient_text, relative_text, DEVICE)
    if inputs is None:
        return _estimate_with_keywords(patient_text, relative_text)

    return predict(inputs["input_ids"], inputs["attention_mask"])
