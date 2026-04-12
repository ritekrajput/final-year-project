from argparse import ArgumentParser

import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score

from backend.models.feature_schema import AudioFeatureVector, VideoFeatureVector
from backend.services.multimodal_service import (
    DEVICE,
    clamp_score,
    get_fusion_model,
    get_text_model,
)
from backend.utils.preprocess import preprocess_text


def risk_level(score: float) -> str:
    if score <= 3:
        return "Low"
    if score <= 6:
        return "Moderate"
    if score <= 8:
        return "High"
    return "Critical"


def evaluate_text(frame: pd.DataFrame):
    model = get_text_model()
    if model is None:
        raise RuntimeError("Text checkpoint is not available.")

    preds = []
    labels = frame["score"].astype(float).tolist()
    for row in frame.to_dict("records"):
        inputs = preprocess_text(row["patient_text"], row.get("relative_text", ""), DEVICE)
        if inputs is None:
            raise RuntimeError("Tokenizer unavailable during evaluation.")
        with torch.no_grad():
            pred = model(inputs["input_ids"], inputs["attention_mask"]).item()
        preds.append(clamp_score(pred))
    return preds, labels


def evaluate_multimodal(frame: pd.DataFrame):
    model = get_fusion_model()
    if model is None:
        raise RuntimeError("Multimodal checkpoint is not available.")

    preds = []
    labels = frame["score"].astype(float).tolist()
    for row in frame.to_dict("records"):
        inputs = preprocess_text(row["patient_text"], row.get("relative_text", ""), DEVICE)
        if inputs is None:
            raise RuntimeError("Tokenizer unavailable during evaluation.")
        audio = AudioFeatureVector(
            **{k: float(row.get(k, 0.0)) for k in AudioFeatureVector.__dataclass_fields__}
        )
        video = VideoFeatureVector(
            **{k: float(row.get(k, 0.0)) for k in VideoFeatureVector.__dataclass_fields__}
        )
        with torch.no_grad():
            output = model(
                inputs["input_ids"],
                inputs["attention_mask"],
                audio.to_tensor(device=DEVICE).unsqueeze(0),
                video.to_tensor(device=DEVICE).unsqueeze(0),
            )
        preds.append(clamp_score(output.score.item()))
    return preds, labels


def summarize(preds, labels):
    mae = mean_absolute_error(labels, preds)
    r2 = r2_score(labels, preds) if len(set(labels)) > 1 else 0.0
    risk_pairs = list(zip([risk_level(x) for x in labels], [risk_level(x) for x in preds]))
    risk_accuracy = sum(1 for a, b in risk_pairs if a == b) / len(risk_pairs) if risk_pairs else 0.0
    confusion = {}
    for actual, predicted in risk_pairs:
        confusion[(actual, predicted)] = confusion.get((actual, predicted), 0) + 1
    print(f"mae={mae:.4f}")
    print(f"r2={r2:.4f}")
    print(f"risk_accuracy={risk_accuracy:.4f}")
    print("risk_confusion=")
    for key in sorted(confusion):
        print(f"  actual={key[0]} predicted={key[1]} count={confusion[key]}")


def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--mode", choices=["text", "multimodal"], required=True)
    parser.add_argument("--data", required=True)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    frame = pd.read_csv(args.data)
    if "score" not in frame.columns and "severity" in frame.columns:
        frame = frame.rename(columns={"severity": "score"})
    if args.mode == "text":
        preds, labels = evaluate_text(frame)
    else:
        preds, labels = evaluate_multimodal(frame)
    summarize(preds, labels)
