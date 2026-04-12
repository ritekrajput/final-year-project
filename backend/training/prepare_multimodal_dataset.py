from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


def _safe_get(values, index, default=0.0):
    if not isinstance(values, list):
        return default
    if index >= len(values):
        return default
    return float(values[index])


def prepare(input_path: str, output_path: str, max_rows: int | None = None, random_state: int = 42):
    frame = pd.read_json(input_path, lines=True)
    if not {"text", "audio_vec", "visual_vec"}.issubset(frame.columns):
        raise ValueError("JSONL must contain text, audio_vec, and visual_vec fields.")

    records = []
    for _, row in frame.iterrows():
        stats = row.get("stats", {}) if isinstance(row.get("stats"), dict) else {}
        audio_vec = row.get("audio_vec", [])
        visual_vec = row.get("visual_vec", [])
        phq9 = float(row.get("phq9", 0.0))

        record = {
            "patient_text": row.get("text", ""),
            "relative_text": "",
            "score": round((phq9 / 27.0) * 10.0, 4),
            "energy": _safe_get(audio_vec, 0),
            "spectral_centroid": _safe_get(audio_vec, 1),
            "mfcc": _safe_get(audio_vec, 2),
            "speaking_rate": float(stats.get("posts_per_week", 0.0)),
            "pause_duration": float(stats.get("late_night_ratio", 0.0)),
            "jitter": _safe_get(audio_vec, 3),
            "shimmer": _safe_get(audio_vec, 4),
            "hnr": _safe_get(audio_vec, 5),
            "pitch": _safe_get(audio_vec, 6),
            "formant1": _safe_get(audio_vec, 7),
            "formant2": _safe_get(audio_vec, 8),
            "transcript_length": float(len(str(row.get("text", "")).split())),
            "audio_emotion_confidence": abs(_safe_get(audio_vec, 9)),
            "audio_sad": max(_safe_get(audio_vec, 10), 0.0),
            "audio_fear": max(_safe_get(audio_vec, 11), 0.0),
            "audio_angry": max(_safe_get(audio_vec, 12), 0.0),
            "happy": max(_safe_get(visual_vec, 0), 0.0),
            "sad": max(_safe_get(visual_vec, 1), 0.0),
            "angry": max(_safe_get(visual_vec, 2), 0.0),
            "fear": max(_safe_get(visual_vec, 3), 0.0),
            "surprise": max(_safe_get(visual_vec, 4), 0.0),
            "disgust": max(_safe_get(visual_vec, 5), 0.0),
            "neutral": max(_safe_get(visual_vec, 6), 0.0),
        }
        records.append(record)

    out = pd.DataFrame.from_records(records)
    if max_rows is not None and max_rows > 0 and len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=random_state).reset_index(drop=True)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"saved={output}")
    print(f"rows={len(out)}")


def build_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="dataset/synthetic_samples.jsonl",
        help="Source multimodal JSONL",
    )
    parser.add_argument(
        "--output",
        default="dataset/multimodal_dataset.csv",
        help="Prepared multimodal training CSV",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Optional sample size for faster experiments")
    parser.add_argument("--random-state", type=int, default=42)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    prepare(args.input, args.output, max_rows=args.max_rows, random_state=args.random_state)
