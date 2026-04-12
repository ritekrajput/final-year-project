from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import math
import random
from typing import Iterable

import pandas as pd

from backend.models.feature_schema import AUDIO_FEATURE_KEYS, VIDEO_FEATURE_KEYS


LOW_PATIENT_BANK = [
    "I feel steady and able to manage my routine.",
    "I am generally okay and can keep up with daily life.",
    "I still enjoy things and I do not feel overwhelmed right now.",
    "My mood feels stable and I can focus when I need to.",
    "I am doing fine overall, with only small ups and downs.",
]

MILD_PATIENT_BANK = [
    "I have been a bit stressed and tired lately, but I am still coping.",
    "Some days feel heavier, and it takes more effort to stay on track.",
    "I am a little uneasy and distracted, though I can still function.",
    "I notice mild low mood and reduced energy, but I am managing.",
    "Things feel somewhat harder than usual, especially in the evenings.",
]

MODERATE_PATIENT_BANK = [
    "I often feel overwhelmed, withdrawn, and low on energy.",
    "I have lost some interest in things and it is harder to concentrate.",
    "Most days feel emotionally heavy and it takes effort to get through them.",
    "I am struggling with motivation, sleep, and staying engaged.",
    "I feel persistently down, and even simple tasks take more effort.",
]

HIGH_PATIENT_BANK = [
    "I feel hopeless, isolated, and drained most days.",
    "It is getting hard to keep up with basic tasks and I feel stuck.",
    "The distress feels severe and I am struggling to stay connected.",
    "I have frequent low mood, very little energy, and strong withdrawal.",
    "I feel mentally exhausted and it is affecting how I function.",
]

CRITICAL_PATIENT_BANK = [
    "I feel deeply unsafe, numb, and overwhelmed, and I need help right away.",
    "The distress has become severe and I am struggling to cope at all.",
    "I feel in crisis, with very little ability to manage my thoughts or emotions.",
    "Everything feels unmanageable and I need immediate support.",
    "I am overwhelmed to the point that I cannot function normally.",
]

LOW_RELATIVE_BANK = [
    "The relative reports the patient is interacting normally and appears stable.",
    "They seem socially engaged and there are no major concerns at present.",
    "The patient appears calm, communicative, and consistent in daily routine.",
    "Family members notice steady behavior and normal day-to-day functioning.",
    "The relative does not see any major warning signs right now.",
]

MILD_RELATIVE_BANK = [
    "The relative notices mild stress, lower energy, and some inconsistency.",
    "They report the patient seems a bit quieter and less active than usual.",
    "The patient looks somewhat tired and a little less engaged socially.",
    "Family members have observed small changes in sleep and motivation.",
    "The relative thinks the patient is coping, but not as comfortably as before.",
]

MODERATE_RELATIVE_BANK = [
    "The relative says the patient has become quieter, less engaged, and more withdrawn.",
    "They have noticed clear signs of low mood and reduced interest in normal activities.",
    "The patient appears tired, isolated, and less responsive than before.",
    "Family members report noticeable changes in routine, energy, and concentration.",
    "The relative is concerned that the patient is struggling on most days.",
]

HIGH_RELATIVE_BANK = [
    "The relative observes clear withdrawal, persistent sadness, and reduced daily functioning.",
    "They say the patient seems hopeless, exhausted, and difficult to reassure.",
    "The patient has been isolating more and appears emotionally overwhelmed.",
    "Family members notice strong signs of distress and little recovery between episodes.",
    "The relative believes the patient is having a hard time managing basic activities.",
]

CRITICAL_RELATIVE_BANK = [
    "The relative is concerned about severe distress and believes immediate support is needed.",
    "They report the patient seems unsafe, overwhelmed, and unable to cope normally.",
    "Family members think the situation is urgent and requires close attention.",
    "The patient appears to be in crisis and the relative wants help right away.",
    "The relative sees a major change in functioning and is worried about safety.",
]

TEXT_OPENERS = [
    "Lately,",
    "Over the past few days,",
    "Recently,",
    "For a while now,",
    "In the last week,",
]

RELATIVE_OPENERS = [
    "According to the relative,",
    "They describe that",
    "Family members report that",
    "The relative observes that",
    "The caregiver notes that",
]

TEXT_CLOSERS = [
    "It is affecting my concentration and daily routine.",
    "I need more effort just to get through ordinary tasks.",
    "It has started to affect my sleep, focus, and motivation.",
    "I am still functioning, but it takes more energy than before.",
    "It feels harder to stay balanced throughout the day.",
]

CRITICAL_TEXT_CLOSERS = [
    "I need immediate support and cannot manage this alone.",
    "It feels urgent, and I need help right away.",
    "My ability to cope has dropped sharply and I need support now.",
]

EMOTION_NOISE_KEYS = [
    "happy",
    "sad",
    "angry",
    "fear",
    "surprise",
    "disgust",
    "neutral",
]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def severity_band(score: float) -> str:
    if score <= 2:
        return "low"
    if score <= 4:
        return "mild"
    if score <= 6:
        return "moderate"
    if score <= 8:
        return "high"
    return "critical"


def seeded_choice(rng: random.Random, values: list[str]) -> str:
    return values[rng.randrange(len(values))]


def normalize_whitespace(text: str) -> str:
    return " ".join(str(text).split())


def paraphrase_patient_text(score: float, rng: random.Random, source_text: str = "") -> str:
    band = severity_band(score)
    opener = seeded_choice(rng, TEXT_OPENERS)
    if band == "low":
        core = seeded_choice(rng, LOW_PATIENT_BANK)
        closer = seeded_choice(rng, TEXT_CLOSERS)
    elif band == "mild":
        core = seeded_choice(rng, MILD_PATIENT_BANK)
        closer = seeded_choice(rng, TEXT_CLOSERS)
    elif band == "moderate":
        core = seeded_choice(rng, MODERATE_PATIENT_BANK)
        closer = seeded_choice(rng, TEXT_CLOSERS)
    elif band == "high":
        core = seeded_choice(rng, HIGH_PATIENT_BANK)
        closer = seeded_choice(rng, TEXT_CLOSERS)
    else:
        core = seeded_choice(rng, CRITICAL_PATIENT_BANK)
        closer = seeded_choice(rng, CRITICAL_TEXT_CLOSERS)

    source_hint = ""
    normalized_source = normalize_whitespace(source_text).strip()
    if normalized_source and rng.random() < 0.35:
        source_hint = f" It still echoes the feeling that {normalized_source.lower()}."

    return normalize_whitespace(f"{opener} {core} {closer}{source_hint}")


def paraphrase_relative_text(score: float, rng: random.Random, source_text: str = "") -> str:
    band = severity_band(score)
    opener = seeded_choice(rng, RELATIVE_OPENERS)
    if band == "low":
        core = seeded_choice(rng, LOW_RELATIVE_BANK)
    elif band == "mild":
        core = seeded_choice(rng, MILD_RELATIVE_BANK)
    elif band == "moderate":
        core = seeded_choice(rng, MODERATE_RELATIVE_BANK)
    elif band == "high":
        core = seeded_choice(rng, HIGH_RELATIVE_BANK)
    else:
        core = seeded_choice(rng, CRITICAL_RELATIVE_BANK)

    source_hint = ""
    normalized_source = normalize_whitespace(source_text).strip()
    if normalized_source and rng.random() < 0.25:
        source_hint = f" The earlier observation that {normalized_source.lower()} still feels relevant."

    return normalize_whitespace(f"{opener} {core}{source_hint}")


def jitter(value: float, rng: random.Random, scale: float, low: float | None = None, high: float | None = None) -> float:
    result = value + rng.gauss(0.0, scale)
    if low is not None:
        result = max(low, result)
    if high is not None:
        result = min(high, result)
    return result


def severity_to_phq9(score: float, rng: random.Random) -> float:
    baseline = (score / 10.0) * 27.0
    return round(clamp(baseline + rng.gauss(0.0, 1.0), 0.0, 27.0), 4)


def synthesize_audio_features(score: float, rng: random.Random, transcript: str) -> dict[str, float]:
    severity = clamp(score / 10.0, 0.0, 1.0)
    words = max(4, len(normalize_whitespace(transcript).split()))
    return {
        "energy": round(clamp(1.25 - 0.85 * severity + rng.gauss(0.0, 0.08), 0.0, 2.0), 4),
        "spectral_centroid": round(clamp(0.85 - 0.25 * severity + rng.gauss(0.0, 0.07), 0.0, 2.0), 4),
        "mfcc": round(rng.gauss(0.0, 0.65) + (0.2 - 0.15 * severity), 4),
        "speaking_rate": round(clamp(19.0 - 9.0 * severity + rng.gauss(0.0, 1.0), 1.0, 30.0), 4),
        "pause_duration": round(clamp(0.15 + 1.15 * severity + rng.gauss(0.0, 0.08), 0.0, 3.0), 4),
        "jitter": round(clamp(0.03 + 0.14 * severity + abs(rng.gauss(0.0, 0.02)), 0.0, 1.0), 4),
        "shimmer": round(clamp(0.04 + 0.13 * severity + abs(rng.gauss(0.0, 0.02)), 0.0, 1.0), 4),
        "hnr": round(clamp(1.05 - 0.7 * severity + rng.gauss(0.0, 0.09), 0.0, 2.0), 4),
        "pitch": round(clamp(1.0 - 0.35 * severity + rng.gauss(0.0, 0.09), 0.0, 2.0), 4),
        "formant1": round(clamp(0.55 + 0.12 * severity + rng.gauss(0.0, 0.06), 0.0, 2.0), 4),
        "formant2": round(clamp(0.5 + 0.10 * severity + rng.gauss(0.0, 0.06), 0.0, 2.0), 4),
        "transcript_length": float(words),
        "audio_emotion_confidence": round(clamp(0.35 + 0.45 * severity + abs(rng.gauss(0.0, 0.05)), 0.0, 1.0), 4),
        "audio_sad": round(clamp(0.05 + 0.75 * severity + rng.gauss(0.0, 0.06), 0.0, 1.5), 4),
        "audio_fear": round(clamp(0.04 + 0.60 * severity + rng.gauss(0.0, 0.05), 0.0, 1.5), 4),
        "audio_angry": round(clamp(0.03 + 0.35 * severity + rng.gauss(0.0, 0.05), 0.0, 1.5), 4),
    }


def synthesize_video_features(score: float, rng: random.Random) -> dict[str, float]:
    severity = clamp(score / 10.0, 0.0, 1.0)
    return {
        "happy": round(clamp(1.05 - 0.95 * severity + rng.gauss(0.0, 0.08), 0.0, 1.5), 4),
        "sad": round(clamp(0.06 + 0.78 * severity + rng.gauss(0.0, 0.06), 0.0, 1.5), 4),
        "angry": round(clamp(0.05 + 0.45 * severity + rng.gauss(0.0, 0.05), 0.0, 1.5), 4),
        "fear": round(clamp(0.04 + 0.55 * severity + rng.gauss(0.0, 0.05), 0.0, 1.5), 4),
        "surprise": round(clamp(0.16 + 0.18 * abs(rng.gauss(0.0, 1.0)), 0.0, 1.0), 4),
        "disgust": round(clamp(0.03 + 0.28 * severity + rng.gauss(0.0, 0.04), 0.0, 1.0), 4),
        "neutral": round(clamp(0.95 - 0.55 * severity + rng.gauss(0.0, 0.07), 0.0, 1.5), 4),
    }


def synthesize_multimodal_row(row: dict, rng: random.Random, variant_index: int) -> dict:
    raw_phq9 = float(row.get("phq9", 0.0))
    if raw_phq9 <= 10.0:
        score = clamp(raw_phq9 + rng.gauss(0.0, 0.6), 0.0, 10.0)
    else:
        score = clamp((raw_phq9 / 27.0) * 10.0 + rng.gauss(0.0, 0.45), 0.0, 10.0)

    patient_text = paraphrase_patient_text(score, rng, row.get("text", ""))
    relative_text = paraphrase_relative_text(score, rng, row.get("text", ""))
    audio = synthesize_audio_features(score, rng, patient_text)
    video = synthesize_video_features(score, rng)
    phq9 = severity_to_phq9(score, rng)

    record = {
        "source_id": row.get("id", ""),
        "augmentation_index": variant_index,
        "patient_text": patient_text,
        "relative_text": relative_text,
        "score": round(score, 4),
        "phq9": phq9,
        "label": int(score >= 6.0),
    }
    record.update(audio)
    record.update(video)
    return record


def synthesize_text_row(row: pd.Series, rng: random.Random, variant_index: int) -> dict:
    score = float(row["severity"])
    patient_text = paraphrase_patient_text(score, rng, row.get("patient_text", ""))
    relative_text = paraphrase_relative_text(score, rng, row.get("relative_text", ""))
    return {
        "source_id": row.get("id", ""),
        "augmentation_index": variant_index,
        "patient_text": patient_text,
        "relative_text": relative_text,
        "score": round(score, 4),
    }


def expand_text_dataset(input_path: Path, output_path: Path, variants_per_row: int, seed: int) -> pd.DataFrame:
    frame = pd.read_csv(input_path)
    if "severity" not in frame.columns and "score" not in frame.columns:
        raise ValueError("metadata CSV must contain 'severity' or 'score'.")
    if "severity" not in frame.columns:
        frame = frame.rename(columns={"score": "severity"})
    required = {"patient_text", "relative_text", "severity"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"metadata CSV is missing columns: {sorted(missing)}")

    rng = random.Random(seed)
    records: list[dict] = []
    for _, row in frame.iterrows():
        for variant_index in range(variants_per_row):
            records.append(synthesize_text_row(row, rng, variant_index))
    out = pd.DataFrame.from_records(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


def expand_multimodal_dataset(input_path: Path, output_path: Path, variants_per_row: int, seed: int) -> pd.DataFrame:
    records: list[dict] = []
    rng = random.Random(seed)
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            for variant_index in range(variants_per_row):
                records.append(synthesize_multimodal_row(row, rng, variant_index))
    out = pd.DataFrame.from_records(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Broaden the synthetic text and multimodal training datasets.")
    parser.add_argument("--metadata-input", default="dataset/metadata.csv", help="Source metadata CSV")
    parser.add_argument(
        "--multimodal-input",
        default="dataset/synthetic_samples.jsonl",
        help="Source multimodal JSONL seed data",
    )
    parser.add_argument(
        "--text-output",
        default="dataset/text_dataset_broadened.csv",
        help="Output CSV for broadened text training data",
    )
    parser.add_argument(
        "--multimodal-output",
        default="dataset/multimodal_dataset_broadened.csv",
        help="Output CSV for broadened multimodal training data",
    )
    parser.add_argument("--text-variants", type=int, default=2, help="How many text rows to generate per source row")
    parser.add_argument(
        "--multimodal-variants",
        type=int,
        default=10,
        help="How many multimodal rows to generate per source row",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata_input = Path(args.metadata_input)
    multimodal_input = Path(args.multimodal_input)
    text_output = Path(args.text_output)
    multimodal_output = Path(args.multimodal_output)

    print(f"building_text_dataset_from={metadata_input}")
    text_frame = expand_text_dataset(metadata_input, text_output, args.text_variants, args.seed)
    print(f"saved_text_dataset={text_output} rows={len(text_frame)}")

    print(f"building_multimodal_dataset_from={multimodal_input}")
    multimodal_frame = expand_multimodal_dataset(multimodal_input, multimodal_output, args.multimodal_variants, args.seed)
    print(f"saved_multimodal_dataset={multimodal_output} rows={len(multimodal_frame)}")

    print("done")


if __name__ == "__main__":
    main()
