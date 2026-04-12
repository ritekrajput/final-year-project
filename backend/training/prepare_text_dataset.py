from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


def prepare(input_path: str, output_path: str, max_rows: int | None = None, random_state: int = 42):
    frame = pd.read_csv(input_path)
    required = {"patient_text", "relative_text"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if "score" not in frame.columns:
        if "severity" in frame.columns:
            frame = frame.rename(columns={"severity": "score"})
        else:
            raise ValueError("Input CSV must contain either 'score' or 'severity'.")

    out = frame[["patient_text", "relative_text", "score"]].copy()
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
        default="dataset/metadata.csv",
        help="Source CSV with patient_text, relative_text, and severity/score",
    )
    parser.add_argument(
        "--output",
        default="dataset/text_dataset.csv",
        help="Prepared text training CSV",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Optional sample size for faster experiments")
    parser.add_argument("--random-state", type=int, default=42)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    prepare(args.input, args.output, max_rows=args.max_rows, random_state=args.random_state)
