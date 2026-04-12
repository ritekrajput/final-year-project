from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from backend.models.feature_schema import (
    AUDIO_FEATURE_KEYS,
    VIDEO_FEATURE_KEYS,
)
from backend.models.fusion_model import CrossAttentionFusionModel
from backend.utils.preprocess import get_tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MultimodalDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, max_len: int = 256):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = get_tokenizer()
        self.max_len = max_len
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer could not be loaded for training.")

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        text = f"{row['patient_text']} </s> {row.get('relative_text', '')}"
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        audio_features = torch.tensor(
            [float(row.get(key, 0.0)) for key in AUDIO_FEATURE_KEYS],
            dtype=torch.float32,
        )
        video_features = torch.tensor(
            [float(row.get(key, 0.0)) for key in VIDEO_FEATURE_KEYS],
            dtype=torch.float32,
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "audio_features": audio_features,
            "video_features": video_features,
            "label": torch.tensor(float(row["score"]), dtype=torch.float32),
        }


def evaluate(model, dataloader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            output = model(
                batch["input_ids"].to(DEVICE),
                batch["attention_mask"].to(DEVICE),
                batch["audio_features"].to(DEVICE),
                batch["video_features"].to(DEVICE),
            )
            preds.extend(output.score.detach().cpu().tolist())
            labels.extend(batch["label"].detach().cpu().tolist())
    mae = mean_absolute_error(labels, preds)
    r2 = r2_score(labels, preds) if len(set(labels)) > 1 else 0.0
    return mae, r2


def train(args):
    print(f"loading_dataset={args.data}")
    frame = pd.read_csv(args.data)
    if args.max_rows is not None and args.max_rows > 0 and len(frame) > args.max_rows:
        frame = frame.sample(n=args.max_rows, random_state=args.random_state).reset_index(drop=True)
    print(f"dataset_rows={len(frame)}")
    print("splitting_train_validation")
    train_df, val_df = train_test_split(frame, test_size=args.val_split, random_state=42)
    print(f"train_rows={len(train_df)} val_rows={len(val_df)}")
    print("building_dataloaders")
    train_loader = DataLoader(MultimodalDataset(train_df), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(MultimodalDataset(val_df), batch_size=args.batch_size)

    print("initializing_model")
    model = CrossAttentionFusionModel().to(DEVICE)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    print(f"device={DEVICE} epochs={args.epochs} batch_size={args.batch_size}")

    best_mae = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"epoch_start={epoch + 1}/{args.epochs}")
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            output = model(
                batch["input_ids"].to(DEVICE),
                batch["attention_mask"].to(DEVICE),
                batch["audio_features"].to(DEVICE),
                batch["video_features"].to(DEVICE),
            )
            loss = criterion(output.score, batch["label"].to(DEVICE))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if step == 1 or step % args.log_every == 0 or step == len(train_loader):
                avg_loss = running_loss / step
                print(
                    f"epoch={epoch + 1} batch={step}/{len(train_loader)} "
                    f"train_loss={avg_loss:.4f}"
                )

        print(f"epoch_validation_start={epoch + 1}")
        val_mae, val_r2 = evaluate(model, val_loader)
        print(f"epoch={epoch + 1} val_mae={val_mae:.4f} val_r2={val_r2:.4f}")
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), output_path)
            print(f"saved_best={output_path}")


def build_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--data",
        required=True,
        help="CSV with patient_text, relative_text, score and engineered audio/video feature columns",
    )
    parser.add_argument(
        "--output",
        default="backend/models/best_multimodal_model.pt",
        help="Path to save best checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional sample size for faster training")
    parser.add_argument("--random-state", type=int, default=42)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
