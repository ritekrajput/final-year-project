from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from backend.models.text_model import TextRegressor
from backend.utils.preprocess import get_tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TextRegressionDataset(Dataset):
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
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(float(row["score"]), dtype=torch.float32),
        }


def evaluate(model, dataloader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            target = batch["label"].to(DEVICE)
            output = model(input_ids, attention_mask)
            preds.extend(output.detach().cpu().tolist())
            labels.extend(target.detach().cpu().tolist())
    mae = mean_absolute_error(labels, preds)
    r2 = r2_score(labels, preds) if len(set(labels)) > 1 else 0.0
    return mae, r2


def train(args):
    print(f"loading_dataset={args.data}")
    frame = pd.read_csv(args.data)
    if "score" not in frame.columns and "severity" in frame.columns:
        frame = frame.rename(columns={"severity": "score"})
    required = {"patient_text", "relative_text", "score"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if args.max_rows is not None and args.max_rows > 0 and len(frame) > args.max_rows:
        frame = frame.sample(n=args.max_rows, random_state=args.random_state).reset_index(drop=True)
    print(f"dataset_rows={len(frame)}")
    print("splitting_train_validation")
    train_df, val_df = train_test_split(frame, test_size=args.val_split, random_state=42)
    print(f"train_rows={len(train_df)} val_rows={len(val_df)}")
    print("building_dataloaders")
    train_loader = DataLoader(TextRegressionDataset(train_df), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TextRegressionDataset(val_df), batch_size=args.batch_size)

    print("initializing_model")
    model = TextRegressor().to(DEVICE)
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
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            target = batch["label"].to(DEVICE)
            prediction = model(input_ids, attention_mask)
            loss = criterion(prediction, target)
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
    parser.add_argument("--data", required=True, help="CSV with patient_text, relative_text, score")
    parser.add_argument(
        "--output",
        default="backend/models/best_regression_model.pt",
        help="Path to save best checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional sample size for faster training")
    parser.add_argument("--random-state", type=int, default=42)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
