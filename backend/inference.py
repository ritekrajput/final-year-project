import torch
from backend.models.text_model import TextRegressor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "backend/models/best_regression_model.pt"

def load_model():
    model = TextRegressor()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

def predict(input_ids, attention_mask):
    with torch.no_grad():
        output = model(input_ids, attention_mask)

    severity = float(output.cpu().item())
    severity = max(1, min(10, severity))
    return severity
