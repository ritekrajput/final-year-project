import torch
import torch.nn as nn
from transformers import AutoModel

MODEL_NAME = "roberta-base"

class TextRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(self.roberta.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_token = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_token)
        return self.fc(x).squeeze(1)
