from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModel

MODEL_NAME = "roberta-base"


@dataclass
class TextModelOutput:
    score: torch.Tensor
    cls_embedding: torch.Tensor
    sequence_embedding: torch.Tensor


class TextRegressor(nn.Module):
    def __init__(self, dropout: float = 0.4):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(dropout)
        self.regression_head = nn.Linear(self.roberta.config.hidden_size, 1)

    @property
    def hidden_size(self) -> int:
        return self.roberta.config.hidden_size

    def encode(self, input_ids, attention_mask) -> TextModelOutput:
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        sequence_embedding = outputs.last_hidden_state
        cls_embedding = sequence_embedding[:, 0, :]
        score = self.regression_head(self.dropout(cls_embedding)).squeeze(1)
        return TextModelOutput(
            score=score,
            cls_embedding=cls_embedding,
            sequence_embedding=sequence_embedding,
        )

    def forward(self, input_ids, attention_mask):
        return self.encode(input_ids, attention_mask).score
