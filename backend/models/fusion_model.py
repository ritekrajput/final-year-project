import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self, text_model):
        super().__init__()
        self.text_model = text_model

    def forward(self, input_ids, attention_mask):
        text_score = self.text_model(input_ids, attention_mask)
        return text_score
