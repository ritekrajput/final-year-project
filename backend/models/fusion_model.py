from dataclasses import dataclass

import torch
import torch.nn as nn

from backend.models.text_model import TextRegressor


@dataclass
class FusionOutput:
    score: torch.Tensor
    fused_embedding: torch.Tensor
    text_embedding: torch.Tensor
    audio_embedding: torch.Tensor
    video_embedding: torch.Tensor


class CrossAttentionFusionModel(nn.Module):
    def __init__(
        self,
        text_model: TextRegressor | None = None,
        audio_dim: int = 16,
        video_dim: int = 7,
        fusion_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.text_model = text_model or TextRegressor()
        self.text_projection = nn.Linear(self.text_model.hidden_size, fusion_dim)
        self.audio_projection = nn.Sequential(
            nn.Linear(audio_dim, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
        )
        self.video_projection = nn.Sequential(
            nn.Linear(video_dim, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
        )
        self.audio_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.video_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.regressor = nn.Sequential(
            nn.LayerNorm(fusion_dim * 3),
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 1),
        )

    def encode_modalities(self, input_ids, attention_mask, audio_features, video_features):
        text_output = self.text_model.encode(input_ids, attention_mask)
        text_embedding = self.text_projection(text_output.cls_embedding).unsqueeze(1)
        audio_embedding = self.audio_projection(audio_features).unsqueeze(1)
        video_embedding = self.video_projection(video_features).unsqueeze(1)

        audio_attended, _ = self.audio_attention(
            query=text_embedding,
            key=audio_embedding,
            value=audio_embedding,
        )
        video_attended, _ = self.video_attention(
            query=text_embedding,
            key=video_embedding,
            value=video_embedding,
        )
        fused = torch.cat(
            [
                text_embedding.squeeze(1),
                audio_attended.squeeze(1),
                video_attended.squeeze(1),
            ],
            dim=-1,
        )
        return text_output, fused, audio_embedding.squeeze(1), video_embedding.squeeze(1)

    def forward(self, input_ids, attention_mask, audio_features, video_features):
        text_output, fused, audio_embedding, video_embedding = self.encode_modalities(
            input_ids,
            attention_mask,
            audio_features,
            video_features,
        )
        score = self.regressor(fused).squeeze(1)
        return FusionOutput(
            score=score,
            fused_embedding=fused,
            text_embedding=text_output.cls_embedding,
            audio_embedding=audio_embedding,
            video_embedding=video_embedding,
        )
