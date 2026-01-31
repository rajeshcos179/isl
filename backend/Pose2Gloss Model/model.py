# model.py
import torch
import torch.nn as nn

class PoseToGloss(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, 256)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=512,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.cls = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (B, T, D)
        x = self.input_proj(x)
        x = self.encoder(x)
        x = x.mean(dim=1)      # temporal pooling
        return self.cls(x)

