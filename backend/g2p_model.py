import torch
import torch.nn as nn

class GlossToPose(nn.Module):
    def __init__(self, num_glosses, max_len=64):
        super().__init__()

        self.max_len = max_len
        self.gloss_embed = nn.Embedding(num_glosses, 256)

        self.pos_embed = nn.Parameter(torch.randn(1, max_len, 256))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=512,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)

        self.out = nn.Linear(256, 384)

    def forward(self, gloss_id):
        """
        gloss_id: (B,)
        """
        B = gloss_id.size(0)

        memory = self.gloss_embed(gloss_id).unsqueeze(1)  # (B,1,256)
        tgt = self.pos_embed[:, :self.max_len, :].repeat(B, 1, 1)

        x = self.decoder(tgt, memory)
        return self.out(x)


class PoseToGloss(nn.Module):
    """
    Convert pose keypoints to gloss (sign language text).
    Input: (B, T, input_dim) - batch of sequences of keypoint vectors
    Output: (B, num_classes) - logits for each gloss class
    """
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


