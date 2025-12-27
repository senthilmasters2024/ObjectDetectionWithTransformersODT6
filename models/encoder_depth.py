import torch
import torch.nn as nn
from transformers import SegformerModel


class DepthEncoder(nn.Module):
    def __init__(self, model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512"):
        super().__init__()
        self.backbone = SegformerModel.from_pretrained(model_name)

    def forward(self, depth: torch.Tensor):
        # depth: (B, 1, H, W) or (B, 3, H, W)
        if depth.ndim != 4:
            raise ValueError(f"Expected depth shape (B,C,H,W), got {depth.shape}")

        if depth.shape[1] == 1:
            depth = depth.repeat(1, 3, 1, 1)
        elif depth.shape[1] != 3:
            raise ValueError(f"Depth must have 1 or 3 channels, got {depth.shape[1]}")

        out = self.backbone(pixel_values=depth, output_hidden_states=True)
        return out.hidden_states
