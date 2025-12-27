import torch
import torch.nn as nn
from transformers import SegformerModel


class RGBEncoder(nn.Module):
    def __init__(self, model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512"):
        super().__init__()
        self.backbone = SegformerModel.from_pretrained(model_name)

    def forward(self, rgb: torch.Tensor):
        # rgb: (B, 3, H, W)
        out = self.backbone(pixel_values=rgb, output_hidden_states=True)
        return out.hidden_states
