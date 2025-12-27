import torch
import torch.nn as nn
from models.encoder_rgb import RGBEncoder
from models.encoder_depth import DepthEncoder


class RGBDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_encoder = RGBEncoder()
        self.depth_encoder = DepthEncoder()

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor):
        rgb_features = self.rgb_encoder(rgb)
        depth_features = self.depth_encoder(depth)

        return {
            "rgb": rgb_features,
            "depth": depth_features,
        }
