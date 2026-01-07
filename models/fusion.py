"""
Feature Fusion Module for RGB-Depth Features
Combines multi-scale features from RGB and Depth encoders
"""

import torch
import torch.nn as nn


class FeatureFusion(nn.Module):
    """
    Fuses RGB and Depth features from SegFormer encoders
    Supports multiple fusion strategies
    """

    def __init__(self, fusion_type='concat', feature_dims=[32, 64, 160, 256]):
        """
        Args:
            fusion_type: 'concat', 'add', 'attention'
            feature_dims: List of feature dimensions at each level (SegFormer-B0)
        """
        super().__init__()
        self.fusion_type = fusion_type
        self.feature_dims = feature_dims

        if fusion_type == 'concat':
            # Projection layers to reduce concatenated features
            self.projections = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(dim * 2, dim, 1),  # 2x dim -> dim
                    nn.BatchNorm2d(dim),
                    nn.ReLU(inplace=True)
                )
                for dim in feature_dims
            ])

        elif fusion_type == 'attention':
            # Attention-based fusion
            self.attention_modules = nn.ModuleList([
                AttentionFusion(dim)
                for dim in feature_dims
            ])

    def forward(self, rgb_features, depth_features):
        """
        Fuse RGB and Depth features

        Args:
            rgb_features: List of 4 feature maps from RGB encoder
            depth_features: List of 4 feature maps from Depth encoder

        Returns:
            List of 4 fused feature maps
        """
        if self.fusion_type == 'add':
            # Simple addition
            fused = [rgb + depth for rgb, depth in zip(rgb_features, depth_features)]

        elif self.fusion_type == 'concat':
            # Concatenate and project
            fused = []
            for i, (rgb, depth) in enumerate(zip(rgb_features, depth_features)):
                concat = torch.cat([rgb, depth], dim=1)
                fused_feat = self.projections[i](concat)
                fused.append(fused_feat)

        elif self.fusion_type == 'attention':
            # Attention-based fusion
            fused = []
            for i, (rgb, depth) in enumerate(zip(rgb_features, depth_features)):
                fused_feat = self.attention_modules[i](rgb, depth)
                fused.append(fused_feat)

        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

        return fused


class AttentionFusion(nn.Module):
    """
    Attention-based fusion module
    Learns to weight RGB and Depth features
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 2, 1),  # 2 channels: weight for RGB and Depth
            nn.Softmax(dim=1)
        )

        # Feature projection
        self.projection = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb, depth):
        """
        Args:
            rgb: RGB features (B, C, H, W)
            depth: Depth features (B, C, H, W)

        Returns:
            Fused features (B, C, H, W)
        """
        # Concatenate
        concat = torch.cat([rgb, depth], dim=1)

        # Compute attention weights
        attention = self.attention(concat)  # (B, 2, H, W)
        rgb_weight = attention[:, 0:1, :, :]
        depth_weight = attention[:, 1:2, :, :]

        # Weighted fusion
        weighted = torch.cat([
            rgb * rgb_weight,
            depth * depth_weight
        ], dim=1)

        # Project to output dimension
        fused = self.projection(weighted)

        return fused
