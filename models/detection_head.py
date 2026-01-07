"""
Detection Head for SegFormer-based Object Detection
Converts multi-scale features to bounding box predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionHead(nn.Module):
    """
    Transformer-based detection head for SegFormer features
    Similar to DETR but adapted for multi-scale features
    """

    def __init__(
        self,
        feature_dims=[32, 64, 160, 256],
        hidden_dim=256,
        num_queries=100,
        num_classes=80,  # COCO classes
        num_decoder_layers=6
    ):
        """
        Args:
            feature_dims: Dimensions of multi-scale features from encoder
            hidden_dim: Hidden dimension for transformer
            num_queries: Number of object queries (max detections)
            num_classes: Number of object classes
            num_decoder_layers: Number of transformer decoder layers
        """
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # Feature projection layers - project all levels to same dimension
        self.input_projections = nn.ModuleList([
            nn.Conv2d(dim, hidden_dim, 1)
            for dim in feature_dims
        ])

        # Positional encoding
        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2)

        # Object queries (learnable)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )

        # Prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)  # 4 bbox coords

    def forward(self, multi_scale_features):
        """
        Args:
            multi_scale_features: List of 4 feature maps from fusion module
                                  [(B,C1,H1,W1), (B,C2,H2,W2), ...]

        Returns:
            dict with:
                'pred_logits': (B, num_queries, num_classes)
                'pred_boxes': (B, num_queries, 4) in [cx, cy, w, h] format
        """
        batch_size = multi_scale_features[0].shape[0]

        # Project all features to same dimension and flatten
        projected_features = []
        pos_embeddings = []

        for i, feat in enumerate(multi_scale_features):
            # Project to hidden_dim
            proj_feat = self.input_projections[i](feat)  # (B, hidden_dim, H, W)

            # Get positional encoding
            pos = self.position_embedding(proj_feat)  # (B, hidden_dim, H, W)

            # Flatten spatial dimensions
            B, C, H, W = proj_feat.shape
            proj_feat = proj_feat.flatten(2).permute(2, 0, 1)  # (H*W, B, C)
            pos = pos.flatten(2).permute(2, 0, 1)  # (H*W, B, C)

            projected_features.append(proj_feat)
            pos_embeddings.append(pos)

        # Concatenate all scales
        memory = torch.cat(projected_features, dim=0)  # (sum(H*W), B, C)
        pos_embed = torch.cat(pos_embeddings, dim=0)  # (sum(H*W), B, C)

        # Object queries
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)  # (num_queries, B, C)

        # Transformer decoder
        tgt = torch.zeros_like(query_embed)  # (num_queries, B, C)
        hs = self.transformer_decoder(
            tgt,
            memory,
            memory_key_padding_mask=None,
            pos=pos_embed,
            query_pos=query_embed
        )  # (num_queries, B, C)

        # Predictions
        hs = hs.permute(1, 0, 2)  # (B, num_queries, C)

        outputs_class = self.class_embed(hs)  # (B, num_queries, num_classes)
        outputs_coord = self.bbox_embed(hs).sigmoid()  # (B, num_queries, 4)

        return {
            'pred_logits': outputs_class,
            'pred_boxes': outputs_coord
        }


class MLP(nn.Module):
    """Simple multi-layer perceptron"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PositionEmbeddingSine(nn.Module):
    """
    Sine-based positional encoding
    """

    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * 3.141592653589793
        self.scale = scale

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)

        Returns:
            Positional encoding (B, C, H, W)
        """
        B, C, H, W = x.shape
        mask = torch.zeros(B, H, W, dtype=torch.bool, device=x.device)

        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos
