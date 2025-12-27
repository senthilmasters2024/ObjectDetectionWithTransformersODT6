Transformer-Based RGB–Depth Encoder (Base Implementation)
Scope of This Implementation

This part of the project focuses on implementing the base transformer encoder architecture for RGB–Depth perception.
Only the encoder-level model structure is implemented here. No fusion, decoder, training, or evaluation is included at this stage.

Implemented Components
1. RGB Encoder

Implemented using a SegFormer transformer backbone.

Input shape: (B, 3, H, W)

The encoder processes RGB images and extracts multi-scale feature representations.

Output consists of feature maps from multiple encoder stages.

2. Depth Encoder

Implemented using the same SegFormer backbone as the RGB encoder.

Input shape: (B, 1, H, W)

The single-channel depth input is converted to three channels before being passed to the encoder.

Produces multi-scale feature maps aligned with the RGB encoder output.

3. Dual-Encoder Base Model

A base model structure is created to combine the RGB and Depth encoders.

Both modalities are processed in parallel using separate encoders.

The model outputs RGB and Depth feature maps without applying any fusion or decoding.

Output structure:

{
  "rgb":   [feature_level_0, feature_level_1, feature_level_2, feature_level_3],
  "depth": [feature_level_0, feature_level_1, feature_level_2, feature_level_3]
}

Input Format

RGB input: (B, 3, H, W)

Depth input: (B, 1, H, W)

At this stage, inputs are expected to be provided as tensors.
The source of the input (dataset images or camera frames) is not handled in this implementation.

Current Status

Transformer-based RGB encoder implemented

Transformer-based Depth encoder implemented

Dual-encoder base model structure completed

Forward pass skeleton defined without fusion