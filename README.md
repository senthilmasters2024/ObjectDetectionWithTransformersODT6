# RealSense D435 + Transformer Object Detection POC

Proof of Concept for object detection using Transformer models with Intel RealSense D435 depth camera.

## Features

- ✅ DETR (Detection Transformer) integration
- ✅ Single frame and continuous detection modes
- ✅ Depth estimation for detected objects
- ✅ Multiple transformer models comparison
- ✅ Real-time visualization

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements_detr.txt
```

Or manually:
```bash
pip install pyrealsense2 torch torchvision transformers opencv-python pillow numpy
```

### 2. Verify RealSense

Make sure your D435 is connected and working in RealSense Viewer.

## Usage

### Quick Start - Single Frame Detection

```bash
python realsense_detr_poc.py
```

Select option **1** for single frame detection.

**What it does:**
- Captures one frame from D435
- Runs DETR object detection
- Shows detected objects with confidence scores
- Displays distance to each object
- Saves annotated image

### Continuous Detection (Live)

```bash
python realsense_detr_poc.py
```

Select option **2** for continuous detection.

**Controls:**
- Press **'q'** to quit
- Press **'s'** to save screenshot

### Compare Multiple Transformer Models

```bash
python transformer_models_comparison.py
```

**Available models:**
1. **DETR (ResNet-50)** - Good balance, ~2-3 FPS
2. **DETR (ResNet-101)** - Better accuracy, ~1-2 FPS  
3. **Conditional DETR** - Faster convergence
4. **Table Transformer** - Specialized for tables
5. **DETA** - State-of-the-art (requires more memory)

## Output

### Single Frame Mode
- **detection_result.jpg** - Annotated image with bounding boxes

### Continuous Mode
- **detection_YYYYMMDD_HHMMSS.jpg** - Screenshot when pressing 's'

### Benchmark Mode
- **result_MODEL_NAME.jpg** - Results for each model tested

## Expected Performance

**On CPU (Intel i5/i7):**
- DETR ResNet-50: ~2-4 seconds per frame
- DETR ResNet-101: ~4-6 seconds per frame

**On GPU (NVIDIA RTX 3060+):**
- DETR ResNet-50: ~0.1-0.3 seconds per frame (3-10 FPS)
- DETR ResNet-101: ~0.2-0.5 seconds per frame (2-5 FPS)

## Detection Output Format

Each detection includes:
- **Label**: Object class (e.g., "person", "car", "laptop")
- **Confidence**: Detection confidence (0.0 to 1.0)
- **Bounding Box**: [xmin, ymin, xmax, ymax]
- **Depth**: Distance in meters (from RealSense depth sensor)

Example output:
```
Found 3 objects:
  1. person: 0.95 | Distance: 1.23m
  2. laptop: 0.87 | Distance: 0.85m
  3. cup: 0.73 | Distance: 1.45m
```

## Customization

### Adjust Confidence Threshold

In `realsense_detr_poc.py`, line 96:
```python
detector = RealsenseDETR(confidence_threshold=0.7)  # Change 0.7 to your value
```

**Recommendations:**
- **0.5-0.6**: More detections, more false positives
- **0.7**: Balanced (default)
- **0.8-0.9**: High confidence only, fewer false positives

### Change Camera Resolution

In `realsense_detr_poc.py`, lines 64-65:
```python
self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
```

**Options:**
- **640×480** - Faster, lower quality
- **1280×720** - Balanced (default)
- **1920×1080** - Higher quality, slower

### Skip Frames for Speed (Continuous Mode)

In `realsense_detr_poc.py`, line 218:
```python
if frame_count % 1 == 0:  # Process every frame
```

Change to:
```python
if frame_count % 5 == 0:  # Process every 5th frame (faster)
```

## Detected Object Classes

DETR is trained on COCO dataset with 80 classes:

**Common objects:**
- person, car, bicycle, motorcycle, bus, truck
- dog, cat, bird, horse, cow, sheep
- laptop, keyboard, mouse, cell phone, tv
- chair, couch, bed, dining table
- cup, bottle, wine glass, fork, knife, spoon
- apple, banana, orange, sandwich, pizza
- book, clock, vase, scissors
- ... and 50+ more

Full list: [COCO Classes](https://cocodataset.org/#explore)

## Troubleshooting

### "No frames received" error
- Make sure RealSense Viewer shows camera working
- Check USB 3.0 connection (blue port)
- Update firmware in RealSense Viewer

### "Model download failed"
- Check internet connection
- Models download from Hugging Face (~160MB)
- First run takes longer due to download

### "CUDA out of memory"
- Use CPU instead (automatic fallback)
- Reduce resolution (640×480)
- Close other applications

### Slow performance
- GPU strongly recommended for real-time
- On CPU: Use single frame mode
- Skip frames in continuous mode
- Use DETR ResNet-50 (fastest)

## Advanced Usage

### Use Different Model

```python
from transformers import DetrImageProcessor, DetrForObjectDetection

# Change model
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101")
```

### Get 3D Coordinates

```python
# After detection, get 3D point
depth = depth_frame.get_distance(x, y)  # Distance in meters
intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
print(f"3D coordinates: X={point_3d[0]:.2f}m, Y={point_3d[1]:.2f}m, Z={point_3d[2]:.2f}m")
```

### Save Point Cloud with Detections

```python
pc = rs.pointcloud()
points = pc.calculate(depth_frame)
pc.export_to_ply("output.ply", color_frame)
```

## Model Comparison

| Model | Accuracy | Speed | Memory | Best For |
|-------|----------|-------|--------|----------|
| DETR ResNet-50 | Good | Fast | 160MB | General POC |
| DETR ResNet-101 | Better | Slow | 230MB | High accuracy |
| Conditional DETR | Good | Fast | 160MB | Fine-tuning |
| DETA | Best | Slowest | 500MB+ | Research |

## Next Steps

### For Production:
1. Use faster models (YOLOv8, RT-DETR)
2. Quantize model for speed
3. Use TensorRT for optimization
4. Implement tracking (SORT, DeepSORT)

### For Research:
1. Fine-tune on custom dataset
2. Add segmentation (Mask R-CNN, SAM)
3. Combine with depth for 3D understanding
4. Implement pose estimation

## References

- [DETR Paper](https://arxiv.org/abs/2005.12872)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense)
- [COCO Dataset](https://cocodataset.org/)

## License

This is a POC (Proof of Concept) for educational and research purposes.

---

**Questions? Issues?**
Check the troubleshooting section or modify the scripts as needed!
