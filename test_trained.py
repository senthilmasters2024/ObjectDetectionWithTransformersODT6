import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import cv2
import numpy as np

# Load model
model_path = "/home/frauas/ODT6/ObjectDetectionWithTransformersODT6/water_bottle_model/huggingface_model"
processor = DetrImageProcessor.from_pretrained(model_path)
model = DetrForObjectDetection.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print(f"Device: {device}")
print(f"Num labels: {model.config.num_labels}")
print(f"Label mapping: {model.config.id2label}")

# Load test image
test_image_path = "./my_dataset/rgb/rgb_00543_20260119_151951_950506.jpg"  # Change this!
image = Image.open(test_image_path).convert('RGB')
print(f"\nImage size: {image.size}")

# Process
inputs = processor(images=image, return_tensors="pt").to(device)

# Inference
with torch.no_grad():
    outputs = model(**inputs)
    print(f"\nRaw outputs:")
    print(f"  Logits shape: {outputs.logits.shape}")
    print(f"  Logits max: {outputs.logits.max().item():.4f}")
    print(f"  Logits min: {outputs.logits.min().item():.4f}")

# Try different thresholds
for threshold in [0.01, 0.05, 0.1, 0.3, 0.5]:
    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    results = processor.post_process_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=threshold
    )[0]
    
    print(f"\nThreshold {threshold}: {len(results['scores'])} detections")
    if len(results['scores']) > 0:
        for i, (score, label, box) in enumerate(zip(results['scores'], results['labels'], results['boxes'])):
            print(f"  {i+1}. {model.config.id2label[label.item()]}: {score.item():.4f}")