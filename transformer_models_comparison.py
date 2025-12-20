"""
Comparison of Different Transformer Models for Object Detection
Supports: DETR, DINO, RT-DETR, YOLOv8 (for comparison)
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from transformers import (
    DetrImageProcessor, DetrForObjectDetection,
    AutoImageProcessor, AutoModelForObjectDetection
)
from PIL import Image
import time

# COCO class names
COCO_CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class TransformerDetector:
    """
    Unified interface for different transformer detection models
    """
    
    MODELS = {
        'detr-resnet50': {
            'name': 'DETR (ResNet-50)',
            'model_id': 'facebook/detr-resnet-50',
            'description': 'Original DETR, good accuracy, moderate speed'
        },
        'detr-resnet101': {
            'name': 'DETR (ResNet-101)',
            'model_id': 'facebook/detr-resnet-101',
            'description': 'DETR with larger backbone, better accuracy, slower'
        },
        'conditional-detr': {
            'name': 'Conditional DETR',
            'model_id': 'microsoft/conditional-detr-resnet-50',
            'description': 'Faster convergence, better for fine-tuning'
        },
        'table-transformer': {
            'name': 'Table Transformer',
            'model_id': 'microsoft/table-transformer-detection',
            'description': 'Specialized for table detection'
        },
        'deta': {
            'name': 'DETA',
            'model_id': 'jozhang97/deta-swin-large-o365',
            'description': 'State-of-the-art, very accurate, slow'
        }
    }
    
    def __init__(self, model_key='detr-resnet50', confidence_threshold=0.7):
        """
        Initialize transformer detection model
        
        Args:
            model_key: Key from MODELS dict
            confidence_threshold: Minimum confidence (0-1)
        """
        if model_key not in self.MODELS:
            raise ValueError(f"Unknown model: {model_key}. Choose from: {list(self.MODELS.keys())}")
        
        self.model_info = self.MODELS[model_key]
        self.confidence_threshold = confidence_threshold
        
        print(f"\n{'='*60}")
        print(f"Loading: {self.model_info['name']}")
        print(f"Description: {self.model_info['description']}")
        print(f"{'='*60}\n")
        
        # Load model
        model_id = self.model_info['model_id']
        print(f"Loading from: {model_id}")
        
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModelForObjectDetection.from_pretrained(model_id)
        except:
            # Fallback to DETR specific
            from transformers import DetrImageProcessor, DetrForObjectDetection
            self.processor = DetrImageProcessor.from_pretrained(model_id)
            self.model = DetrForObjectDetection.from_pretrained(model_id)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ Model loaded on: {self.device}\n")
    
    def detect(self, image):
        """
        Run detection on image
        
        Args:
            image: BGR numpy array
            
        Returns:
            List of detections
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Preprocess
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process
        target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, 
            target_sizes=target_sizes, 
            threshold=self.confidence_threshold
        )[0]
        
        # Extract detections
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label_idx = label.item()
            if label_idx < len(COCO_CLASSES):
                label_name = COCO_CLASSES[label_idx]
            else:
                label_name = f"class_{label_idx}"
            
            detections.append({
                'label': label_name,
                'confidence': score.item(),
                'bbox': box.cpu().numpy()
            })
        
        return detections


def benchmark_models(image_path=None):
    """
    Benchmark different transformer models
    
    Args:
        image_path: Optional path to test image
    """
    # Get test image
    if image_path:
        image = cv2.imread(image_path)
    else:
        # Capture from RealSense
        print("Capturing test frame from RealSense...")
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        pipeline.start(config)
        
        # Warm up
        for _ in range(30):
            pipeline.wait_for_frames()
        
        # Capture
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        image = np.asanyarray(color_frame.get_data())
        pipeline.stop()
        
        # Save test image
        cv2.imwrite('test_frame.jpg', image)
        print("✅ Test frame saved as test_frame.jpg\n")
    
    # Test models
    models_to_test = ['detr-resnet50', 'conditional-detr']
    
    results = {}
    
    for model_key in models_to_test:
        print(f"\n{'='*60}")
        print(f"Testing: {model_key}")
        print(f"{'='*60}")
        
        try:
            detector = TransformerDetector(model_key, confidence_threshold=0.7)
            
            # Warm up
            print("Warming up...")
            _ = detector.detect(image)
            
            # Benchmark
            print("Running benchmark (3 iterations)...")
            times = []
            for i in range(3):
                start = time.time()
                detections = detector.detect(image)
                elapsed = time.time() - start
                times.append(elapsed)
                print(f"  Iteration {i+1}: {elapsed:.2f}s | {len(detections)} objects")
            
            avg_time = np.mean(times)
            
            # Draw results
            annotated = image.copy()
            for det in detections:
                bbox = det['bbox'].astype(int)
                label = f"{det['label']}: {det['confidence']:.2f}"
                cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(annotated, label, (bbox[0], bbox[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save
            output_path = f"result_{model_key}.jpg"
            cv2.imwrite(output_path, annotated)
            
            results[model_key] = {
                'avg_time': avg_time,
                'detections': len(detections),
                'output': output_path
            }
            
            print(f"✅ Results saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error with {model_key}: {str(e)}")
            results[model_key] = {'error': str(e)}
    
    # Summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}\n")
    
    print(f"{'Model':<25} {'Avg Time':<12} {'Objects':<10} {'Output'}")
    print("-" * 70)
    
    for model_key, result in results.items():
        if 'error' in result:
            print(f"{model_key:<25} ERROR: {result['error']}")
        else:
            print(f"{model_key:<25} {result['avg_time']:.2f}s{'':<7} {result['detections']:<10} {result['output']}")


def main():
    """Main function"""
    print("\n" + "="*60)
    print("Transformer Object Detection Models Comparison")
    print("="*60 + "\n")
    
    print("Available models:")
    for i, (key, info) in enumerate(TransformerDetector.MODELS.items(), 1):
        print(f"{i}. {info['name']}")
        print(f"   {info['description']}\n")
    
    print("\nOptions:")
    print("1. Run single model test")
    print("2. Benchmark multiple models")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "2":
        benchmark_models()
    else:
        print("\nSelect model:")
        for i, key in enumerate(TransformerDetector.MODELS.keys(), 1):
            print(f"{i}. {key}")
        
        model_idx = int(input("\nEnter model number: ").strip()) - 1
        model_key = list(TransformerDetector.MODELS.keys())[model_idx]
        
        # Capture frame
        print("\nCapturing from RealSense...")
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        pipeline.start(config)
        
        for _ in range(30):
            pipeline.wait_for_frames()
        
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        image = np.asanyarray(color_frame.get_data())
        pipeline.stop()
        
        # Detect
        detector = TransformerDetector(model_key)
        print("\nRunning detection...")
        detections = detector.detect(image)
        
        print(f"\nFound {len(detections)} objects:")
        for i, det in enumerate(detections, 1):
            print(f"{i}. {det['label']}: {det['confidence']:.2f}")
        
        # Draw and save
        annotated = image.copy()
        for det in detections:
            bbox = det['bbox'].astype(int)
            label = f"{det['label']}: {det['confidence']:.2f}"
            cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(annotated, label, (bbox[0], bbox[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imwrite('detection_result.jpg', annotated)
        print("\n✅ Result saved to: detection_result.jpg")
        
        cv2.imshow('Result', annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
