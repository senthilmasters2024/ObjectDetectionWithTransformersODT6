#!/usr/bin/env python3
"""
Simplified RGB-D Detection using Trained SegFormer Segmentation
Converts segmentation masks to bounding boxes
Simpler approach that works with your existing trained model
"""

import torch
import numpy as np
import cv2
import pyrealsense2 as rs
from PIL import Image
import time
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


class SegFormerBottleDetector:
    """
    Bottle detection using SegFormer segmentation
    Converts pixel masks to bounding boxes
    """
    
    def __init__(
        self,
        model_path,
        confidence_threshold=0.5,
        min_area=500  # Minimum blob area in pixels
    ):
        """
        Args:
            model_path: Path to trained SegFormer model
            confidence_threshold: Confidence threshold for segmentation
            min_area: Minimum area for valid detection
        """
        self.confidence_threshold = confidence_threshold
        self.min_area = min_area
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model
        print(f"Loading model from: {model_path}")
        self.processor = SegformerImageProcessor.from_pretrained(model_path)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded")
        print(f"   Num labels: {self.model.config.num_labels}")
        print(f"   Label mapping: {self.model.config.id2label}")
        
        # RealSense
        self.pipeline = None
        self._cleaned_up = False
        self.initialize_camera()
        
        # Metrics
        self.inference_times = []
        
        print("✅ Detector ready!")
    
    def initialize_camera(self):
        """Initialize RealSense D435"""
        print("\nInitializing RealSense D435...")
        
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            
            if len(devices) == 0:
                raise RuntimeError("No RealSense devices found!")
            
            dev = devices[0]
            print(f"Device: {dev.get_info(rs.camera_info.name)}")
            
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(dev.get_info(rs.camera_info.serial_number))
            
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            
            profile = self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)
            
            print("✅ Camera started at 640x480")
            
            # Warm up
            for _ in range(30):
                self.pipeline.wait_for_frames(timeout_ms=5000)
            
            print("✅ Camera ready!")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            raise
    
    def get_frames(self):
        """Get aligned RGB and Depth frames"""
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            aligned_frames = self.align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return None, None, None
            
            rgb_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data()) * 0.001  # mm to m
            
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image * 1000, alpha=0.03),
                cv2.COLORMAP_JET
            )
            
            return rgb_image, depth_image, depth_colormap
            
        except Exception as e:
            return None, None, None
    
    def segment_to_boxes(self, mask, depth_image):
        """
        Convert segmentation mask to bounding boxes
        
        Args:
            mask: (H, W) binary mask where 1 = bottle
            depth_image: (H, W) depth in meters
            
        Returns:
            List of detections
        """
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        
        detections = []
        
        for i in range(1, num_labels):  # Skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area < self.min_area:
                continue
            
            # Get bounding box
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Get center depth
            cx, cy = int(centroids[i][0]), int(centroids[i][1])
            if 0 <= cy < depth_image.shape[0] and 0 <= cx < depth_image.shape[1]:
                depth = depth_image[cy, cx]
            else:
                depth = 0.0
            
            # Compute confidence (IoU of component vs its bbox)
            component_mask = (labels == i).astype(np.uint8)
            bbox_mask = np.zeros_like(component_mask)
            bbox_mask[y:y+h, x:x+w] = 1
            
            intersection = (component_mask & bbox_mask).sum()
            union = (component_mask | bbox_mask).sum()
            confidence = intersection / union if union > 0 else 0.0
            
            detections.append({
                'label': 'bottle',
                'confidence': confidence,
                'bbox': [x, y, x + w, y + h],
                'depth': depth,
                'area': area
            })
        
        return detections
    
    def detect(self, rgb_image, depth_image):
        """
        Run detection on RGB image
        
        Args:
            rgb_image: (H, W, 3) BGR
            depth_image: (H, W) depth in meters
            
        Returns:
            List of detections
        """
        start_time = time.time()
        
        # Convert to PIL
        rgb_pil = Image.fromarray(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        
        # Process
        inputs = self.processor(images=rgb_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Resize to original size
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=rgb_image.shape[:2],
                mode="bilinear",
                align_corners=False
            )
            
            # Get segmentation mask
            pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
            
            # Get confidence
            probs = torch.nn.functional.softmax(upsampled_logits, dim=1)[0]
            bottle_prob = probs[1].cpu().numpy()  # Bottle class
        
        # Convert mask to boxes
        bottle_mask = (pred_seg == 1).astype(np.uint8)
        detections = self.segment_to_boxes(bottle_mask, depth_image)
        
        # Filter by confidence
        detections = [
            det for det in detections
            if det['confidence'] >= self.confidence_threshold
        ]
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        return detections, bottle_mask
    
    def draw_detections(self, rgb_image, detections):
        """Draw bounding boxes"""
        annotated = rgb_image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            depth = det['depth']
            
            color = (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            text = f"Bottle: {conf:.2f} | {depth:.2f}m"
            cv2.putText(annotated, text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated
    
    def run_single_frame(self):
        """Process single frame"""
        print("\n🎯 Capturing single frame...")
        
        rgb_image, depth_image, depth_colormap = self.get_frames()
        
        if rgb_image is None:
            print("❌ Failed to get frames")
            return
        
        print("Running detection...")
        detections, bottle_mask = self.detect(rgb_image, depth_image)
        
        print(f"✅ Found {len(detections)} bottles:")
        for i, det in enumerate(detections, 1):
            print(f"  {i}. Confidence: {det['confidence']:.3f}, "
                  f"Depth: {det['depth']:.2f}m, Area: {det['area']} px")
        
        # Visualize
        annotated = self.draw_detections(rgb_image, detections)
        
        # Create mask overlay
        mask_overlay = rgb_image.copy()
        mask_overlay[bottle_mask == 1] = mask_overlay[bottle_mask == 1] * 0.5 + np.array([0, 255, 0]) * 0.5
        
        # Create display
        top_row = np.hstack([annotated, mask_overlay])
        bottom_row = np.hstack([depth_colormap, cv2.cvtColor(bottle_mask * 255, cv2.COLOR_GRAY2BGR)])
        display = np.vstack([top_row, bottom_row])
        
        cv2.imshow('Detection (RGB | Mask Overlay | Depth | Segmentation)', display)
        cv2.imwrite('segformer_detection_result.jpg', display)
        print("✅ Saved: segformer_detection_result.jpg")
        
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def run_continuous(self):
        """Run continuous detection"""
        print("\n🎥 Running continuous detection...")
        print("Press 'q' to quit, 's' to save screenshot")
        
        frame_count = 0
        last_detections = []
        last_mask = None
        
        try:
            while True:
                rgb_image, depth_image, depth_colormap = self.get_frames()
                
                if rgb_image is None:
                    continue
                
                # Run detection every 3 frames
                if frame_count % 3 == 0:
                    last_detections, last_mask = self.detect(rgb_image, depth_image)
                
                # Draw
                annotated = self.draw_detections(rgb_image, last_detections)
                
                # Add FPS
                if len(self.inference_times) > 0:
                    fps = 1.0 / np.mean(self.inference_times[-30:])
                    cv2.putText(annotated, f"SegFormer FPS: {fps:.1f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(annotated, f"Bottles: {len(last_detections)}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display
                cv2.imshow('SegFormer Detection', annotated)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"segformer_detection_{timestamp}.jpg"
                    cv2.imwrite(filename, annotated)
                    print(f"Saved: {filename}")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup"""
        if self._cleaned_up:
            return
        self._cleaned_up = True
        
        print("\nCleaning up...")
        if self.pipeline:
            try:
                self.pipeline.stop()
            except:
                pass
        cv2.destroyAllWindows()
        print("✅ Done!")


def main():
    print("="*60)
    print("SegFormer Bottle Detection (Segmentation → Boxes)")
    print("="*60)
    
    # UPDATE THIS PATH!
    MODEL_PATH = "/home/frauas/ODT6/ODTSegformer/ObjectDetectionWithTransformersODT6/water_bottle_segformer_model/huggingface_model"
    
    import os
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at: {MODEL_PATH}")
        print("Please update MODEL_PATH in the script")
        return
    
    try:
        print("\nConfiguration:")
        print(f"  Model: {MODEL_PATH}")
        print(f"  Confidence Threshold: 0.5")
        print(f"  Min Area: 500 pixels")
        
        print("\nSelect mode:")
        print("1. Single frame")
        print("2. Continuous detection")
        
        choice = input("Enter choice (1 or 2): ").strip()
        
        detector = SegFormerBottleDetector(
            model_path=MODEL_PATH,
            confidence_threshold=0.5,
            min_area=500
        )
        
        if choice == "1":
            detector.run_single_frame()
        else:
            detector.run_continuous()
        
        detector.cleanup()
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()