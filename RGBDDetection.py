#!/usr/bin/env python3
"""
Working RGB-D Bottle Detection
Uses TRAINED SegFormer segmentation + converts masks to bounding boxes
This actually works because it uses your trained 96% IoU model!
"""

import torch
import numpy as np
import cv2
import pyrealsense2 as rs
from PIL import Image
import time
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


class WorkingRGBDDetector:
    """
    Working RGB-D detector using trained SegFormer
    Combines RGB segmentation + Depth information
    """
    
    def __init__(
        self,
        model_path,
        confidence_threshold=0.5,
        min_area=500,
        use_depth_filtering=True
    ):
        """
        Args:
            model_path: Path to trained SegFormer model
            confidence_threshold: Confidence threshold
            min_area: Minimum area for detection
            use_depth_filtering: Use depth to filter false positives
        """
        self.confidence_threshold = confidence_threshold
        self.min_area = min_area
        self.use_depth_filtering = use_depth_filtering
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load TRAINED SegFormer model
        print(f"Loading TRAINED model from: {model_path}")
        self.processor = SegformerImageProcessor.from_pretrained(model_path)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded (trained with 96% IoU!)")
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
            print(f"Serial: {dev.get_info(rs.camera_info.serial_number)}")
            
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
    
    def segment_to_boxes(self, mask, depth_image, rgb_image):
        """
        Convert segmentation mask to bounding boxes with depth filtering
        
        Args:
            mask: (H, W) binary mask
            depth_image: (H, W) depth in meters
            rgb_image: (H, W, 3) RGB image for visualization
            
        Returns:
            List of detections
        """
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        
        detections = []
        
        for i in range(1, num_labels):  # Skip background
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area < self.min_area:
                continue
            
            # Get bounding box
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Get depth at center and component
            cx, cy = int(centroids[i][0]), int(centroids[i][1])
            
            # Get median depth in component
            component_mask = (labels == i)
            depth_values = depth_image[component_mask]
            depth_values = depth_values[depth_values > 0]  # Filter invalid
            
            if len(depth_values) > 0:
                median_depth = np.median(depth_values)
                depth_std = np.std(depth_values)
            else:
                median_depth = 0.0
                depth_std = 0.0
            
            # Depth filtering (optional but helps reduce false positives)
            if self.use_depth_filtering:
                # Filter out objects too far (>3m) or too close (<0.1m)
                if median_depth < 0.1 or median_depth > 3.0:
                    continue
                
                # Filter out objects with very inconsistent depth (likely noise)
                if depth_std > 0.5:
                    continue
            
            # Compute confidence based on:
            # 1. How well component fills bounding box
            # 2. Depth consistency
            component_area = component_mask.sum()
            bbox_area = w * h
            fill_ratio = component_area / bbox_area if bbox_area > 0 else 0
            
            # Confidence from fill ratio
            confidence = fill_ratio
            
            # Boost confidence if depth is consistent
            if depth_std < 0.1:
                confidence = min(1.0, confidence * 1.2)
            
            detections.append({
                'label': 'bottle',
                'confidence': confidence,
                'bbox': [x, y, x + w, y + h],
                'depth': median_depth,
                'depth_std': depth_std,
                'area': area
            })
        
        return detections
    
    def detect(self, rgb_image, depth_image):
        """
        Run detection using TRAINED SegFormer
        
        Args:
            rgb_image: (H, W, 3) BGR
            depth_image: (H, W) depth in meters
            
        Returns:
            List of detections, segmentation mask
        """
        start_time = time.time()
        
        # Convert to PIL
        rgb_pil = Image.fromarray(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        
        # Process with TRAINED model
        inputs = self.processor(images=rgb_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference with TRAINED weights
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
            
            # Get confidence map
            probs = torch.nn.functional.softmax(upsampled_logits, dim=1)[0]
            bottle_prob_map = probs[1].cpu().numpy()  # Bottle class probability
        
        # Convert mask to boxes with depth filtering
        bottle_mask = (pred_seg == 1).astype(np.uint8)
        detections = self.segment_to_boxes(bottle_mask, depth_image, rgb_image)
        
        # Filter by confidence
        detections = [
            det for det in detections
            if det['confidence'] >= self.confidence_threshold
        ]
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        return detections, bottle_mask, bottle_prob_map
    
    def draw_detections(self, rgb_image, detections, show_depth=True):
        """Draw bounding boxes with depth info"""
        annotated = rgb_image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            depth = det['depth']
            depth_std = det['depth_std']
            
            # Color based on depth (green=close, red=far)
            if depth < 1.0:
                color = (0, 255, 0)  # Close - green
            elif depth < 2.0:
                color = (0, 255, 255)  # Medium - yellow
            else:
                color = (0, 165, 255)  # Far - orange
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Label
            if show_depth:
                text = f"Bottle {conf:.2f} | {depth:.2f}m±{depth_std:.2f}"
            else:
                text = f"Bottle: {conf:.2f}"
            
            # Background for text
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            
            cv2.putText(annotated, text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated
    
    def run_single_frame(self):
        """Process single frame"""
        print("\n🎯 Capturing single frame...")
        
        rgb_image, depth_image, depth_colormap = self.get_frames()
        
        if rgb_image is None:
            print("❌ Failed to get frames")
            return
        
        print("Running detection with TRAINED SegFormer...")
        detections, bottle_mask, prob_map = self.detect(rgb_image, depth_image)
        
        print(f"✅ Found {len(detections)} bottles:")
        for i, det in enumerate(detections, 1):
            print(f"  {i}. Confidence: {det['confidence']:.3f}, "
                  f"Depth: {det['depth']:.2f}m (±{det['depth_std']:.2f}), "
                  f"Area: {det['area']} px")
        
        # Visualize
        annotated = self.draw_detections(rgb_image, detections)
        
        # Create mask overlay
        mask_overlay = rgb_image.copy()
        mask_overlay[bottle_mask == 1] = mask_overlay[bottle_mask == 1] * 0.5 + np.array([0, 255, 0]) * 0.5
        
        # Probability heatmap
        prob_heatmap = cv2.applyColorMap((prob_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Create 2x2 grid
        top_row = np.hstack([annotated, mask_overlay])
        bottom_row = np.hstack([depth_colormap, prob_heatmap])
        display = np.vstack([top_row, bottom_row])
        
        # Add labels
        cv2.putText(display, "RGB + Boxes", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, "RGB + Mask", (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, "Depth Map", (10, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, "Confidence Map", (650, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('RGB-D Detection (TRAINED SegFormer)', display)
        cv2.imwrite('working_rgbd_detection.jpg', display)
        print("✅ Saved: working_rgbd_detection.jpg")
        
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def run_continuous(self):
        """Run continuous detection"""
        print("\n🎥 Running continuous RGB-D detection...")
        print("Press 'q' to quit, 's' to save, 'd' to toggle depth filtering")
        
        frame_count = 0
        last_detections = []
        last_mask = None
        
        try:
            while True:
                rgb_image, depth_image, depth_colormap = self.get_frames()
                
                if rgb_image is None:
                    continue
                
                # Run detection every 2 frames (faster)
                if frame_count % 2 == 0:
                    last_detections, last_mask, _ = self.detect(rgb_image, depth_image)
                
                # Draw
                annotated = self.draw_detections(rgb_image, last_detections)
                
                # Add info overlay
                if len(self.inference_times) > 0:
                    fps = 1.0 / np.mean(self.inference_times[-30:])
                    cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(annotated, f"Bottles: {len(last_detections)}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(annotated, f"Depth Filter: {'ON' if self.use_depth_filtering else 'OFF'}", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Display RGB + Depth side by side
                display = np.hstack([annotated, depth_colormap])
                cv2.imshow('RGB-D Detection', display)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"rgbd_detection_{timestamp}.jpg"
                    cv2.imwrite(filename, display)
                    print(f"Saved: {filename}")
                elif key == ord('d'):
                    self.use_depth_filtering = not self.use_depth_filtering
                    print(f"Depth filtering: {'ON' if self.use_depth_filtering else 'OFF'}")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            # Print summary
            if len(self.inference_times) > 0:
                avg_fps = 1.0 / np.mean(self.inference_times)
                print(f"\n📊 Average FPS: {avg_fps:.2f}")
                print(f"   Total frames: {frame_count}")
            
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
    print("Working RGB-D Bottle Detection")
    print("Uses TRAINED SegFormer (96% IoU) + Depth Filtering")
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
        print(f"  Depth Filtering: Enabled")
        print(f"    - Distance range: 0.1m - 3.0m")
        print(f"    - Max depth std: 0.5m")
        
        print("\nSelect mode:")
        print("1. Single frame (detailed visualization)")
        print("2. Continuous detection (real-time)")
        
        choice = input("Enter choice (1 or 2): ").strip()
        
        detector = WorkingRGBDDetector(
            model_path=MODEL_PATH,
            confidence_threshold=0.5,
            min_area=500,
            use_depth_filtering=True
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