#!/usr/bin/env python3
"""
RealSense D435 + DETR Object Detection - FIXED FOR TRAINED MODEL
Uses trained bottle detection model
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import time
import json
import os


class MetricsTracker:
    """Track detection metrics in real-time and for evaluation"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.inference_times = []
        self.frame_count = 0
        self.total_detections = 0
        self.detections_per_class = {}
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0

    def update_inference(self, inference_time, detections):
        """Update metrics after each inference"""
        self.inference_times.append(inference_time)
        self.frame_count += 1
        self.total_detections += len(detections)

        for det in detections:
            label = det['label']
            if label not in self.detections_per_class:
                self.detections_per_class[label] = 0
            self.detections_per_class[label] += 1

    def get_fps(self):
        """Get current FPS based on recent inference times"""
        if len(self.inference_times) == 0:
            return 0.0
        recent = self.inference_times[-30:]
        avg_time = sum(recent) / len(recent)
        return 1.0 / avg_time if avg_time > 0 else 0.0

    def get_avg_inference_time(self):
        """Get average inference time in milliseconds"""
        if len(self.inference_times) == 0:
            return 0.0
        return (sum(self.inference_times) / len(self.inference_times)) * 1000

    def compute_iou(self, box1, box2):
        """Compute IoU between two boxes"""
        if len(box1) == 4 and box1[2] < box1[0] + box1[2]:
            b1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
        else:
            b1 = box1
        if len(box2) == 4 and box2[2] < box2[0] + box2[2]:
            b2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]
        else:
            b2 = box2

        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def evaluate_frame(self, detections, ground_truths, iou_threshold=0.5):
        """Evaluate detections against ground truth for one frame"""
        matched_gt = set()

        for det in detections:
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(ground_truths):
                if gt_idx in matched_gt:
                    continue
                if det['label'] != gt['label']:
                    continue

                iou = self.compute_iou(det['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                self.true_positives += 1
                matched_gt.add(best_gt_idx)
            else:
                self.false_positives += 1

        self.false_negatives += len(ground_truths) - len(matched_gt)

    def get_precision(self):
        """Calculate precision"""
        total = self.true_positives + self.false_positives
        return self.true_positives / total if total > 0 else 0.0

    def get_recall(self):
        """Calculate recall"""
        total = self.true_positives + self.false_negatives
        return self.true_positives / total if total > 0 else 0.0

    def get_f1_score(self):
        """Calculate F1 score"""
        precision = self.get_precision()
        recall = self.get_recall()
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def get_summary(self):
        """Get summary of all metrics"""
        return {
            'frames': self.frame_count,
            'avg_fps': self.get_fps(),
            'avg_inference_ms': self.get_avg_inference_time(),
            'total_detections': self.total_detections,
            'detections_per_class': self.detections_per_class,
            'precision': self.get_precision(),
            'recall': self.get_recall(),
            'f1_score': self.get_f1_score(),
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives
        }

    def print_summary(self):
        """Print formatted summary"""
        s = self.get_summary()
        print("\n" + "="*60)
        print("DETECTION METRICS SUMMARY")
        print("="*60)
        print(f"Frames processed:     {s['frames']}")
        print(f"Average FPS:          {s['avg_fps']:.2f}")
        print(f"Avg inference time:   {s['avg_inference_ms']:.2f} ms")
        print(f"Total detections:     {s['total_detections']}")
        print(f"\nDetections per class:")
        for cls, count in s['detections_per_class'].items():
            print(f"  - {cls}: {count}")
        if s['true_positives'] + s['false_positives'] + s['false_negatives'] > 0:
            print(f"\nEvaluation Metrics (IoU >= 0.7):")
            print(f"  Precision:  {s['precision']:.4f}")
            print(f"  Recall:     {s['recall']:.4f}")
            print(f"  F1 Score:   {s['f1_score']:.4f}")
            print(f"  TP: {s['true_positives']} | FP: {s['false_positives']} | FN: {s['false_negatives']}")
        print("="*60)


class RealsenseDETR:
    def __init__(self, confidence_threshold=0.99):
        """
        Initialize RealSense camera and DETR model
        
        Args:
            confidence_threshold: Minimum confidence for detections (0-1)
        """
        print("Initializing DETR Object Detection...")
        
        self.confidence_threshold = confidence_threshold
        
        # Load trained bottle detection model
        print("Loading trained bottle detection model...")
        
        # ABSOLUTE PATH to your trained model
        model_path = "/home/frauas/ODT6/ObjectDetectionWithTransformersODT6/water_bottle_model/huggingface_model"
        
        # Verify model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model not found at: {model_path}")
        
        self.processor = DetrImageProcessor.from_pretrained(model_path)
        self.model = DetrForObjectDetection.from_pretrained(model_path)
        
        print(f"✅ Model loaded successfully!")
        print(f"   Num labels: {self.model.config.num_labels}")
        print(f"   Label mapping: {self.model.config.id2label}")
        print(f"   Confidence threshold: {self.confidence_threshold}")

        # Check GPU
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            try:
                test_tensor = torch.zeros(1).cuda()
                _ = test_tensor + 1
                del test_tensor
                torch.cuda.empty_cache()
                self.device = torch.device("cuda")
                print(f"   Using device: cuda ({torch.cuda.get_device_name(0)})")
            except RuntimeError as e:
                print(f"   CUDA error: {e}, falling back to CPU")
                self.device = torch.device("cpu")
        else:
            print("   Using device: cpu")

        self.model.to(self.device)
        self.model.eval()
        
        # Initialize RealSense pipeline
        self.pipeline = None
        self._cleaned_up = False
        self.initialize_camera()

        # Initialize metrics tracker
        self.metrics = MetricsTracker()
    
    def initialize_camera(self):
        """Initialize RealSense camera"""
        print("\nInitializing RealSense D435...")
        
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            
            if len(devices) == 0:
                raise RuntimeError("No RealSense devices found!")
            
            print(f"Found {len(devices)} RealSense device(s)")
            
            dev = devices[0]
            print(f"Device: {dev.get_info(rs.camera_info.name)}")
            print(f"Serial: {dev.get_info(rs.camera_info.serial_number)}")
            print(f"Firmware: {dev.get_info(rs.camera_info.firmware_version)}")
            
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(dev.get_info(rs.camera_info.serial_number))
            
            resolutions = [
                (640, 480, 30),
                (848, 480, 30),
                (1280, 720, 30),
            ]
            
            camera_started = False
            
            for width, height, fps in resolutions:
                try:
                    print(f"\nTrying resolution: {width}x{height} @ {fps}fps...")
                    
                    config = rs.config()
                    config.enable_device(dev.get_info(rs.camera_info.serial_number))
                    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
                    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
                    
                    profile = self.pipeline.start(config)
                    
                    print(f"✅ Camera started at {width}x{height}")
                    camera_started = True
                    break
                    
                except Exception as e:
                    print(f"Failed with {width}x{height}: {str(e)}")
                    continue
            
            if not camera_started:
                raise RuntimeError("Failed to start camera")
            
            # Warm up
            print("\nWarming up camera...")
            for i in range(30):
                try:
                    self.pipeline.wait_for_frames(timeout_ms=5000)
                except RuntimeError:
                    continue
            
            print("✅ RealSense D435 ready!")
            
        except Exception as e:
            print(f"\n❌ Error initializing camera: {str(e)}")
            raise
    
    def get_frame(self, timeout_ms=5000):
        """Get color and depth frames from RealSense"""
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return None, None
            
            color_image = np.asanyarray(color_frame.get_data())
            
            return color_image, depth_frame
            
        except RuntimeError as e:
            print(f"Warning: Frame timeout - {str(e)}")
            return None, None
    
    def detect_objects(self, image):
        """
        Run DETR object detection on image

        Args:
            image: BGR image (numpy array)

        Returns:
            List of detections
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Preprocess
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

            # Post-process
            target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)
            results = self.processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=self.confidence_threshold
            )[0]

            # Extract detections - USE MODEL'S LABEL MAPPING
            detections = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                # ✅ FIXED: Use model's label mapping
                class_name = self.model.config.id2label.get(label.item(), 'unknown')
                
                detections.append({
                    'label': class_name,
                    'confidence': score.item(),
                    'bbox': box.cpu().numpy()
                })

        # Clear GPU memory
        del inputs, outputs, target_sizes, results
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return detections
    
    def get_depth_at_bbox(self, depth_frame, bbox):
        """Get median depth within bounding box"""
        xmin, ymin, xmax, ymax = bbox.astype(int)
        
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(depth_frame.get_width(), xmax)
        ymax = min(depth_frame.get_height(), ymax)
        
        center_x = (xmin + xmax) // 2
        center_y = (ymin + ymax) // 2
        sample_size = 10
        
        depths = []
        for dx in range(-sample_size, sample_size):
            for dy in range(-sample_size, sample_size):
                x = center_x + dx
                y = center_y + dy
                if 0 <= x < depth_frame.get_width() and 0 <= y < depth_frame.get_height():
                    depth = depth_frame.get_distance(x, y)
                    if depth > 0:
                        depths.append(depth)
        
        if depths:
            return np.median(depths)
        return 0.0
    
    def draw_detections(self, image, detections, depth_frame=None):
        """Draw bounding boxes and labels on image"""
        annotated = image.copy()
        
        for det in detections:
            label = det['label']
            confidence = det['confidence']
            bbox = det['bbox']
            
            xmin, ymin, xmax, ymax = bbox.astype(int)
            
            # Get depth if available
            depth_text = ""
            if depth_frame is not None:
                depth = self.get_depth_at_bbox(depth_frame, bbox)
                if depth > 0:
                    depth_text = f" | {depth:.2f}m"
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), color, 2)
            
            # Draw label
            text = f"{label}: {confidence:.2f}{depth_text}"
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                annotated, 
                (xmin, ymin - text_height - 10), 
                (xmin + text_width, ymin), 
                color, 
                -1
            )
            
            cv2.putText(
                annotated, 
                text, 
                (xmin, ymin - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (0, 0, 0), 
                2
            )
        
        return annotated
    
    def run_single_frame(self, save_path="detection_result.jpg"):
        """Process single frame and save result"""
        print("\n🎯 Running single frame detection...")
        
        # Get frame
        for attempt in range(5):
            print(f"Capturing frame ({attempt+1}/5)...")
            color_image, depth_frame = self.get_frame(timeout_ms=10000)
            
            if color_image is not None:
                print("✅ Frame captured")
                break
            time.sleep(1)
        
        if color_image is None:
            print("❌ Failed to get frame")
            return
        
        # Run detection
        print("Running detection...")
        start_time = time.time()
        detections = self.detect_objects(color_image)
        inference_time = time.time() - start_time
        
        print(f"✅ Detection complete in {inference_time:.2f}s")
        print(f"Found {len(detections)} objects:")
        
        for i, det in enumerate(detections, 1):
            depth = self.get_depth_at_bbox(depth_frame, det['bbox'])
            depth_str = f"{depth:.2f}m" if depth > 0 else "N/A"
            print(f"  {i}. {det['label']}: {det['confidence']:.3f} | Distance: {depth_str}")
        
        # Draw detections
        annotated = self.draw_detections(color_image, detections, depth_frame)
        
        # Add info
        info_text = f"DETR | {len(detections)} objects | {inference_time:.2f}s"
        cv2.putText(annotated, info_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save
        cv2.imwrite(save_path, annotated)
        print(f"✅ Saved: {save_path}")
        
        # Display
        cv2.imshow('DETR Detection Result', annotated)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def draw_metrics_overlay(self, image):
        """Draw metrics overlay on image"""
        h, w = image.shape[:2]

        overlay = image.copy()
        cv2.rectangle(overlay, (w - 220, 0), (w, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        fps = self.metrics.get_fps()
        avg_ms = self.metrics.get_avg_inference_time()
        frames = self.metrics.frame_count

        cv2.putText(image, f"FPS: {fps:.1f}", (w - 210, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(image, f"Inference: {avg_ms:.1f}ms", (w - 210, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(image, f"Frames: {frames}", (w - 210, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(image, f"Detections: {self.metrics.total_detections}", (w - 210, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return image

    def run_continuous(self):
        """Run continuous object detection"""
        print("\n🎥 Running continuous detection...")
        print("Press 'q' to quit, 's' to save screenshot, 'm' to print metrics")

        self.metrics.reset()
        frame_count = 0
        last_detections = []

        try:
            while True:
                color_image, depth_frame = self.get_frame(timeout_ms=5000)

                if color_image is None:
                    continue

                # Run detection every 3 frames
                if frame_count % 3 == 0:
                    start_time = time.time()
                    detections = self.detect_objects(color_image)
                    inference_time = time.time() - start_time
                    last_detections = detections

                    self.metrics.update_inference(inference_time, detections)

                    annotated = self.draw_detections(color_image, detections, depth_frame)

                    fps = self.metrics.get_fps()
                    info_text = f"DETR | {len(detections)} objects | {fps:.1f} FPS"
                    cv2.putText(annotated, info_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    annotated = self.draw_metrics_overlay(annotated)
                else:
                    annotated = self.draw_detections(color_image, last_detections, depth_frame)
                    fps = self.metrics.get_fps()
                    cv2.putText(annotated, f"DETR | {len(last_detections)} objects | {fps:.1f} FPS", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    annotated = self.draw_metrics_overlay(annotated)

                cv2.imshow('RealSense + DETR', annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"detection_{timestamp}.jpg"
                    cv2.imwrite(filename, annotated)
                    print(f"Screenshot saved: {filename}")
                elif key == ord('m'):
                    self.metrics.print_summary()

                frame_count += 1

        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            self.metrics.print_summary()
            self.cleanup()

    def cleanup(self):
        """Stop pipeline and close windows"""
        if self._cleaned_up:
            return
        self._cleaned_up = True
        print("\nCleaning up...")
        if self.pipeline:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass
        cv2.destroyAllWindows()
        print("✅ Done!")


def main():
    """Main function"""
    print("="*60)
    print("RealSense D435 + Trained DETR Bottle Detection")
    print("="*60)

    try:
        print("\nSelect mode:")
        print("1. Single frame detection")
        print("2. Continuous detection")

        choice = input("Enter choice (1 or 2): ").strip()

        detector = RealsenseDETR(confidence_threshold=0.99)

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