#!/usr/bin/env python3
"""
RealSense D435 + DUAL DETR Models - ENSEMBLE DETECTION
Uses BOTH positive-trained and hard-negative-trained models for better accuracy
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
        self.model_agreement_count = 0
        self.model1_only = 0
        self.model2_only = 0

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

    def get_summary(self):
        """Get summary of all metrics"""
        return {
            'frames': self.frame_count,
            'avg_fps': self.get_fps(),
            'avg_inference_ms': self.get_avg_inference_time(),
            'total_detections': self.total_detections,
            'detections_per_class': self.detections_per_class,
            'model_agreement': self.model_agreement_count,
            'model1_only': self.model1_only,
            'model2_only': self.model2_only
        }

    def print_summary(self):
        """Print formatted summary"""
        s = self.get_summary()
        print("\n" + "="*60)
        print("DUAL-MODEL DETECTION METRICS SUMMARY")
        print("="*60)
        print(f"Frames processed:     {s['frames']}")
        print(f"Average FPS:          {s['avg_fps']:.2f}")
        print(f"Avg inference time:   {s['avg_inference_ms']:.2f} ms")
        print(f"Total detections:     {s['total_detections']}")
        print(f"\nModel Agreement:")
        print(f"  Both models agree:  {s['model_agreement']}")
        print(f"  Model 1 only:       {s['model1_only']}")
        print(f"  Model 2 only:       {s['model2_only']}")
        print(f"\nDetections per class:")
        for cls, count in s['detections_per_class'].items():
            print(f"  - {cls}: {count}")
        print("="*60)


class DualModelRealsenseDETR:
    def __init__(self, 
                 confidence_threshold_1=0.99, 
                 confidence_threshold_2=0.99,
                 ensemble_strategy="voting"):
        """
        Initialize RealSense camera and DUAL DETR models
        
        Args:
            confidence_threshold_1: Threshold for model 1 (positive-trained)
            confidence_threshold_2: Threshold for model 2 (hard-negative-trained)
            ensemble_strategy: "voting", "intersection", "union", or "weighted"
        """
        print("="*70)
        print("INITIALIZING DUAL-MODEL DETR DETECTION")
        print("="*70)
        
        self.confidence_threshold_1 = confidence_threshold_1
        self.confidence_threshold_2 = confidence_threshold_2
        self.ensemble_strategy = ensemble_strategy
        
        print(f"\nEnsemble Strategy: {ensemble_strategy}")
        print(f"Model 1 Threshold: {confidence_threshold_1}")
        print(f"Model 2 Threshold: {confidence_threshold_2}")
        
        # Check GPU
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            try:
                test_tensor = torch.zeros(1).cuda()
                _ = test_tensor + 1
                del test_tensor
                torch.cuda.empty_cache()
                self.device = torch.device("cuda")
                print(f"\n🖥️  Using device: cuda ({torch.cuda.get_device_name(0)})")
            except RuntimeError as e:
                print(f"\n⚠️  CUDA error: {e}, falling back to CPU")
                self.device = torch.device("cpu")
        else:
            print("\n🖥️  Using device: cpu")
        
        # Load MODEL 1: Positive-trained (good at detecting bottles)
        print("\n" + "="*70)
        print("LOADING MODEL 1: POSITIVE-TRAINED (Bottle Detection)")
        print("="*70)
        
        model1_path = "/home/frauas/ODT6/ObjectDetectionWithTransformersODT6/water_bottle_model/huggingface_model"
        
        if not os.path.exists(model1_path):
            # Try relative path
            model1_path = "./water_bottle_model/huggingface_model"
        
        if not os.path.exists(model1_path):
            raise FileNotFoundError(f"❌ Model 1 not found at: {model1_path}")
        
        print(f"Loading from: {model1_path}")
        self.processor_1 = DetrImageProcessor.from_pretrained(model1_path)
        self.model_1 = DetrForObjectDetection.from_pretrained(model1_path)
        self.model_1.to(self.device)
        self.model_1.eval()
        
        print(f"✅ Model 1 loaded")
        print(f"   Labels: {self.model_1.config.id2label}")
        
        # Load MODEL 2: Hard-negative-trained (good at rejecting false positives)
        print("\n" + "="*70)
        print("LOADING MODEL 2: HARD-NEGATIVE-TRAINED (False Positive Reduction)")
        print("="*70)
        
        model2_path = "/home/frauas/ODT6/ObjectDetectionWithTransformersODT6/water_bottle_model_hard_negatives/huggingface_model"
        
        if not os.path.exists(model2_path):
            # Try relative path
            model2_path = "./water_bottle_model_hard_negatives/huggingface_model"
        
        if not os.path.exists(model2_path):
            raise FileNotFoundError(f"❌ Model 2 not found at: {model2_path}")
        
        print(f"Loading from: {model2_path}")
        self.processor_2 = DetrImageProcessor.from_pretrained(model2_path)
        self.model_2 = DetrForObjectDetection.from_pretrained(model2_path)
        self.model_2.to(self.device)
        self.model_2.eval()
        
        print(f"✅ Model 2 loaded")
        print(f"   Labels: {self.model_2.config.id2label}")
        
        print("\n✅ BOTH MODELS READY!")
        
        # Initialize RealSense pipeline
        self.pipeline = None
        self._cleaned_up = False
        self.initialize_camera()

        # Initialize metrics tracker
        self.metrics = MetricsTracker()
    
    def initialize_camera(self):
        """Initialize RealSense camera"""
        print("\n" + "="*70)
        print("INITIALIZING REALSENSE D435")
        print("="*70)
        
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
    
    def detect_with_model(self, image, model, processor, threshold):
        """
        Run detection with a single model
        
        Returns:
            List of detections
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Preprocess
        inputs = processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)

            # Post-process
            target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)
            results = processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=threshold
            )[0]

            # Extract detections
            detections = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                class_name = model.config.id2label.get(label.item(), 'unknown')
                
                # Skip no-object class
                if class_name in ['no-object', 'background']:
                    continue
                
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
    
    def compute_iou(self, box1, box2):
        """Compute IoU between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Intersection
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)
        
        if xi_max < xi_min or yi_max < yi_min:
            return 0.0
        
        intersection = (xi_max - xi_min) * (yi_max - yi_min)
        
        # Union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def ensemble_detections(self, detections_1, detections_2):
        """
        Combine detections from both models using ensemble strategy
        
        Strategies:
        - "voting": Keep detections where both models agree (high precision)
        - "intersection": Keep only boxes detected by both (very conservative)
        - "union": Keep all detections from both (high recall)
        - "weighted": Average confidence scores for overlapping boxes
        """
        if self.ensemble_strategy == "union":
            # Simply combine all detections
            all_detections = detections_1 + detections_2
            # Remove duplicates by IoU
            return self.nms_detections(all_detections, iou_threshold=0.5)
        
        elif self.ensemble_strategy == "intersection":
            # Only keep detections that appear in both models
            final_detections = []
            
            for det1 in detections_1:
                for det2 in detections_2:
                    iou = self.compute_iou(det1['bbox'], det2['bbox'])
                    
                    if iou > 0.5:  # Boxes overlap significantly
                        # Take average of both
                        final_detections.append({
                            'label': det1['label'],
                            'confidence': (det1['confidence'] + det2['confidence']) / 2,
                            'bbox': (det1['bbox'] + det2['bbox']) / 2,
                            'source': 'both'
                        })
                        self.metrics.model_agreement_count += 1
            
            return final_detections
        
        elif self.ensemble_strategy == "weighted":
            # Combine with weighted averaging
            final_detections = []
            used_det2 = set()
            
            for det1 in detections_1:
                best_match = None
                best_iou = 0
                best_idx = -1
                
                for idx, det2 in enumerate(detections_2):
                    if idx in used_det2:
                        continue
                    iou = self.compute_iou(det1['bbox'], det2['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_match = det2
                        best_idx = idx
                
                if best_iou > 0.3:  # Found matching detection
                    # Weighted average (model 2 has higher weight for confidence)
                    w1 = 0.4  # Weight for model 1
                    w2 = 0.6  # Weight for model 2 (trained on hard negatives)
                    
                    final_detections.append({
                        'label': det1['label'],
                        'confidence': w1 * det1['confidence'] + w2 * best_match['confidence'],
                        'bbox': (det1['bbox'] + best_match['bbox']) / 2,
                        'source': 'both'
                    })
                    used_det2.add(best_idx)
                    self.metrics.model_agreement_count += 1
                else:
                    # Only model 1 detected it
                    final_detections.append({
                        'label': det1['label'],
                        'confidence': det1['confidence'] * 0.8,  # Lower confidence
                        'bbox': det1['bbox'],
                        'source': 'model1'
                    })
                    self.metrics.model1_only += 1
            
            # Add detections only from model 2
            for idx, det2 in enumerate(detections_2):
                if idx not in used_det2:
                    final_detections.append({
                        'label': det2['label'],
                        'confidence': det2['confidence'] * 0.9,  # Slightly lower confidence
                        'bbox': det2['bbox'],
                        'source': 'model2'
                    })
                    self.metrics.model2_only += 1
            
            return final_detections
        
        else:  # "voting" (default)
            # Keep detections where both models roughly agree
            final_detections = []
            used_det2 = set()
            
            for det1 in detections_1:
                found_match = False
                for idx, det2 in enumerate(detections_2):
                    if idx in used_det2:
                        continue
                    iou = self.compute_iou(det1['bbox'], det2['bbox'])
                    
                    if iou > 0.4:  # Models agree on this detection
                        final_detections.append({
                            'label': det1['label'],
                            'confidence': max(det1['confidence'], det2['confidence']),
                            'bbox': (det1['bbox'] + det2['bbox']) / 2,
                            'source': 'both'
                        })
                        used_det2.add(idx)
                        found_match = True
                        self.metrics.model_agreement_count += 1
                        break
                
                # If no match, still include if confidence is very high
                if not found_match and det1['confidence'] > 0.95:
                    final_detections.append({
                        'label': det1['label'],
                        'confidence': det1['confidence'],
                        'bbox': det1['bbox'],
                        'source': 'model1'
                    })
                    self.metrics.model1_only += 1
            
            # Add high-confidence detections from model 2 only
            for idx, det2 in enumerate(detections_2):
                if idx not in used_det2 and det2['confidence'] > 0.98:
                    final_detections.append({
                        'label': det2['label'],
                        'confidence': det2['confidence'],
                        'bbox': det2['bbox'],
                        'source': 'model2'
                    })
                    self.metrics.model2_only += 1
            
            return final_detections
    
    def nms_detections(self, detections, iou_threshold=0.5):
        """Non-maximum suppression to remove duplicate detections"""
        if len(detections) == 0:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while len(detections) > 0:
            best = detections[0]
            keep.append(best)
            detections = detections[1:]
            
            # Remove overlapping detections
            filtered = []
            for det in detections:
                iou = self.compute_iou(best['bbox'], det['bbox'])
                if iou < iou_threshold:
                    filtered.append(det)
            detections = filtered
        
        return keep
    
    def detect_objects(self, image):
        """
        Run DUAL-MODEL detection
        
        Returns:
            List of ensemble detections
        """
        # Get detections from both models
        detections_1 = self.detect_with_model(
            image, self.model_1, self.processor_1, self.confidence_threshold_1
        )
        
        detections_2 = self.detect_with_model(
            image, self.model_2, self.processor_2, self.confidence_threshold_2
        )
        
        # Combine using ensemble strategy
        final_detections = self.ensemble_detections(detections_1, detections_2)
        
        return final_detections
    
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
            source = det.get('source', 'unknown')
            
            xmin, ymin, xmax, ymax = bbox.astype(int)
            
            # Get depth if available
            depth_text = ""
            if depth_frame is not None:
                depth = self.get_depth_at_bbox(depth_frame, bbox)
                if depth > 0:
                    depth_text = f" | {depth:.2f}m"
            
            # Color based on source
            if source == 'both':
                color = (0, 255, 0)  # Green - high confidence
                source_text = "✓✓"
            elif source == 'model1':
                color = (0, 255, 255)  # Yellow - model 1 only
                source_text = "M1"
            elif source == 'model2':
                color = (255, 0, 255)  # Magenta - model 2 only
                source_text = "M2"
            else:
                color = (255, 255, 255)  # White - unknown
                source_text = "?"
            
            # Draw bounding box
            cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), color, 2)
            
            # Draw label
            text = f"{label}: {confidence:.2f} [{source_text}]{depth_text}"
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
    
    def draw_metrics_overlay(self, image):
        """Draw metrics overlay on image"""
        h, w = image.shape[:2]

        overlay = image.copy()
        cv2.rectangle(overlay, (w - 280, 0), (w, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        fps = self.metrics.get_fps()
        avg_ms = self.metrics.get_avg_inference_time()
        frames = self.metrics.frame_count

        y_pos = 25
        cv2.putText(image, f"FPS: {fps:.1f}", (w - 270, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 25
        cv2.putText(image, f"Inference: {avg_ms:.1f}ms", (w - 270, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_pos += 25
        cv2.putText(image, f"Frames: {frames}", (w - 270, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += 25
        cv2.putText(image, f"Detections: {self.metrics.total_detections}", (w - 270, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += 25
        cv2.putText(image, f"Strategy: {self.ensemble_strategy}", (w - 270, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return image

    def run_continuous(self):
        """Run continuous dual-model detection"""
        print("\n" + "="*70)
        print("🎥 RUNNING DUAL-MODEL CONTINUOUS DETECTION")
        print("="*70)
        print("\nControls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("  'm' - Print metrics")
        print("  'v' - Voting strategy (both models agree)")
        print("  'i' - Intersection strategy (very conservative)")
        print("  'u' - Union strategy (high recall)")
        print("  'w' - Weighted strategy (balanced)")
        print()

        self.metrics.reset()
        frame_count = 0
        last_detections = []

        try:
            while True:
                color_image, depth_frame = self.get_frame(timeout_ms=5000)

                if color_image is None:
                    continue

                # Run detection every 3 frames for performance
                if frame_count % 3 == 0:
                    start_time = time.time()
                    detections = self.detect_objects(color_image)
                    inference_time = time.time() - start_time
                    last_detections = detections

                    self.metrics.update_inference(inference_time, detections)

                    annotated = self.draw_detections(color_image, detections, depth_frame)

                    fps = self.metrics.get_fps()
                    info_text = f"DUAL-MODEL DETR ({self.ensemble_strategy}) | {len(detections)} objects | {fps:.1f} FPS"
                    cv2.putText(annotated, info_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    annotated = self.draw_metrics_overlay(annotated)
                else:
                    annotated = self.draw_detections(color_image, last_detections, depth_frame)
                    fps = self.metrics.get_fps()
                    cv2.putText(annotated, f"DUAL-MODEL DETR ({self.ensemble_strategy}) | {len(last_detections)} objects | {fps:.1f} FPS", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    annotated = self.draw_metrics_overlay(annotated)

                # Add legend
                cv2.putText(annotated, "Green=Both | Yellow=M1 | Magenta=M2", 
                           (10, annotated.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow('RealSense + Dual DETR', annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"dual_detection_{timestamp}.jpg"
                    cv2.imwrite(filename, annotated)
                    print(f"Screenshot saved: {filename}")
                elif key == ord('m'):
                    self.metrics.print_summary()
                elif key == ord('v'):
                    self.ensemble_strategy = "voting"
                    print("Switched to VOTING strategy")
                elif key == ord('i'):
                    self.ensemble_strategy = "intersection"
                    print("Switched to INTERSECTION strategy")
                elif key == ord('u'):
                    self.ensemble_strategy = "union"
                    print("Switched to UNION strategy")
                elif key == ord('w'):
                    self.ensemble_strategy = "weighted"
                    print("Switched to WEIGHTED strategy")

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
    print("="*70)
    print("RealSense D435 + DUAL DETR MODELS")
    print("Ensemble Detection with Positive + Hard-Negative Models")
    print("="*70)

    try:
        print("\nSelect ensemble strategy:")
        print("1. Voting (both models agree) - HIGH PRECISION")
        print("2. Intersection (very conservative) - VERY HIGH PRECISION")
        print("3. Union (combine all) - HIGH RECALL")
        print("4. Weighted (balanced) - BALANCED (RECOMMENDED)")

        choice = input("Enter choice (1-4, default=4): ").strip() or "4"

        strategy_map = {
            "1": "voting",
            "2": "intersection",
            "3": "union",
            "4": "weighted"
        }

        strategy = strategy_map.get(choice, "weighted")

        detector = DualModelRealsenseDETR(
            confidence_threshold_1=0.95,  # Model 1 threshold
            confidence_threshold_2=0.95,  # Model 2 threshold
            ensemble_strategy=strategy
        )

        detector.run_continuous()
        detector.cleanup()

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()