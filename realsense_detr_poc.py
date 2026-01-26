"""
RealSense D435 + DETR Object Detection POC (Fixed Version)
Uses Facebook's DETR (Detection Transformer) for object detection
With improved camera initialization and error handling
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

# COCO class names (91 classes - includes N/A placeholders)
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

# Filter to detect only these classes (set to None to detect all)
DETECT_ONLY = {'person'}  # Add more classes like {'person', 'chair', 'cup'}


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
        # For evaluation against ground truth
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
        # Use last 30 frames for smoothed FPS
        recent = self.inference_times[-30:]
        avg_time = sum(recent) / len(recent)
        return 1.0 / avg_time if avg_time > 0 else 0.0

    def get_avg_inference_time(self):
        """Get average inference time in milliseconds"""
        if len(self.inference_times) == 0:
            return 0.0
        return (sum(self.inference_times) / len(self.inference_times)) * 1000

    def compute_iou(self, box1, box2):
        """Compute IoU between two boxes [x, y, w, h] or [x1, y1, x2, y2]"""
        # Convert to [x1, y1, x2, y2] if needed
        if len(box1) == 4 and box1[2] < box1[0] + box1[2]:  # [x, y, w, h] format
            b1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
        else:
            b1 = box1
        if len(box2) == 4 and box2[2] < box2[0] + box2[2]:
            b2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]
        else:
            b2 = box2

        # Intersection
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
        """
        Evaluate detections against ground truth for one frame

        Args:
            detections: List of {'label': str, 'bbox': [x1,y1,x2,y2], ...}
            ground_truths: List of {'label': str, 'bbox': [x,y,w,h]}
            iou_threshold: IoU threshold for matching
        """
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

        # Unmatched ground truths are false negatives
        self.false_negatives += len(ground_truths) - len(matched_gt)

    def get_precision(self):
        """Calculate precision: TP / (TP + FP)"""
        total = self.true_positives + self.false_positives
        return self.true_positives / total if total > 0 else 0.0

    def get_recall(self):
        """Calculate recall: TP / (TP + FN)"""
        total = self.true_positives + self.false_negatives
        return self.true_positives / total if total > 0 else 0.0

    def get_f1_score(self):
        """Calculate F1 score: 2 * (precision * recall) / (precision + recall)"""
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
            print(f"\nEvaluation Metrics (IoU >= 0.5):")
            print(f"  Precision:  {s['precision']:.4f}")
            print(f"  Recall:     {s['recall']:.4f}")
            print(f"  F1 Score:   {s['f1_score']:.4f}")
            print(f"  TP: {s['true_positives']} | FP: {s['false_positives']} | FN: {s['false_negatives']}")
        print("="*60)


class RealsenseDETR:
    def __init__(self, confidence_threshold=0.7):
        """
        Initialize RealSense camera and DETR model
        
        Args:
            confidence_threshold: Minimum confidence for detections (0-1)
        """
        print("Initializing DETR Object Detection POC...")
        
        # Confidence threshold
        self.confidence_threshold = confidence_threshold
        
        # Load DETR model
        print("Loading DETR model (this may take a minute)...")
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        
        # Check GPU compatibility - RTX 50 series (sm_120) not yet supported
        self.device = torch.device("cpu")  # Default to CPU
        if torch.cuda.is_available():
            try:
                # Test if CUDA actually works with a small tensor operation
                test_tensor = torch.zeros(1).cuda()
                _ = test_tensor + 1
                del test_tensor
                torch.cuda.empty_cache()
                self.device = torch.device("cuda")
                print(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
            except RuntimeError as e:
                print(f"CUDA available but not compatible: {e}")
                print("Falling back to CPU (RTX 50 series requires newer PyTorch)")
                self.device = torch.device("cpu")
        else:
            print("CUDA not available, using CPU")

        self.model.to(self.device)
        self.model.eval()
        print(f"Using device: {self.device}")
        
        # Initialize RealSense pipeline
        self.pipeline = None
        self._cleaned_up = False
        self.initialize_camera()

        # Initialize metrics tracker
        self.metrics = MetricsTracker()
    
    def initialize_camera(self):
        """Initialize RealSense camera with better error handling"""
        print("\nInitializing RealSense D435...")
        
        try:
            # Create context and check for devices
            ctx = rs.context()
            devices = ctx.query_devices()
            
            if len(devices) == 0:
                raise RuntimeError("No RealSense devices found! Please check connection.")
            
            print(f"Found {len(devices)} RealSense device(s)")
            
            # Get device info
            dev = devices[0]
            print(f"Device: {dev.get_info(rs.camera_info.name)}")
            print(f"Serial: {dev.get_info(rs.camera_info.serial_number)}")
            print(f"Firmware: {dev.get_info(rs.camera_info.firmware_version)}")
            
            # Create pipeline
            self.pipeline = rs.pipeline()
            config = rs.config()
            
            # Enable device by serial number (more reliable)
            config.enable_device(dev.get_info(rs.camera_info.serial_number))
            
            # Try different resolutions (start with lower res for compatibility)
            resolutions = [
                (640, 480, 30),
                (848, 480, 30),
                (1280, 720, 30),
            ]
            
            camera_started = False
            
            for width, height, fps in resolutions:
                try:
                    print(f"\nTrying resolution: {width}x{height} @ {fps}fps...")
                    
                    # Clear previous config
                    config = rs.config()
                    config.enable_device(dev.get_info(rs.camera_info.serial_number))
                    
                    # Configure streams
                    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
                    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
                    
                    # Start pipeline
                    profile = self.pipeline.start(config)
                    
                    # Get device from profile
                    device = profile.get_device()
                    
                    # Disable auto-exposure for more consistent results (optional)
                    # Uncomment if needed:
                    # depth_sensor = device.first_depth_sensor()
                    # depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
                    
                    print(f"✅ Camera started successfully at {width}x{height}")
                    camera_started = True
                    break
                    
                except Exception as e:
                    print(f"Failed with {width}x{height}: {str(e)}")
                    continue
            
            if not camera_started:
                raise RuntimeError("Failed to start camera with any resolution")
            
            # Warm up - wait for auto-exposure to stabilize
            print("\nWarming up camera (waiting for auto-exposure)...")
            successful_frames = 0
            max_attempts = 60  # 60 attempts = 2 seconds at 30fps
            
            for i in range(max_attempts):
                try:
                    # Use timeout
                    frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                    if frames.get_color_frame() and frames.get_depth_frame():
                        successful_frames += 1
                        if successful_frames >= 10:  # Got 10 good frames
                            break
                except RuntimeError:
                    print(f"Timeout on frame {i+1}, retrying...")
                    continue
            
            if successful_frames < 10:
                print(f"⚠️  Warning: Only got {successful_frames} frames during warmup")
            else:
                print(f"✅ Camera warmed up successfully ({successful_frames} frames)")
            
            print("✅ RealSense D435 ready!")
            
        except Exception as e:
            print(f"\n❌ Error initializing camera: {str(e)}")
            print("\nTroubleshooting:")
            print("1. Make sure RealSense Viewer can see the camera")
            print("2. Close RealSense Viewer if it's running")
            print("3. Try unplugging and replugging the camera")
            print("4. Make sure you're using a USB 3.0 port (blue port)")
            raise
    
    def get_frame(self, timeout_ms=5000):
        """
        Get color and depth frames from RealSense
        
        Args:
            timeout_ms: Timeout in milliseconds
            
        Returns:
            Tuple of (color_image, depth_frame) or (None, None) on error
        """
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return None, None
            
            # Convert to numpy arrays
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
            List of detections: [(label, confidence, bbox, depth), ...]
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

            # Post-process (inside no_grad block)
            target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)
            results = self.processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=self.confidence_threshold
            )[0]

            # Extract detections (move to CPU immediately)
            detections = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                class_name = COCO_CLASSES[label.item()]
                # Filter by DETECT_ONLY if specified
                if DETECT_ONLY is None or class_name in DETECT_ONLY:
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
        """
        Get median depth within bounding box
        
        Args:
            depth_frame: RealSense depth frame
            bbox: [xmin, ymin, xmax, ymax]
            
        Returns:
            Depth in meters (float)
        """
        xmin, ymin, xmax, ymax = bbox.astype(int)
        
        # Ensure bbox is within frame bounds
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(depth_frame.get_width(), xmax)
        ymax = min(depth_frame.get_height(), ymax)
        
        # Sample depth values in center region of bbox
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
                    if depth > 0:  # Valid depth
                        depths.append(depth)
        
        if depths:
            return np.median(depths)
        return 0.0
    
    def draw_detections(self, image, detections, depth_frame=None):
        """
        Draw bounding boxes and labels on image
        
        Args:
            image: BGR image
            detections: List of detections
            depth_frame: Optional RealSense depth frame for distance
            
        Returns:
            Annotated image
        """
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
            
            # Draw label background
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
            
            # Draw label text
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
        """
        Process single frame and save result
        
        Args:
            save_path: Path to save annotated image
        """
        print("\n🎯 Running single frame detection...")
        
        # Get frame with retry
        max_retries = 5
        for attempt in range(max_retries):
            print(f"Attempting to capture frame ({attempt+1}/{max_retries})...")
            color_image, depth_frame = self.get_frame(timeout_ms=10000)
            
            if color_image is not None:
                print("✅ Frame captured successfully")
                break
            else:
                print(f"⚠️  Frame capture failed, retrying...")
                time.sleep(1)
        
        if color_image is None:
            print("❌ Failed to get frame after all retries")
            return
        
        # Run detection
        print("Running DETR inference...")
        start_time = time.time()
        detections = self.detect_objects(color_image)
        inference_time = time.time() - start_time
        
        print(f"✅ Detection complete in {inference_time:.2f}s")
        print(f"Found {len(detections)} objects:")
        
        # Print detections
        for i, det in enumerate(detections, 1):
            depth = self.get_depth_at_bbox(depth_frame, det['bbox'])
            depth_str = f"{depth:.2f}m" if depth > 0 else "N/A"
            print(f"  {i}. {det['label']}: {det['confidence']:.2f} | Distance: {depth_str}")
        
        # Draw detections
        annotated = self.draw_detections(color_image, detections, depth_frame)
        
        # Add info text
        info_text = f"DETR Object Detection | {len(detections)} objects | {inference_time:.2f}s"
        cv2.putText(
            annotated, info_text, (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
        
        # Save result
        cv2.imwrite(save_path, annotated)
        print(f"✅ Result saved to: {save_path}")
        
        # Display result
        cv2.imshow('DETR Detection Result', annotated)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def draw_metrics_overlay(self, image):
        """Draw metrics overlay on image"""
        h, w = image.shape[:2]

        # Semi-transparent background for metrics
        overlay = image.copy()
        cv2.rectangle(overlay, (w - 220, 0), (w, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        # Metrics text
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
        """
        Run continuous object detection with metrics tracking
        """
        print("\n🎥 Running continuous detection...")
        print("Press 'q' to quit, 's' to save screenshot, 'm' to print metrics")

        # Reset metrics for this session
        self.metrics.reset()

        frame_count = 0
        last_detections = []
        error_count = 0
        max_errors = 5

        try:
            while True:
                # Get frame
                color_image, depth_frame = self.get_frame(timeout_ms=5000)

                if color_image is None:
                    print("⚠️  Frame skipped")
                    continue

                # Run detection every 3 frames (for speed)
                if frame_count % 3 == 0:
                    try:
                        start_time = time.time()
                        detections = self.detect_objects(color_image)
                        inference_time = time.time() - start_time
                        last_detections = detections
                        error_count = 0  # Reset error count on success

                        # Update metrics
                        self.metrics.update_inference(inference_time, detections)

                        # Draw detections
                        annotated = self.draw_detections(color_image, detections, depth_frame)

                        # Add info text
                        fps = self.metrics.get_fps()
                        info_text = f"DETR | {len(detections)} objects | {fps:.1f} FPS"
                        cv2.putText(
                            annotated, info_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                        )

                        # Draw metrics overlay
                        annotated = self.draw_metrics_overlay(annotated)

                    except RuntimeError as e:
                        error_count += 1
                        print(f"⚠️  CUDA error ({error_count}/{max_errors}): {str(e)}")

                        # Try to recover by clearing CUDA cache
                        if self.device.type == 'cuda':
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()

                        if error_count >= max_errors:
                            print("❌ Too many CUDA errors, stopping...")
                            break

                        # Use last detections or empty
                        annotated = self.draw_detections(color_image, last_detections, depth_frame)
                        cv2.putText(
                            annotated, "DETR | Recovering from error...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                        )
                else:
                    # Reuse last detections
                    annotated = self.draw_detections(color_image, last_detections, depth_frame)
                    fps = self.metrics.get_fps()
                    cv2.putText(
                        annotated, f"DETR | {len(last_detections)} objects | {fps:.1f} FPS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                    )
                    annotated = self.draw_metrics_overlay(annotated)

                # Display
                cv2.imshow('RealSense + DETR Object Detection', annotated)

                # Handle keyboard
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
            # Print final metrics summary
            self.metrics.print_summary()
            self.cleanup()
    
    def evaluate_dataset(self, images_dir, annotations_file, iou_threshold=0.5, visualize=False):
        """
        Evaluate model on a COCO-format dataset and compute precision/recall/F1

        Args:
            images_dir: Path to images directory
            annotations_file: Path to COCO format annotations JSON
            iou_threshold: IoU threshold for matching detections to ground truth
            visualize: Whether to show visualizations

        Returns:
            dict: Metrics summary
        """
        print("\n" + "="*60)
        print("EVALUATING MODEL ON DATASET")
        print("="*60)

        # Load annotations
        with open(annotations_file, 'r') as f:
            coco = json.load(f)

        print(f"Images: {len(coco['images'])}")
        print(f"Annotations: {len(coco['annotations'])}")
        print(f"IoU threshold: {iou_threshold}")

        # Build lookup tables
        img_id_to_info = {img['id']: img for img in coco['images']}
        cat_id_to_name = {cat['id']: cat['name'] for cat in coco['categories']}

        # Group annotations by image
        img_to_anns = {}
        for ann in coco['annotations']:
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append({
                'label': cat_id_to_name[ann['category_id']],
                'bbox': ann['bbox']  # [x, y, width, height]
            })

        # Reset metrics
        self.metrics.reset()

        # Process each image
        for idx, img_info in enumerate(coco['images']):
            img_path = os.path.join(images_dir, img_info['file_name'])

            if not os.path.exists(img_path):
                print(f"⚠️  Image not found: {img_path}")
                continue

            # Load image
            image = cv2.imread(img_path)
            if image is None:
                print(f"⚠️  Could not load: {img_path}")
                continue

            # Run detection
            start_time = time.time()
            detections = self.detect_objects(image)
            inference_time = time.time() - start_time

            # Update inference metrics
            self.metrics.update_inference(inference_time, detections)

            # Get ground truths for this image
            ground_truths = img_to_anns.get(img_info['id'], [])

            # Evaluate detections against ground truth
            self.metrics.evaluate_frame(detections, ground_truths, iou_threshold)

            # Progress
            if (idx + 1) % 10 == 0 or idx == len(coco['images']) - 1:
                print(f"Processed {idx + 1}/{len(coco['images'])} images...")

            # Visualize if requested
            if visualize:
                annotated = self.draw_detections(image, detections)
                # Draw ground truth in blue
                for gt in ground_truths:
                    x, y, w, h = [int(v) for v in gt['bbox']]
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(annotated, f"GT: {gt['label']}", (x, y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                cv2.imshow('Evaluation (Green=Det, Blue=GT) - Press any key', annotated)
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    break

        # Print final summary
        self.metrics.print_summary()

        if visualize:
            cv2.destroyAllWindows()

        return self.metrics.get_summary()

    def cleanup(self):
        """Stop pipeline and close windows"""
        if self._cleaned_up:
            return  # Already cleaned up
        self._cleaned_up = True
        print("\nCleaning up...")
        if self.pipeline:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass  # Already stopped
        cv2.destroyAllWindows()
        print("✅ Done!")


def main():
    """Main function"""
    print("="*60)
    print("RealSense D435 + DETR Object Detection POC")
    print("="*60)

    try:
        # Choose mode first (evaluation doesn't need camera)
        print("\nSelect mode:")
        print("1. Single frame detection (requires camera)")
        print("2. Continuous detection with metrics (requires camera)")
        print("3. Evaluate on dataset (compute Precision/Recall/F1)")

        choice = input("Enter choice (1, 2, or 3): ").strip()

        if choice == "3":
            # Evaluation mode - initialize without camera requirement
            print("\n--- Dataset Evaluation Mode ---")
            images_dir = input("Enter path to images directory: ").strip()
            annotations_file = input("Enter path to COCO annotations JSON: ").strip()
            visualize = input("Visualize results? (y/n): ").strip().lower() == 'y'

            # Initialize detector (will still try camera, but we won't use it)
            print("\nNote: Camera initialization may fail - that's OK for evaluation mode.")
            try:
                detector = RealsenseDETR(confidence_threshold=0.7)
            except Exception as cam_error:
                print(f"Camera init failed (expected): {cam_error}")
                # Create detector without camera
                detector = RealsenseDETR.__new__(RealsenseDETR)
                detector.confidence_threshold = 0.7
                detector.pipeline = None
                detector._cleaned_up = True
                detector.metrics = MetricsTracker()

                # Load model
                print("Loading DETR model...")
                detector.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
                detector.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
                detector.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                detector.model.to(detector.device)
                detector.model.eval()
                print(f"Using device: {detector.device}")

            # Run evaluation
            detector.evaluate_dataset(images_dir, annotations_file, visualize=visualize)
        else:
            # Camera modes - initialize normally
            detector = RealsenseDETR(confidence_threshold=0.7)

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