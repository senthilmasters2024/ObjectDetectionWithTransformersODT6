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

# COCO class names (80 classes)
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
        
        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print(f"Using device: {self.device}")
        
        # Initialize RealSense pipeline
        self.pipeline = None
        self.initialize_camera()
    
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
            detections.append({
                'label': COCO_CLASSES[label.item()],
                'confidence': score.item(),
                'bbox': box.cpu().numpy()
            })
        
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
    
    def run_continuous(self):
        """
        Run continuous object detection
        """
        print("\n🎥 Running continuous detection...")
        print("Press 'q' to quit, 's' to save screenshot")
        
        frame_count = 0
        last_detections = []
        
        try:
            while True:
                # Get frame
                color_image, depth_frame = self.get_frame(timeout_ms=5000)
                
                if color_image is None:
                    print("⚠️  Frame skipped")
                    continue
                
                # Run detection every 3 frames (for speed)
                if frame_count % 3 == 0:
                    start_time = time.time()
                    detections = self.detect_objects(color_image)
                    inference_time = time.time() - start_time
                    last_detections = detections
                    
                    # Draw detections
                    annotated = self.draw_detections(color_image, detections, depth_frame)
                    
                    # Add FPS
                    fps = 1.0 / inference_time if inference_time > 0 else 0
                    info_text = f"DETR | {len(detections)} objects | {fps:.1f} FPS"
                    cv2.putText(
                        annotated, info_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                    )
                else:
                    # Reuse last detections
                    annotated = self.draw_detections(color_image, last_detections, depth_frame)
                    cv2.putText(
                        annotated, f"DETR | {len(last_detections)} objects", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                    )
                
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
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Stop pipeline and close windows"""
        print("\nCleaning up...")
        if self.pipeline:
            self.pipeline.stop()
        cv2.destroyAllWindows()
        print("✅ Done!")


def main():
    """Main function"""
    print("="*60)
    print("RealSense D435 + DETR Object Detection POC")
    print("="*60)
    
    try:
        # Initialize
        detector = RealsenseDETR(confidence_threshold=0.7)
        
        # Choose mode
        print("\nSelect mode:")
        print("1. Single frame detection")
        print("2. Continuous detection (live)")
        
        choice = input("Enter choice (1 or 2): ").strip()
        
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