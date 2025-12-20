"""
RealSense D435 + DETR Object Detection POC
Uses Facebook's DETR (Detection Transformer) for object detection
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import time
import argparse
import warnings

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
    def __init__(self, confidence_threshold=0.7, force_cpu=False):
        """
        Initialize RealSense camera and DETR model

        Args:
            confidence_threshold: Minimum confidence for detections (0-1)
            force_cpu: Force CPU usage even if GPU is available
        """
        print("Initializing DETR Object Detection POC...")

        # Confidence threshold
        self.confidence_threshold = confidence_threshold

        # Load DETR model
        print("Loading DETR model (this may take a minute)...")
        # Suppress some warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='torch')

        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        # Use GPU if available and compatible
        if force_cpu:
            self.device = torch.device("cpu")
            print("Using device: cpu (forced)")
        else:
            if torch.cuda.is_available():
                # Check if GPU is compatible
                try:
                    self.device = torch.device("cuda")
                    # Test if GPU works
                    test_tensor = torch.zeros(1).to(self.device)
                    print(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
                except Exception as e:
                    print(f"GPU available but not compatible: {e}")
                    print("Falling back to CPU")
                    self.device = torch.device("cpu")
            else:
                self.device = torch.device("cpu")
                print("Using device: cpu (no GPU available)")

        self.model.to(self.device)
        self.model.eval()
        
        # Initialize RealSense pipeline
        print("Initializing RealSense D435...")
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure streams
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        
        # Start streaming
        self.pipeline.start(self.config)
        print("✅ RealSense D435 initialized successfully!")
        
        # Warm up camera (skip first few frames)
        for _ in range(30):
            self.pipeline.wait_for_frames()
    
    def get_frame(self):
        """Get color and depth frames from RealSense"""
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            return None, None
        
        # Convert to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        return color_image, depth_frame
    
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
        
        # Get frame
        color_image, depth_frame = self.get_frame()
        if color_image is None:
            print("❌ Failed to get frame")
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
        
        try:
            while True:
                # Get frame
                color_image, depth_frame = self.get_frame()
                if color_image is None:
                    continue
                
                # Run detection every frame (or skip frames for speed)
                if frame_count % 1 == 0:  # Process every frame
                    start_time = time.time()
                    detections = self.detect_objects(color_image)
                    inference_time = time.time() - start_time
                    
                    # Draw detections
                    annotated = self.draw_detections(color_image, detections, depth_frame)
                    
                    # Add FPS
                    fps = 1.0 / inference_time if inference_time > 0 else 0
                    info_text = f"DETR | {len(detections)} objects | {fps:.1f} FPS"
                    cv2.putText(
                        annotated, info_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                    )
                    
                    # Display
                    cv2.imshow('RealSense + DETR Object Detection', annotated)
                else:
                    # Just show original frame
                    cv2.imshow('RealSense + DETR Object Detection', color_image)
                
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
    
    def run_video_file(self, video_path, output_path="output_detection.mp4"):
        """
        Process video file and save with detections

        Args:
            video_path: Path to input video file
            output_path: Path to save output video
        """
        print(f"\n🎬 Processing video file: {video_path}")

        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video file: {video_path}")
            return

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video info: {width}x{height} @ {fps} FPS, {total_frames} frames")

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Run detection
                detections = self.detect_objects(frame)

                # Draw detections (no depth info for video files)
                annotated = self.draw_detections(frame, detections, depth_frame=None)

                # Add frame info
                info_text = f"Frame {frame_count}/{total_frames} | {len(detections)} objects"
                cv2.putText(
                    annotated, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )

                # Write frame
                out.write(annotated)

                # Progress update every 30 frames
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    elapsed = time.time() - start_time
                    fps_processing = frame_count / elapsed if elapsed > 0 else 0
                    eta = (total_frames - frame_count) / fps_processing if fps_processing > 0 else 0
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) | "
                          f"Processing: {fps_processing:.2f} FPS | ETA: {eta:.1f}s")

        finally:
            cap.release()
            out.release()

        elapsed_time = time.time() - start_time
        avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        print(f"\n✅ Video processing complete!")
        print(f"Processed {frame_count} frames in {elapsed_time:.1f}s ({avg_fps:.2f} FPS)")
        print(f"Output saved to: {output_path}")

    def cleanup(self):
        """Stop pipeline and close windows"""
        print("\nCleaning up...")
        if hasattr(self, 'pipeline'):
            self.pipeline.stop()
        cv2.destroyAllWindows()
        print("✅ Done!")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="RealSense D435 + DETR Object Detection POC")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "continuous", "video"],
        default="single",
        help="Detection mode: 'single' for one frame, 'continuous' for live camera, 'video' for video file"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to input video file (required for video mode)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.7,
        help="Confidence threshold for detections (0-1)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage even if GPU is available"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (default: detection_result.jpg for single, output_detection.mp4 for video)"
    )

    args = parser.parse_args()

    # Validate video mode
    if args.mode == "video" and not args.video:
        parser.error("--video is required when using video mode")

    print("="*60)
    print("RealSense D435 + DETR Object Detection POC")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Confidence threshold: {args.confidence}")

    # Initialize detector (skip camera init for video mode)
    if args.mode == "video":
        # For video mode, create a simplified detector without RealSense
        class VideoDetector:
            def __init__(self, confidence_threshold, force_cpu):
                print("Initializing DETR Object Detection for Video...")
                self.confidence_threshold = confidence_threshold

                # Load DETR model
                print("Loading DETR model (this may take a minute)...")
                warnings.filterwarnings('ignore', category=UserWarning, module='torch')

                from transformers import DetrImageProcessor, DetrForObjectDetection
                self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
                self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

                # Use GPU if available and compatible
                if force_cpu:
                    self.device = torch.device("cpu")
                    print("Using device: cpu (forced)")
                else:
                    if torch.cuda.is_available():
                        try:
                            self.device = torch.device("cuda")
                            test_tensor = torch.zeros(1).to(self.device)
                            print(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
                        except Exception as e:
                            print(f"GPU available but not compatible: {e}")
                            print("Falling back to CPU")
                            self.device = torch.device("cpu")
                    else:
                        self.device = torch.device("cpu")
                        print("Using device: cpu (no GPU available)")

                self.model.to(self.device)
                self.model.eval()

        # Copy methods from RealsenseDETR that we need
        VideoDetector.detect_objects = RealsenseDETR.detect_objects
        VideoDetector.draw_detections = RealsenseDETR.draw_detections
        VideoDetector.run_video_file = RealsenseDETR.run_video_file

        detector = VideoDetector(confidence_threshold=args.confidence, force_cpu=args.cpu)
        output = args.output if args.output else "output_detection.mp4"
        detector.run_video_file(args.video, output)

    else:
        # Camera modes
        detector = RealsenseDETR(confidence_threshold=args.confidence, force_cpu=args.cpu)

        try:
            if args.mode == "single":
                output = args.output if args.output else "detection_result.jpg"
                detector.run_single_frame(save_path=output)
            else:  # continuous
                detector.run_continuous()
        finally:
            detector.cleanup()


if __name__ == "__main__":
    main()
