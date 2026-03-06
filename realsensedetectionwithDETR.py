"""
RealSense D435 + DETR Object Detection POC (Fixed Version)
Uses Facebook's DETR (Detection Transformer) for object detection
With improved camera initialization and error handling

Configuration is loaded from detector_config.json — edit that file
to change detected classes and thresholds without touching this code.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import time
import csv
import json
import os
from collections import defaultdict

CONFIG_FILE = "detector_config.json"

def load_config(path=CONFIG_FILE):
    if not os.path.exists(path):
        print(f"Warning: {path} not found, using defaults.")
        return {"detect_only": None, "confidence_threshold": 0.7, "inference_every_n_frames": 3}
    with open(path) as f:
        cfg = json.load(f)
    # Strip keys starting with _ (comments/reference fields)
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    # Convert list to set (or None for all classes)
    if "detect_only" in cfg and cfg["detect_only"] is not None:
        cfg["detect_only"] = set(cfg["detect_only"])
    return cfg

CONFIG = load_config()
print(f"Loaded config from {CONFIG_FILE}")
print(f"  detect_only          : {CONFIG.get('detect_only')}")
print(f"  confidence_threshold : {CONFIG.get('confidence_threshold')}")
print(f"  inference_every      : every {CONFIG.get('inference_every_n_frames')} frames")

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

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

DETECT_ONLY = CONFIG.get("detect_only")  # loaded from detector_config.json


# ──────────────────────────────────────────────────────────────────────────────
# Metrics Tracker
# ──────────────────────────────────────────────────────────────────────────────

class MetricsTracker:
    """
    Tracks per-frame and per-session detection metrics.

    Collected data:
      - Inference time (ms) per frame
      - FPS (rolling 30-frame average)
      - Number of detections per frame
      - Per-class detection count and confidence scores
      - Precision / Recall / F1 against ground truth (evaluate_dataset mode)

    Export options:
      - CSV  : per-frame log + per-class summary
      - JSON : full session summary
      - Plots: bar chart, confidence histogram, FPS/inference time line charts
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.session_start = time.time()
        self.frame_records = []                      # per-frame dicts
        self.class_confidences = defaultdict(list)   # label -> [conf, ...]
        self.fps_history = []                        # rolling window
        # Ground-truth evaluation counters
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0

    # ── Data collection ────────────────────────────────────────────────────────

    def update(self, inference_time_s, detections):
        """Call once per inference with elapsed seconds and detection list."""
        frame_num = len(self.frame_records) + 1
        ms = inference_time_s * 1000

        self.fps_history.append(inference_time_s)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        fps = 1.0 / (sum(self.fps_history) / len(self.fps_history))

        per_det = []
        for d in detections:
            self.class_confidences[d['label']].append(d['confidence'])
            per_det.append({'label': d['label'], 'confidence': round(d['confidence'], 4)})

        self.frame_records.append({
            'frame': frame_num,
            'timestamp': round(time.time() - self.session_start, 3),
            'inference_ms': round(ms, 2),
            'fps': round(fps, 2),
            'n_detections': len(detections),
            'detections': per_det,
        })

    # Legacy alias used by evaluate_dataset
    def update_inference(self, inference_time_s, detections):
        self.update(inference_time_s, detections)

    # ── Ground-truth evaluation ────────────────────────────────────────────────

    def compute_iou(self, box1, box2):
        """Compute IoU between two boxes ([x,y,w,h] or [x1,y1,x2,y2])."""
        def to_xyxy(b):
            if b[2] < b[0] + b[2]:
                return [b[0], b[1], b[0] + b[2], b[1] + b[3]]
            return list(b)
        b1, b2 = to_xyxy(box1), to_xyxy(box2)
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        if x2 < x1 or y2 < y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def evaluate_frame(self, detections, ground_truths, iou_threshold=0.5):
        """Match detections to ground truth and accumulate TP/FP/FN."""
        matched_gt = set()
        for det in detections:
            best_iou, best_idx = 0, -1
            for gt_idx, gt in enumerate(ground_truths):
                if gt_idx in matched_gt or det['label'] != gt['label']:
                    continue
                iou = self.compute_iou(det['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou, best_idx = iou, gt_idx
            if best_iou >= iou_threshold and best_idx >= 0:
                self.true_positives += 1
                matched_gt.add(best_idx)
            else:
                self.false_positives += 1
        self.false_negatives += len(ground_truths) - len(matched_gt)

    def get_precision(self):
        total = self.true_positives + self.false_positives
        return self.true_positives / total if total > 0 else 0.0

    def get_recall(self):
        total = self.true_positives + self.false_negatives
        return self.true_positives / total if total > 0 else 0.0

    def get_f1_score(self):
        p, r = self.get_precision(), self.get_recall()
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    # ── Computed summaries ─────────────────────────────────────────────────────

    def total_frames(self):
        return len(self.frame_records)

    def current_fps(self):
        if not self.fps_history:
            return 0.0
        return round(1.0 / (sum(self.fps_history) / len(self.fps_history)), 2)

    def avg_fps(self):
        if not self.frame_records:
            return 0.0
        return round(sum(r['fps'] for r in self.frame_records) / len(self.frame_records), 2)

    def avg_inference_ms(self):
        if not self.frame_records:
            return 0.0
        return round(sum(r['inference_ms'] for r in self.frame_records) / len(self.frame_records), 2)

    def total_detections(self):
        return sum(r['n_detections'] for r in self.frame_records)

    def class_summary(self):
        """Returns dict: label -> {count, avg_conf, min_conf, max_conf}"""
        summary = {}
        for label, confs in self.class_confidences.items():
            summary[label] = {
                'count': len(confs),
                'avg_confidence': round(float(np.mean(confs)), 4),
                'min_confidence': round(float(np.min(confs)), 4),
                'max_confidence': round(float(np.max(confs)), 4),
            }
        return dict(sorted(summary.items(), key=lambda x: -x[1]['count']))

    def session_summary(self):
        duration = round(time.time() - self.session_start, 2)
        summary = {
            'session_duration_s': duration,
            'total_frames': self.total_frames(),
            'avg_fps': self.avg_fps(),
            'avg_inference_ms': self.avg_inference_ms(),
            'total_detections': self.total_detections(),
            'unique_classes_detected': len(self.class_confidences),
            'class_summary': self.class_summary(),
        }
        # Include precision/recall/F1 if ground-truth evaluation was done
        if self.true_positives + self.false_positives + self.false_negatives > 0:
            summary['evaluation'] = {
                'precision': round(self.get_precision(), 4),
                'recall': round(self.get_recall(), 4),
                'f1_score': round(self.get_f1_score(), 4),
                'true_positives': self.true_positives,
                'false_positives': self.false_positives,
                'false_negatives': self.false_negatives,
            }
        return summary

    # Legacy alias
    def get_summary(self):
        return self.session_summary()

    def print_summary(self):
        s = self.session_summary()
        print("\n" + "=" * 60)
        print("  DETECTION METRICS SUMMARY")
        print("=" * 60)
        print(f"  Session duration   : {s['session_duration_s']} s")
        print(f"  Frames processed   : {s['total_frames']}")
        print(f"  Average FPS        : {s['avg_fps']}")
        print(f"  Avg inference time : {s['avg_inference_ms']} ms")
        print(f"  Total detections   : {s['total_detections']}")
        print(f"  Unique classes     : {s['unique_classes_detected']}")
        if s['class_summary']:
            print(f"\n  {'Class':<20} {'Count':>6}  {'Avg Conf':>9}  {'Min':>6}  {'Max':>6}")
            print(f"  {'-'*20}  {'-'*6}  {'-'*9}  {'-'*6}  {'-'*6}")
            for label, stats in s['class_summary'].items():
                print(f"  {label:<20} {stats['count']:>6}  {stats['avg_confidence']:>9.4f}"
                      f"  {stats['min_confidence']:>6.4f}  {stats['max_confidence']:>6.4f}")
        if 'evaluation' in s:
            ev = s['evaluation']
            print(f"\n  Ground-truth Evaluation (IoU >= 0.5):")
            print(f"  Precision : {ev['precision']:.4f}")
            print(f"  Recall    : {ev['recall']:.4f}")
            print(f"  F1 Score  : {ev['f1_score']:.4f}")
            print(f"  TP: {ev['true_positives']}  FP: {ev['false_positives']}  FN: {ev['false_negatives']}")
        print("=" * 60)

    # ── Export ─────────────────────────────────────────────────────────────────

    def export_csv(self, output_dir):
        """Save two CSV files: per-frame log and per-class summary."""
        os.makedirs(output_dir, exist_ok=True)

        frame_path = os.path.join(output_dir, 'frames.csv')
        with open(frame_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'timestamp_s', 'inference_ms', 'fps', 'n_detections', 'labels'])
            for r in self.frame_records:
                labels = ';'.join(d['label'] for d in r['detections'])
                writer.writerow([r['frame'], r['timestamp'], r['inference_ms'],
                                 r['fps'], r['n_detections'], labels])
        print(f"  Saved: {frame_path}")

        class_path = os.path.join(output_dir, 'class_summary.csv')
        with open(class_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['class', 'count', 'avg_confidence', 'min_confidence', 'max_confidence'])
            for label, stats in self.class_summary().items():
                writer.writerow([label, stats['count'], stats['avg_confidence'],
                                 stats['min_confidence'], stats['max_confidence']])
        print(f"  Saved: {class_path}")

    def export_json(self, output_dir):
        """Save full session summary as JSON."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'session_summary.json')
        with open(path, 'w') as f:
            json.dump(self.session_summary(), f, indent=2)
        print(f"  Saved: {path}")

    def generate_plots(self, output_dir):
        """Generate and save four report-ready matplotlib figures."""
        os.makedirs(output_dir, exist_ok=True)
        cs = self.class_summary()

        if not cs:
            print("  No detections to plot.")
            return

        # ── 1. Detections per class ────────────────────────────────────────────
        labels = list(cs.keys())
        counts = [cs[l]['count'] for l in labels]

        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), 5))
        bars = ax.bar(labels, counts, color='steelblue', edgecolor='white')
        ax.bar_label(bars, padding=3, fontsize=9)
        ax.set_title('Detections per Class', fontsize=14, fontweight='bold')
        ax.set_xlabel('Class')
        ax.set_ylabel('Detection Count')
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.tight_layout()
        path = os.path.join(output_dir, 'detections_per_class.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")

        # ── 2. Average confidence per class ───────────────────────────────────
        avg_confs = [cs[l]['avg_confidence'] for l in labels]

        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), 5))
        bars = ax.bar(labels, avg_confs, color='darkorange', edgecolor='white')
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title('Average Confidence per Class', fontsize=14, fontweight='bold')
        ax.set_xlabel('Class')
        ax.set_ylabel('Avg Confidence Score')
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.tight_layout()
        path = os.path.join(output_dir, 'avg_confidence_per_class.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")

        # ── 3. Confidence score distribution ──────────────────────────────────
        all_confs = [c for confs in self.class_confidences.values() for c in confs]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(all_confs, bins=20, range=(0, 1), color='mediumseagreen', edgecolor='white')
        ax.axvline(float(np.mean(all_confs)), color='red', linestyle='--',
                   label=f'Mean = {np.mean(all_confs):.3f}')
        ax.set_title('Confidence Score Distribution (All Detections)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Frequency')
        ax.legend()
        plt.tight_layout()
        path = os.path.join(output_dir, 'confidence_distribution.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")

        # ── 4. FPS & inference time over frames ───────────────────────────────
        frames = [r['frame'] for r in self.frame_records]
        fps_vals = [r['fps'] for r in self.frame_records]
        ms_vals = [r['inference_ms'] for r in self.frame_records]

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(frames, fps_vals, color='steelblue', linewidth=1.5, label='FPS')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('FPS', color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue')

        ax2 = ax1.twinx()
        ax2.plot(frames, ms_vals, color='tomato', linewidth=1.5, linestyle='--', label='Inference (ms)')
        ax2.set_ylabel('Inference Time (ms)', color='tomato')
        ax2.tick_params(axis='y', labelcolor='tomato')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        ax1.set_title('FPS & Inference Time over Frames', fontsize=14, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(output_dir, 'fps_and_inference_time.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")

        # ── 5. Precision / Recall / F1 bar (only when GT evaluation was run) ──
        if self.true_positives + self.false_positives + self.false_negatives > 0:
            metrics = ['Precision', 'Recall', 'F1 Score']
            values = [self.get_precision(), self.get_recall(), self.get_f1_score()]
            colors = ['steelblue', 'darkorange', 'mediumseagreen']

            fig, ax = plt.subplots(figsize=(6, 5))
            bars = ax.bar(metrics, values, color=colors, edgecolor='white')
            ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=11)
            ax.set_ylim(0, 1.1)
            ax.set_title('Precision / Recall / F1 Score', fontsize=14, fontweight='bold')
            ax.set_ylabel('Score')
            plt.tight_layout()
            path = os.path.join(output_dir, 'precision_recall_f1.png')
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"  Saved: {path}")

    def save_report(self, output_dir=None):
        """Save CSV + JSON + all plots to a timestamped folder."""
        if output_dir is None:
            output_dir = f"report_{time.strftime('%Y%m%d_%H%M%S')}"
        print(f"\nSaving report to: {output_dir}/")
        self.export_csv(output_dir)
        self.export_json(output_dir)
        self.generate_plots(output_dir)
        self.print_summary()
        print(f"\nReport complete -> {output_dir}/")
        return output_dir


# ──────────────────────────────────────────────────────────────────────────────
# Main detector class
# ──────────────────────────────────────────────────────────────────────────────

class RealsenseDETR:
    def __init__(self, confidence_threshold=0.7):
        """
        Initialize RealSense camera and DETR model

        Args:
            confidence_threshold: Minimum confidence for detections (0-1)
        """
        print("Initializing DETR Object Detection POC...")

        self.confidence_threshold = confidence_threshold

        print("Loading DETR model (this may take a minute)...")
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        # Check GPU compatibility - RTX 50 series (sm_120) not yet supported
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            try:
                test_tensor = torch.zeros(1).cuda()
                _ = test_tensor + 1
                del test_tensor
                torch.cuda.empty_cache()
                self.device = torch.device("cuda")
                print(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
            except RuntimeError as e:
                print(f"CUDA available but not compatible: {e}")
                print("Falling back to CPU (RTX 50 series requires newer PyTorch)")
        else:
            print("CUDA not available, using CPU")

        self.model.to(self.device)
        self.model.eval()
        print(f"Using device: {self.device}")

        self.pipeline = None
        self._cleaned_up = False
        self.initialize_camera()

        self.metrics = MetricsTracker()

    def initialize_camera(self):
        """Initialize RealSense camera with better error handling"""
        print("\nInitializing RealSense D435...")

        try:
            ctx = rs.context()
            devices = ctx.query_devices()

            if len(devices) == 0:
                raise RuntimeError("No RealSense devices found! Please check connection.")

            print(f"Found {len(devices)} RealSense device(s)")
            dev = devices[0]
            print(f"Device: {dev.get_info(rs.camera_info.name)}")
            print(f"Serial: {dev.get_info(rs.camera_info.serial_number)}")
            print(f"Firmware: {dev.get_info(rs.camera_info.firmware_version)}")

            self.pipeline = rs.pipeline()
            resolutions = [(640, 480, 30), (848, 480, 30), (1280, 720, 30)]
            camera_started = False

            for width, height, fps in resolutions:
                try:
                    print(f"\nTrying resolution: {width}x{height} @ {fps}fps...")
                    config = rs.config()
                    config.enable_device(dev.get_info(rs.camera_info.serial_number))
                    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
                    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
                    self.pipeline.start(config)
                    print(f"Camera started successfully at {width}x{height}")
                    camera_started = True
                    break
                except Exception as e:
                    print(f"Failed with {width}x{height}: {str(e)}")
                    continue

            if not camera_started:
                raise RuntimeError("Failed to start camera with any resolution")

            print("\nWarming up camera (waiting for auto-exposure)...")
            successful_frames = 0
            for i in range(60):
                try:
                    frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                    if frames.get_color_frame() and frames.get_depth_frame():
                        successful_frames += 1
                        if successful_frames >= 10:
                            break
                except RuntimeError:
                    print(f"Timeout on frame {i+1}, retrying...")
                    continue

            if successful_frames < 10:
                print(f"Warning: Only got {successful_frames} frames during warmup")
            else:
                print(f"Camera warmed up successfully ({successful_frames} frames)")
            print("RealSense D435 ready!")

        except Exception as e:
            print(f"\nError initializing camera: {str(e)}")
            print("\nTroubleshooting:")
            print("1. Make sure RealSense Viewer can see the camera")
            print("2. Close RealSense Viewer if it's running")
            print("3. Try unplugging and replugging the camera")
            print("4. Make sure you're using a USB 3.0 port (blue port)")
            raise

    def get_frame(self, timeout_ms=5000):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                return None, None
            return np.asanyarray(color_frame.get_data()), depth_frame
        except RuntimeError as e:
            print(f"Warning: Frame timeout - {str(e)}")
            return None, None

    def detect_objects(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)
            results = self.processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=self.confidence_threshold
            )[0]

            detections = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                class_name = COCO_CLASSES[label.item()]
                if DETECT_ONLY is None or class_name in DETECT_ONLY:
                    detections.append({
                        'label': class_name,
                        'confidence': score.item(),
                        'bbox': box.cpu().numpy()
                    })

        del inputs, outputs, target_sizes, results
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return detections

    def get_depth_at_bbox(self, depth_frame, bbox):
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
                x, y = center_x + dx, center_y + dy
                if 0 <= x < depth_frame.get_width() and 0 <= y < depth_frame.get_height():
                    depth = depth_frame.get_distance(x, y)
                    if depth > 0:
                        depths.append(depth)
        return float(np.median(depths)) if depths else 0.0

    def draw_detections(self, image, detections, depth_frame=None):
        annotated = image.copy()
        for det in detections:
            label = det['label']
            confidence = det['confidence']
            bbox = det['bbox']
            xmin, ymin, xmax, ymax = bbox.astype(int)

            depth_text = ""
            if depth_frame is not None:
                depth = self.get_depth_at_bbox(depth_frame, bbox)
                if depth > 0:
                    depth_text = f" | {depth:.2f}m"

            color = (0, 255, 0)
            cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), color, 2)
            text = f"{label}: {confidence:.2f}{depth_text}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (xmin, ymin - th - 10), (xmin + tw, ymin), color, -1)
            cv2.putText(annotated, text, (xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return annotated

    def draw_metrics_overlay(self, image):
        """Draw live metrics panel in the top-right corner."""
        h, w = image.shape[:2]
        panel_w, panel_h = 240, 150
        overlay = image.copy()
        cv2.rectangle(overlay, (w - panel_w, 0), (w, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)

        fps = self.metrics.current_fps()
        avg_ms = self.metrics.avg_inference_ms()

        def put(text, row, color=(255, 255, 255)):
            cv2.putText(image, text, (w - panel_w + 8, 22 + row * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        put(f"FPS        : {fps:.1f}", 0, (0, 255, 0))
        put(f"Inference  : {avg_ms:.1f} ms", 1, (0, 255, 255))
        put(f"Frames     : {self.metrics.total_frames()}", 2)
        put(f"Total dets : {self.metrics.total_detections()}", 3)

        cs = self.metrics.class_summary()
        if cs:
            top_class = next(iter(cs))
            put(f"Top class  : {top_class}", 4, (255, 200, 0))

        return image

    # ── Run modes ──────────────────────────────────────────────────────────────

    def run_single_frame(self, save_path="detection_result.jpg"):
        """Process single frame, save annotated image and metrics report."""
        print("\nRunning single frame detection...")
        self.metrics.reset()

        color_image, depth_frame = None, None
        for attempt in range(5):
            print(f"Attempting to capture frame ({attempt+1}/5)...")
            color_image, depth_frame = self.get_frame(timeout_ms=10000)
            if color_image is not None:
                print("Frame captured successfully")
                break
            time.sleep(1)

        if color_image is None:
            print("Failed to get frame after all retries")
            return

        print("Running DETR inference...")
        start_time = time.time()
        detections = self.detect_objects(color_image)
        inference_time = time.time() - start_time
        self.metrics.update(inference_time, detections)

        print(f"Detection complete in {inference_time:.2f}s")
        print(f"Found {len(detections)} objects:")
        for i, det in enumerate(detections, 1):
            depth = self.get_depth_at_bbox(depth_frame, det['bbox'])
            depth_str = f"{depth:.2f}m" if depth > 0 else "N/A"
            print(f"  {i}. {det['label']}: {det['confidence']:.2f} | Distance: {depth_str}")

        annotated = self.draw_detections(color_image, detections, depth_frame)
        info_text = f"DETR | {len(detections)} objects | {inference_time:.2f}s"
        cv2.putText(annotated, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imwrite(save_path, annotated)
        print(f"Result saved to: {save_path}")

        # Auto-save report
        self.metrics.save_report()

        cv2.imshow('DETR Detection Result', annotated)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def run_continuous(self):
        """
        Run continuous object detection with metrics tracking.
          q  - quit and save report
          s  - save screenshot
          r  - save report now (without quitting)
          m  - print metrics summary to terminal
        """
        print("\nRunning continuous detection...")
        print("Controls: [q] quit+report  [s] screenshot  [r] save report  [m] print metrics")

        self.metrics.reset()
        frame_count = 0
        last_detections = []
        error_count = 0
        max_errors = 5

        try:
            while True:
                color_image, depth_frame = self.get_frame(timeout_ms=5000)
                if color_image is None:
                    print("Frame skipped")
                    continue

                if frame_count % CONFIG.get("inference_every_n_frames", 3) == 0:
                    try:
                        start_time = time.time()
                        detections = self.detect_objects(color_image)
                        inference_time = time.time() - start_time
                        last_detections = detections
                        error_count = 0

                        self.metrics.update(inference_time, detections)

                        annotated = self.draw_detections(color_image, detections, depth_frame)
                        fps = self.metrics.current_fps()
                        cv2.putText(annotated, f"DETR | {len(detections)} objects | {fps:.1f} FPS",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        annotated = self.draw_metrics_overlay(annotated)

                    except RuntimeError as e:
                        error_count += 1
                        print(f"CUDA error ({error_count}/{max_errors}): {str(e)}")
                        if self.device.type == 'cuda':
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        if error_count >= max_errors:
                            print("Too many CUDA errors, stopping...")
                            break
                        annotated = self.draw_detections(color_image, last_detections, depth_frame)
                        cv2.putText(annotated, "DETR | Recovering from error...", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    annotated = self.draw_detections(color_image, last_detections, depth_frame)
                    fps = self.metrics.current_fps()
                    cv2.putText(annotated, f"DETR | {len(last_detections)} objects | {fps:.1f} FPS",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    annotated = self.draw_metrics_overlay(annotated)

                cv2.imshow('RealSense + DETR  [q=quit  r=report  s=screenshot]', annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    fname = f"detection_{ts}.jpg"
                    cv2.imwrite(fname, annotated)
                    print(f"Screenshot saved: {fname}")
                elif key == ord('r'):
                    self.metrics.save_report()
                elif key == ord('m'):
                    self.metrics.print_summary()

                frame_count += 1

        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            self.metrics.save_report()
            self.cleanup()

    def evaluate_dataset(self, images_dir, annotations_file, iou_threshold=0.5, visualize=False):
        """
        Evaluate model on a COCO-format dataset and compute Precision/Recall/F1.
        Saves full metrics report to disk after evaluation.
        """
        print("\n" + "="*60)
        print("EVALUATING MODEL ON DATASET")
        print("="*60)

        with open(annotations_file, 'r') as f:
            coco = json.load(f)

        print(f"Images: {len(coco['images'])}")
        print(f"Annotations: {len(coco['annotations'])}")
        print(f"IoU threshold: {iou_threshold}")

        cat_id_to_name = {cat['id']: cat['name'] for cat in coco['categories']}
        img_to_anns = {}
        for ann in coco['annotations']:
            img_id = ann['image_id']
            img_to_anns.setdefault(img_id, []).append({
                'label': cat_id_to_name[ann['category_id']],
                'bbox': ann['bbox']
            })

        self.metrics.reset()

        for idx, img_info in enumerate(coco['images']):
            img_path = os.path.join(images_dir, img_info['file_name'])
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                continue

            image = cv2.imread(img_path)
            if image is None:
                print(f"Could not load: {img_path}")
                continue

            start_time = time.time()
            detections = self.detect_objects(image)
            inference_time = time.time() - start_time

            self.metrics.update(inference_time, detections)
            ground_truths = img_to_anns.get(img_info['id'], [])
            self.metrics.evaluate_frame(detections, ground_truths, iou_threshold)

            if (idx + 1) % 10 == 0 or idx == len(coco['images']) - 1:
                print(f"Processed {idx + 1}/{len(coco['images'])} images...")

            if visualize:
                annotated = self.draw_detections(image, detections)
                for gt in ground_truths:
                    x, y, w, h = [int(v) for v in gt['bbox']]
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(annotated, f"GT: {gt['label']}", (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.imshow('Evaluation (Green=Det, Blue=GT) - Press any key', annotated)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break

        if visualize:
            cv2.destroyAllWindows()

        # Save full report (includes precision_recall_f1.png)
        self.metrics.save_report()
        return self.metrics.session_summary()

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
        print("Done!")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("  RealSense D435 + DETR Object Detection  |  Metrics Report")
    print("="*60)
    print(f"  Filter: {DETECT_ONLY if DETECT_ONLY else 'All classes'}")

    try:
        print("\nSelect mode:")
        print("1. Single frame detection (requires camera)")
        print("2. Continuous detection with metrics (requires camera)")
        print("3. Evaluate on dataset (compute Precision/Recall/F1)")

        choice = input("Enter choice (1, 2, or 3): ").strip()

        if choice == "3":
            print("\n--- Dataset Evaluation Mode ---")
            images_dir = input("Enter path to images directory: ").strip()
            annotations_file = input("Enter path to COCO annotations JSON: ").strip()
            visualize = input("Visualize results? (y/n): ").strip().lower() == 'y'

            print("\nNote: Camera initialization may fail - that's OK for evaluation mode.")
            try:
                detector = RealsenseDETR(confidence_threshold=0.7)
            except Exception as cam_error:
                print(f"Camera init failed (expected): {cam_error}")
                detector = RealsenseDETR.__new__(RealsenseDETR)
                detector.confidence_threshold = 0.7
                detector.pipeline = None
                detector._cleaned_up = True
                detector.metrics = MetricsTracker()
                print("Loading DETR model...")
                detector.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
                detector.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
                detector.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                detector.model.to(detector.device)
                detector.model.eval()
                print(f"Using device: {detector.device}")

            detector.evaluate_dataset(images_dir, annotations_file,
                                      iou_threshold=0.5, visualize=visualize)
        else:
            detector = RealsenseDETR(confidence_threshold=CONFIG.get("confidence_threshold", 0.7))
            if choice == "1":
                detector.run_single_frame()
            else:
                detector.run_continuous()
            detector.cleanup()

    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
