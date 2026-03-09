#!/usr/bin/env python3
"""
RealSense D435 + CASCADE DETR
Model 1: Custom-trained bottle detector (fine-tuned DETR)
Model 2: Hard-negative rejector (trained to flag false positives)
Final detection = M1 AND NOT(M2 overlap)

Showcases the ability to fine-tune DETR on custom data and deploy
a cascade architecture to suppress false positives.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import os
import time
import csv
import json
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# Metrics Tracker
# ──────────────────────────────────────────────────────────────────────────────

class MetricsTracker:
    """
    Tracks per-frame and per-session cascade detection metrics.

    Tracked per frame:
      - Inference time for Model 1 and Model 2 separately
      - Total cascade inference time
      - M1 candidates, M2 rejections, final detections
      - Confidence scores of final detections

    Export: CSV (per-frame), JSON (session summary), matplotlib plots
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.session_start = time.time()
        self.frame_records = []
        self.confidence_history = []
        self.fps_history = []
        # Ground-truth evaluation counters
        self.true_positives  = 0
        self.false_positives = 0
        self.false_negatives = 0

    def update(self, inference_s, m1_count, m2_count, final_count, confidences):
        """
        Call once per inference frame.

        Args:
            inference_s   : total cascade inference time in seconds
            m1_count      : detections from Model 1
            m2_count      : detections from Model 2 (hard-negative flags)
            final_count   : detections after rejection
            confidences   : list of confidence scores for final detections
        """
        frame_num = len(self.frame_records) + 1
        ms = inference_s * 1000

        self.fps_history.append(inference_s)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        fps = 1.0 / (sum(self.fps_history) / len(self.fps_history))

        self.confidence_history.extend(confidences)

        self.frame_records.append({
            'frame': frame_num,
            'timestamp': round(time.time() - self.session_start, 3),
            'inference_ms': round(ms, 2),
            'fps': round(fps, 2),
            'm1_candidates': m1_count,
            'm2_rejections': m2_count,
            'final_detections': final_count,
            'rejected': m1_count - final_count,
            'avg_confidence': round(float(np.mean(confidences)), 4) if confidences else 0.0,
        })

    # ── Ground-truth evaluation ────────────────────────────────────────────────

    def _iou(self, b1, b2):
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        if x2 < x1 or y2 < y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        return inter / (a1 + a2 - inter) if (a1+a2-inter) > 0 else 0.0

    def evaluate_frame(self, detections, ground_truths, iou_threshold=0.5):
        """Match detections to ground truth bboxes and accumulate TP/FP/FN."""
        matched = set()
        for det in detections:
            best_iou, best_idx = 0, -1
            for i, gt in enumerate(ground_truths):
                if i in matched:
                    continue
                iou = self._iou(det['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_iou >= iou_threshold and best_idx >= 0:
                self.true_positives += 1
                matched.add(best_idx)
            else:
                self.false_positives += 1
        self.false_negatives += len(ground_truths) - len(matched)

    def get_precision(self):
        t = self.true_positives + self.false_positives
        return self.true_positives / t if t > 0 else 0.0

    def get_recall(self):
        t = self.true_positives + self.false_negatives
        return self.true_positives / t if t > 0 else 0.0

    def get_f1(self):
        p, r = self.get_precision(), self.get_recall()
        return 2*p*r/(p+r) if (p+r) > 0 else 0.0

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

    def total_final_detections(self):
        return sum(r['final_detections'] for r in self.frame_records)

    def total_rejected(self):
        return sum(r['rejected'] for r in self.frame_records)

    def rejection_rate(self):
        total_m1 = sum(r['m1_candidates'] for r in self.frame_records)
        if total_m1 == 0:
            return 0.0
        return round(self.total_rejected() / total_m1 * 100, 1)

    def avg_confidence(self):
        if not self.confidence_history:
            return 0.0
        return round(float(np.mean(self.confidence_history)), 4)

    def session_summary(self):
        duration = round(time.time() - self.session_start, 2)
        summary = {
            'session_duration_s': duration,
            'total_frames': self.total_frames(),
            'avg_fps': self.avg_fps(),
            'avg_inference_ms': self.avg_inference_ms(),
            'total_m1_candidates': sum(r['m1_candidates'] for r in self.frame_records),
            'total_m2_rejections': self.total_rejected(),
            'total_final_detections': self.total_final_detections(),
            'rejection_rate_pct': self.rejection_rate(),
            'avg_final_confidence': self.avg_confidence(),
        }
        if self.true_positives + self.false_positives + self.false_negatives > 0:
            summary['evaluation'] = {
                'precision':       round(self.get_precision(), 4),
                'recall':          round(self.get_recall(),    4),
                'f1_score':        round(self.get_f1(),        4),
                'true_positives':  self.true_positives,
                'false_positives': self.false_positives,
                'false_negatives': self.false_negatives,
            }
        return summary

    def print_summary(self):
        s = self.session_summary()
        print("\n" + "=" * 60)
        print("  CASCADE DETR  |  METRICS SUMMARY")
        print("=" * 60)
        print(f"  Session duration      : {s['session_duration_s']} s")
        print(f"  Frames processed      : {s['total_frames']}")
        print(f"  Average FPS           : {s['avg_fps']}")
        print(f"  Avg inference time    : {s['avg_inference_ms']} ms")
        print(f"  M1 candidates (total) : {s['total_m1_candidates']}")
        print(f"  M2 rejected  (total)  : {s['total_m2_rejections']}")
        print(f"  Final detections      : {s['total_final_detections']}")
        print(f"  Rejection rate        : {s['rejection_rate_pct']} %")
        print(f"  Avg final confidence  : {s['avg_final_confidence']:.4f}")
        if 'evaluation' in s:
            ev = s['evaluation']
            print(f"\n  Ground-truth Evaluation (IoU >= 0.5):")
            print(f"  Precision  : {ev['precision']:.4f}")
            print(f"  Recall     : {ev['recall']:.4f}")
            print(f"  F1 Score   : {ev['f1_score']:.4f}")
            print(f"  TP: {ev['true_positives']}  FP: {ev['false_positives']}  FN: {ev['false_negatives']}")
        print("=" * 60)

    # ── Export ─────────────────────────────────────────────────────────────────

    def export_csv(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'frames.csv')
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'timestamp_s', 'inference_ms', 'fps',
                             'm1_candidates', 'm2_rejections', 'final_detections',
                             'rejected', 'avg_confidence'])
            for r in self.frame_records:
                writer.writerow([r['frame'], r['timestamp'], r['inference_ms'], r['fps'],
                                 r['m1_candidates'], r['m2_rejections'], r['final_detections'],
                                 r['rejected'], r['avg_confidence']])
        print(f"  Saved: {path}")

    def export_json(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'session_summary.json')
        with open(path, 'w') as f:
            json.dump(self.session_summary(), f, indent=2)
        print(f"  Saved: {path}")

    def generate_plots(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        if not self.frame_records:
            print("  No data to plot.")
            return

        frames = [r['frame'] for r in self.frame_records]

        # ── 1. M1 vs Final detections per frame ───────────────────────────────
        m1_vals = [r['m1_candidates'] for r in self.frame_records]
        final_vals = [r['final_detections'] for r in self.frame_records]
        rejected_vals = [r['rejected'] for r in self.frame_records]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(frames, m1_vals, color='orange', linewidth=1.5, label='M1 Candidates')
        ax.plot(frames, final_vals, color='green', linewidth=1.5, label='Final (after M2)')
        ax.fill_between(frames, final_vals, m1_vals, alpha=0.2, color='red', label='Rejected by M2')
        ax.set_title('Cascade Detection: M1 Candidates vs Final Detections', fontsize=13, fontweight='bold')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Detection Count')
        ax.legend()
        plt.tight_layout()
        path = os.path.join(output_dir, 'cascade_detections_per_frame.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")

        # ── 2. Rejection rate summary bar ────────────────────────────────────
        s = self.session_summary()
        categories = ['M1 Candidates', 'M2 Rejected', 'Final Detections']
        values = [s['total_m1_candidates'], s['total_m2_rejections'], s['total_final_detections']]
        colors = ['orange', 'tomato', 'mediumseagreen']

        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(categories, values, color=colors, edgecolor='white')
        ax.bar_label(bars, padding=3, fontsize=11)
        ax.set_title(f'Cascade Summary  |  Rejection Rate: {s["rejection_rate_pct"]}%',
                     fontsize=13, fontweight='bold')
        ax.set_ylabel('Total Count')
        plt.tight_layout()
        path = os.path.join(output_dir, 'cascade_summary.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")

        # ── 3. Confidence score distribution ─────────────────────────────────
        if self.confidence_history:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(self.confidence_history, bins=20, range=(0, 1),
                    color='steelblue', edgecolor='white')
            ax.axvline(float(np.mean(self.confidence_history)), color='red', linestyle='--',
                       label=f'Mean = {np.mean(self.confidence_history):.3f}')
            ax.set_title('Final Detection Confidence Distribution', fontsize=13, fontweight='bold')
            ax.set_xlabel('Confidence Score')
            ax.set_ylabel('Frequency')
            ax.legend()
            plt.tight_layout()
            path = os.path.join(output_dir, 'confidence_distribution.png')
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"  Saved: {path}")

        # ── 4. FPS & inference time ───────────────────────────────────────────
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
        ax1.set_title('FPS & Inference Time over Frames', fontsize=13, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(output_dir, 'fps_and_inference_time.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")

        # ── 5. Precision / Recall / F1 (only if GT evaluation was done) ──────
        if self.true_positives + self.false_positives + self.false_negatives > 0:
            metrics = ['Precision', 'Recall', 'F1 Score']
            values  = [self.get_precision(), self.get_recall(), self.get_f1()]
            colors  = ['steelblue', 'darkorange', 'mediumseagreen']
            fig, ax = plt.subplots(figsize=(6, 5))
            bars = ax.bar(metrics, values, color=colors, edgecolor='white')
            ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=11)
            ax.set_ylim(0, 1.1)
            ax.set_title('Precision / Recall / F1 Score  (Cascade DETR)',
                         fontsize=13, fontweight='bold')
            ax.set_ylabel('Score')
            plt.tight_layout()
            path = os.path.join(output_dir, 'precision_recall_f1.png')
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"  Saved: {path}")

    def save_report(self, output_dir=None):
        if output_dir is None:
            output_dir = f"report_cascade_{time.strftime('%Y%m%d_%H%M%S')}"
        print(f"\nSaving cascade report to: {output_dir}/")
        self.export_csv(output_dir)
        self.export_json(output_dir)
        self.generate_plots(output_dir)
        self.print_summary()
        print(f"\nReport complete -> {output_dir}/")
        return output_dir


# ──────────────────────────────────────────────────────────────────────────────
# Cascade Detector
# ──────────────────────────────────────────────────────────────────────────────

class CascadeBottleDetector:

    def __init__(self,
                 confidence_threshold_1=0.70,
                 confidence_threshold_2=0.85,
                 reject_iou_threshold=0.6,
                 no_camera=False):

        print("=" * 70)
        print("  CASCADE DETR  |  Custom-Trained Bottle + Hard-Negative Rejector")
        print("=" * 70)

        self.confidence_threshold_1 = confidence_threshold_1
        self.confidence_threshold_2 = confidence_threshold_2
        self.reject_iou_threshold = reject_iou_threshold

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # ── Model 1: Custom bottle detector ───────────────────────────────────
        model1_path = "./water_bottle_model/huggingface_model"
        if not os.path.exists(model1_path):
            raise FileNotFoundError(f"Model 1 not found at: {model1_path}")

        print("\nLoading Model 1 (custom bottle detector)...")
        self.processor_1 = DetrImageProcessor.from_pretrained(model1_path)
        self.model_1 = DetrForObjectDetection.from_pretrained(model1_path)
        self.model_1.to(self.device)
        self.model_1.eval()
        print("  Labels:", self.model_1.config.id2label)

        # ── Model 2: Hard-negative rejector ───────────────────────────────────
        model2_path = "./water_bottle_model_hard_negatives/huggingface_model"
        if not os.path.exists(model2_path):
            raise FileNotFoundError(f"Model 2 not found at: {model2_path}")

        print("\nLoading Model 2 (hard-negative rejector)...")
        self.processor_2 = DetrImageProcessor.from_pretrained(model2_path)
        self.model_2 = DetrForObjectDetection.from_pretrained(model2_path)
        self.model_2.to(self.device)
        self.model_2.eval()
        print("  Labels:", self.model_2.config.id2label)

        self.metrics = MetricsTracker()
        self.pipeline = None
        if not no_camera:
            self.initialize_camera()

    # ── Camera ─────────────────────────────────────────────────────────────────

    def initialize_camera(self):
        print("\nInitializing RealSense D435...")
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(config)

        # Warm up
        for _ in range(20):
            self.pipeline.wait_for_frames()
        print("Camera ready.")

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame:
            return None, None
        return np.asanyarray(color_frame.get_data()), depth_frame

    def get_depth_at_bbox(self, depth_frame, bbox):
        """Get median depth within bounding box center region."""
        if depth_frame is None:
            return 0.0
        xmin, ymin, xmax, ymax = [int(v) for v in bbox]
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(depth_frame.get_width(), xmax)
        ymax = min(depth_frame.get_height(), ymax)
        center_x = (xmin + xmax) // 2
        center_y = (ymin + ymax) // 2
        depths = []
        for dx in range(-8, 8):
            for dy in range(-8, 8):
                x, y = center_x + dx, center_y + dy
                if 0 <= x < depth_frame.get_width() and 0 <= y < depth_frame.get_height():
                    d = depth_frame.get_distance(x, y)
                    if d > 0:
                        depths.append(d)
        return float(np.median(depths)) if depths else 0.0

    # ── Detection ──────────────────────────────────────────────────────────────

    def detect_with_model(self, image, model, processor, threshold):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        inputs = processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=threshold
            )[0]

        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            class_name = model.config.id2label[label.item()]
            if class_name in ["no-object", "background"]:
                continue
            detections.append({
                "label": class_name,
                "confidence": score.item(),
                "bbox": box.cpu().numpy()
            })
        return detections

    def validate_bottle_heuristics(self, image, bbox,
                                    min_aspect_ratio=1.3,
                                    max_aspect_ratio=6.0,
                                    min_blue_ratio=0.10):
        """
        Heuristic validator — third layer after M1 + M2 cascade.

        Checks two conditions on the detected bounding box region:
          1. Aspect ratio  : height/width must be in [min, max] (bottles are tall)
          2. Blue ratio    : at least min_blue_ratio of pixels must be blue (HSV)

        Args:
            image           : BGR frame
            bbox            : [xmin, ymin, xmax, ymax]
            min_aspect_ratio: minimum height/width ratio
            max_aspect_ratio: maximum height/width ratio
            min_blue_ratio  : fraction of pixels that must be blue (0–1)

        Returns:
            (passed: bool, reason: str)
        """
        xmin, ymin, xmax, ymax = [int(v) for v in bbox]
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(image.shape[1], xmax)
        ymax = min(image.shape[0], ymax)

        width  = xmax - xmin
        height = ymax - ymin

        if width <= 0 or height <= 0:
            return False, "invalid bbox"

        # ── Aspect ratio check ─────────────────────────────────────────────────
        aspect = height / width
        if not (min_aspect_ratio <= aspect <= max_aspect_ratio):
            return False, f"aspect {aspect:.2f} out of range [{min_aspect_ratio},{max_aspect_ratio}]"

        # ── Blue color ratio check (HSV) ───────────────────────────────────────
        roi = image[ymin:ymax, xmin:xmax]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Blue hue range in OpenCV HSV: ~100–130
        lower_blue = np.array([90,  50,  50])
        upper_blue = np.array([135, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_ratio = blue_mask.sum() / 255 / (width * height)

        if blue_ratio < min_blue_ratio:
            return False, f"blue ratio {blue_ratio:.2f} < {min_blue_ratio}"

        return True, "ok"

    def compute_iou(self, box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)
        if xi_max < xi_min or yi_max < yi_min:
            return 0.0
        inter = (xi_max - xi_min) * (yi_max - yi_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def detect_bottles(self, image):
        """
        Three-layer cascade:
          1. M1 (custom DETR) proposes bottle candidates
          2. M2 (hard-negative DETR) rejects candidates that overlap flagged regions
          3. Heuristic validator rejects remaining candidates that fail
             aspect-ratio or blue-color checks

        Returns:
            m1_detections      : all M1 candidates
            m2_detections      : M2 hard-negative regions
            heuristic_rejected : passed M2 but failed heuristic (shown separately)
            final_detections   : confirmed bottles
        """
        detections_1 = self.detect_with_model(
            image, self.model_1, self.processor_1, self.confidence_threshold_1
        )
        detections_2 = self.detect_with_model(
            image, self.model_2, self.processor_2, self.confidence_threshold_2
        )

        after_cascade = []
        for det1 in detections_1:
            rejected_by_m2 = any(
                self.compute_iou(det1["bbox"], det2["bbox"]) > self.reject_iou_threshold
                for det2 in detections_2
            )
            if not rejected_by_m2:
                after_cascade.append(det1)

        # Layer 3: heuristic validation
        final = []
        heuristic_rejected = []
        for det in after_cascade:
            passed, reason = self.validate_bottle_heuristics(image, det["bbox"])
            if passed:
                final.append(det)
            else:
                det["reject_reason"] = reason
                heuristic_rejected.append(det)

        return detections_1, detections_2, heuristic_rejected, final

    # ── Drawing ────────────────────────────────────────────────────────────────

    def draw_detections(self, image, m1_detections, m2_detections,
                        heuristic_rejected, final_detections, depth_frame=None):
        """
        Draw four layers:
          - Orange dashed  : M1 candidates rejected by M2
          - Yellow dashed  : Passed M2 but rejected by heuristic (aspect/color)
          - Red overlay    : M2 hard-negative regions
          - Green solid    : Final confirmed bottles (+ depth)
        """
        annotated = image.copy()

        final_bboxes = set(tuple(d["bbox"].astype(int)) for d in final_detections)
        heuristic_bboxes = set(tuple(d["bbox"].astype(int)) for d in heuristic_rejected)

        def draw_dashed_rect(img, bbox, color, label):
            xmin, ymin, xmax, ymax = bbox
            for x in range(xmin, xmax, 10):
                cv2.line(img, (x, ymin), (min(x+6, xmax), ymin), color, 2)
                cv2.line(img, (x, ymax), (min(x+6, xmax), ymax), color, 2)
            for y in range(ymin, ymax, 10):
                cv2.line(img, (xmin, y), (xmin, min(y+6, ymax)), color, 2)
                cv2.line(img, (xmax, y), (xmax, min(y+6, ymax)), color, 2)
            cv2.putText(img, label, (xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        # M1 candidates rejected by M2 (orange dashed)
        for det in m1_detections:
            bbox = tuple(det["bbox"].astype(int))
            if bbox not in final_bboxes and bbox not in heuristic_bboxes:
                draw_dashed_rect(annotated, bbox, (0, 165, 255), "M2-REJECTED")

        # Passed M2 but failed heuristic (yellow dashed)
        for det in heuristic_rejected:
            bbox = tuple(det["bbox"].astype(int))
            reason = det.get("reject_reason", "heuristic")
            draw_dashed_rect(annotated, bbox, (0, 220, 220), f"BAD-SHAPE/COLOR")

        # M2 hard-negative regions (red semi-transparent)
        overlay = annotated.copy()
        for det in m2_detections:
            xmin, ymin, xmax, ymax = det["bbox"].astype(int)
            cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)
        for det in m2_detections:
            xmin, ymin, xmax, ymax = det["bbox"].astype(int)
            cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), (0, 0, 200), 1)
            cv2.putText(annotated, f"HN {det['confidence']:.2f}", (xmin, ymax + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 200), 1)

        # Final confirmed detections (green)
        for det in final_detections:
            xmin, ymin, xmax, ymax = det["bbox"].astype(int)
            depth_text = ""
            if depth_frame is not None:
                depth = self.get_depth_at_bbox(depth_frame, det["bbox"])
                if depth > 0:
                    depth_text = f" | {depth:.2f}m"
            cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), (0, 220, 0), 3)
            label_text = f"Bottle {det['confidence']:.2f}{depth_text}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (xmin, ymin - th - 10), (xmin + tw, ymin), (0, 220, 0), -1)
            cv2.putText(annotated, label_text, (xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return annotated

    def draw_metrics_overlay(self, image):
        """Draw live metrics panel in the top-right corner."""
        h, w = image.shape[:2]
        panel_w, panel_h = 260, 165
        overlay = image.copy()
        cv2.rectangle(overlay, (w - panel_w, 0), (w, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)

        fps = self.metrics.current_fps()
        avg_ms = self.metrics.avg_inference_ms()

        def put(text, row, color=(255, 255, 255)):
            cv2.putText(image, text, (w - panel_w + 8, 22 + row * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)

        put(f"FPS          : {fps:.1f}", 0, (0, 255, 0))
        put(f"Inference    : {avg_ms:.1f} ms", 1, (0, 255, 255))
        put(f"Frames       : {self.metrics.total_frames()}", 2)
        put(f"Confirmed    : {self.metrics.total_final_detections()}", 3, (0, 220, 0))
        put(f"Rejected     : {self.metrics.total_rejected()}", 4, (0, 165, 255))
        put(f"Reject rate  : {self.metrics.rejection_rate()} %", 5, (0, 165, 255))

        return image

    def draw_legend(self, image):
        """Draw color legend in the bottom-left."""
        h, w = image.shape[:2]
        items = [
            ((0, 220, 0),   "Confirmed bottle"),
            ((0, 165, 255), "Rejected by M2 (hard-negative)"),
            ((0, 220, 220), "Rejected by heuristic (shape/color)"),
            ((0, 0, 200),   "Hard-negative region (M2)"),
        ]
        y = h - 10 - len(items) * 22
        for color, label in items:
            cv2.rectangle(image, (10, y - 12), (24, y + 2), color, -1)
            cv2.putText(image, label, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 22
        return image

    # ── Dataset evaluation ─────────────────────────────────────────────────────

    def evaluate_dataset(self, annotations_json, images_dir, iou_threshold=0.5):
        """
        Evaluate Precision / Recall / F1 against COCO-format ground truth.

        Runs the full 3-layer cascade (M1 → M2 → heuristic) on each image
        from disk and compares final detections to GT bboxes from All-1.json.

        Args:
            annotations_json : path to All-1.json (COCO format)
            images_dir       : directory containing the RGB images
            iou_threshold    : IoU threshold for a TP match (default 0.5)
        """
        print("\n" + "=" * 60)
        print("  CASCADE DETR  |  DATASET EVALUATION  (no camera)")
        print("=" * 60)
        print(f"  Annotations : {annotations_json}")
        print(f"  Images dir  : {images_dir}")
        print(f"  IoU thresh  : {iou_threshold}")

        with open(annotations_json) as f:
            coco = json.load(f)

        id2file = {img['id']: img['file_name'] for img in coco['images']}

        gt_by_image = defaultdict(list)
        for ann in coco['annotations']:
            if 'bbox' not in ann:
                continue
            x, y, w, h = ann['bbox']
            gt_by_image[ann['image_id']].append({'bbox': [x, y, x + w, y + h]})

        self.metrics.reset()
        total_images = len(id2file)
        print(f"\nRunning on {total_images} images...\n")

        for idx, (img_id, file_name) in enumerate(id2file.items(), 1):
            img_path = os.path.join(images_dir, file_name)
            if not os.path.exists(img_path):
                img_path = os.path.join(images_dir, os.path.basename(file_name))
            if not os.path.exists(img_path):
                print(f"  [{idx}/{total_images}] SKIP (not found): {file_name}")
                continue

            bgr = cv2.imread(img_path)
            if bgr is None:
                print(f"  [{idx}/{total_images}] SKIP (unreadable): {file_name}")
                continue

            start = time.time()
            m1, m2, hr, final = self.detect_bottles(bgr)
            elapsed = time.time() - start

            confs = [d['confidence'] for d in final]
            self.metrics.update(elapsed, len(m1), len(m2), len(final), confs)

            # Convert final detections to the format evaluate_frame expects
            final_for_eval = [
                {'bbox': list(d['bbox'].astype(float))} for d in final
            ]
            ground_truths = gt_by_image.get(img_id, [])
            self.metrics.evaluate_frame(final_for_eval, ground_truths, iou_threshold)

            if idx % 50 == 0 or idx == total_images:
                print(f"  [{idx}/{total_images}]  "
                      f"P={self.metrics.get_precision():.3f}  "
                      f"R={self.metrics.get_recall():.3f}  "
                      f"F1={self.metrics.get_f1():.3f}")

        report_dir = self.metrics.save_report()
        print(f"\nEvaluation complete → {report_dir}/")

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self):
        """
        Live cascade detection loop.
          q  - quit and save report
          s  - save screenshot
          r  - save report now
          m  - print metrics to terminal
        """
        print("\nRunning cascade bottle detection.")
        print("Controls: [q] quit+report  [s] screenshot  [r] save report  [m] print metrics")
        print("\nVisualization guide:")
        print("  Green box        = confirmed bottle (all 3 layers passed)")
        print("  Orange dashed    = M1 candidate rejected by M2 (hard-negative model)")
        print("  Yellow dashed    = passed M2 but rejected by heuristic (bad shape or not blue enough)")
        print("  Red overlay      = hard-negative region flagged by M2")

        self.metrics.reset()
        frame_count = 0
        last_m1, last_m2, last_hr, last_final = [], [], [], []

        try:
            while True:
                frame, depth_frame = self.get_frame()
                if frame is None:
                    continue

                if frame_count % 3 == 0:
                    start = time.time()
                    m1, m2, hr, final = self.detect_bottles(frame)
                    elapsed = time.time() - start

                    last_m1, last_m2, last_hr, last_final = m1, m2, hr, final
                    confs = [d['confidence'] for d in final]
                    self.metrics.update(elapsed, len(m1), len(m2), len(final), confs)

                annotated = self.draw_detections(
                    frame.copy(), last_m1, last_m2, last_hr, last_final, depth_frame
                )

                fps = self.metrics.current_fps()
                total_rejected = len(last_m1) - len(last_final)
                header = (f"Custom DETR Cascade | M1:{len(last_m1)} "
                          f"Rejected:{total_rejected} "
                          f"Final:{len(last_final)} | {fps:.1f} FPS")
                cv2.putText(annotated, header, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

                annotated = self.draw_metrics_overlay(annotated)
                annotated = self.draw_legend(annotated)

                cv2.imshow("Cascade DETR  [q=quit  r=report  s=screenshot  m=metrics]", annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    fname = f"cascade_{ts}.jpg"
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
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("Done!")


# ──────────────────────────────────────────────────────────────────────────────

def main():
    ANNOTATIONS_JSON = "./my_dataset/All-1.json"
    IMAGES_DIR       = "./my_dataset/rgb"

    print("=" * 60)
    print("  CASCADE DETR  |  Bottle Detection")
    print("=" * 60)
    print("\nSelect mode:")
    print("1. Live detection  (camera)")
    print("2. Dataset evaluation  (Precision / Recall / F1  –  no camera)")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "2":
        detector = CascadeBottleDetector(
            confidence_threshold_1=0.70,
            confidence_threshold_2=0.85,
            reject_iou_threshold=0.6,
            no_camera=True,
        )
        detector.evaluate_dataset(
            annotations_json=ANNOTATIONS_JSON,
            images_dir=IMAGES_DIR,
            iou_threshold=0.5,
        )
    else:
        detector = CascadeBottleDetector(
            confidence_threshold_1=0.70,
            confidence_threshold_2=0.85,
            reject_iou_threshold=0.6,
        )
        detector.run()


if __name__ == "__main__":
    main()
