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
import csv
import json
import os
from collections import defaultdict
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# Metrics Tracker
# ──────────────────────────────────────────────────────────────────────────────

class MetricsTracker:
    """
    Tracks per-frame and per-session SegFormer detection metrics.

    Per frame:
      - Inference time (ms)
      - FPS (rolling 30-frame average)
      - Number of detections
      - Confidence scores
      - Depth values
      - Pixel area of each detection

    Export: CSV, JSON, matplotlib plots (same structure as DETR reports)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.session_start   = time.time()
        self.frame_records   = []
        self.fps_history     = []
        self.all_confidences = []
        self.all_depths      = []
        self.all_areas       = []
        # Ground-truth evaluation counters
        self.true_positives  = 0
        self.false_positives = 0
        self.false_negatives = 0

    # ── Per-frame update ───────────────────────────────────────────────────────

    def update(self, inference_s, detections):
        """
        Call once per inference frame.
        detections: list of dicts with keys confidence, depth, area, bbox
        """
        frame_num = len(self.frame_records) + 1
        ms = inference_s * 1000

        self.fps_history.append(inference_s)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        fps = 1.0 / (sum(self.fps_history) / len(self.fps_history))

        confs  = [d['confidence'] for d in detections]
        depths = [d['depth']      for d in detections if d['depth'] > 0]
        areas  = [d['area']       for d in detections]

        self.all_confidences.extend(confs)
        self.all_depths.extend(depths)
        self.all_areas.extend(areas)

        self.frame_records.append({
            'frame':          frame_num,
            'timestamp':      round(time.time() - self.session_start, 3),
            'inference_ms':   round(ms, 2),
            'fps':            round(fps, 2),
            'n_detections':   len(detections),
            'avg_confidence': round(float(np.mean(confs)),  4) if confs  else 0.0,
            'avg_depth_m':    round(float(np.mean(depths)), 3) if depths else 0.0,
            'avg_area_px':    round(float(np.mean(areas)),  1) if areas  else 0.0,
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

    # ── Summaries ──────────────────────────────────────────────────────────────

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

    def session_summary(self):
        duration = round(time.time() - self.session_start, 2)
        summary = {
            'session_duration_s':   duration,
            'total_frames':         len(self.frame_records),
            'avg_fps':              self.avg_fps(),
            'avg_inference_ms':     self.avg_inference_ms(),
            'total_detections':     self.total_detections(),
            'avg_confidence':       round(float(np.mean(self.all_confidences)), 4) if self.all_confidences else 0.0,
            'avg_depth_m':          round(float(np.mean(self.all_depths)),      3) if self.all_depths      else 0.0,
            'avg_area_px':          round(float(np.mean(self.all_areas)),        1) if self.all_areas       else 0.0,
        }
        if self.true_positives + self.false_positives + self.false_negatives > 0:
            summary['evaluation'] = {
                'precision':        round(self.get_precision(), 4),
                'recall':           round(self.get_recall(),    4),
                'f1_score':         round(self.get_f1(),        4),
                'true_positives':   self.true_positives,
                'false_positives':  self.false_positives,
                'false_negatives':  self.false_negatives,
            }
        return summary

    def print_summary(self):
        s = self.session_summary()
        print("\n" + "=" * 60)
        print("  SEGFORMER DETECTION  |  METRICS SUMMARY")
        print("=" * 60)
        print(f"  Session duration   : {s['session_duration_s']} s")
        print(f"  Frames processed   : {s['total_frames']}")
        print(f"  Average FPS        : {s['avg_fps']}")
        print(f"  Avg inference time : {s['avg_inference_ms']} ms")
        print(f"  Total detections   : {s['total_detections']}")
        print(f"  Avg confidence     : {s['avg_confidence']:.4f}")
        print(f"  Avg depth          : {s['avg_depth_m']:.3f} m")
        print(f"  Avg blob area      : {s['avg_area_px']:.0f} px")
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
                             'n_detections', 'avg_confidence', 'avg_depth_m', 'avg_area_px'])
            for r in self.frame_records:
                writer.writerow([r['frame'], r['timestamp'], r['inference_ms'], r['fps'],
                                 r['n_detections'], r['avg_confidence'],
                                 r['avg_depth_m'], r['avg_area_px']])
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

        frames   = [r['frame']        for r in self.frame_records]
        fps_vals = [r['fps']          for r in self.frame_records]
        ms_vals  = [r['inference_ms'] for r in self.frame_records]
        det_vals = [r['n_detections'] for r in self.frame_records]

        # 1. FPS & inference time
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(frames, fps_vals, color='steelblue', linewidth=1.5, label='FPS')
        ax1.set_xlabel('Frame'); ax1.set_ylabel('FPS', color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue')
        ax2 = ax1.twinx()
        ax2.plot(frames, ms_vals, color='tomato', linewidth=1.5, linestyle='--', label='Inference (ms)')
        ax2.set_ylabel('Inference Time (ms)', color='tomato')
        ax2.tick_params(axis='y', labelcolor='tomato')
        lines1, lab1 = ax1.get_legend_handles_labels()
        lines2, lab2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1+lines2, lab1+lab2, loc='upper right')
        ax1.set_title('FPS & Inference Time over Frames', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fps_and_inference_time.png'), dpi=150)
        plt.close()
        print(f"  Saved: fps_and_inference_time.png")

        # 2. Detections per frame
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(frames, det_vals, color='mediumseagreen', width=0.8)
        ax.set_title('Bottle Detections per Frame', fontsize=13, fontweight='bold')
        ax.set_xlabel('Frame'); ax.set_ylabel('Detection Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'detections_per_frame.png'), dpi=150)
        plt.close()
        print(f"  Saved: detections_per_frame.png")

        # 3. Confidence score distribution
        if self.all_confidences:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(self.all_confidences, bins=20, range=(0, 1),
                    color='darkorange', edgecolor='white')
            ax.axvline(float(np.mean(self.all_confidences)), color='red', linestyle='--',
                       label=f"Mean = {np.mean(self.all_confidences):.3f}")
            ax.set_title('Confidence Score Distribution', fontsize=13, fontweight='bold')
            ax.set_xlabel('Confidence'); ax.set_ylabel('Frequency')
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'), dpi=150)
            plt.close()
            print(f"  Saved: confidence_distribution.png")

        # 4. Depth distribution
        if self.all_depths:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(self.all_depths, bins=20, color='steelblue', edgecolor='white')
            ax.axvline(float(np.mean(self.all_depths)), color='red', linestyle='--',
                       label=f"Mean = {np.mean(self.all_depths):.2f} m")
            ax.set_title('Depth Distribution of Detections', fontsize=13, fontweight='bold')
            ax.set_xlabel('Depth (m)'); ax.set_ylabel('Frequency')
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'depth_distribution.png'), dpi=150)
            plt.close()
            print(f"  Saved: depth_distribution.png")

        # 5. Precision/Recall/F1 (only if GT evaluation was done)
        if self.true_positives + self.false_positives + self.false_negatives > 0:
            metrics = ['Precision', 'Recall', 'F1 Score']
            values  = [self.get_precision(), self.get_recall(), self.get_f1()]
            colors  = ['steelblue', 'darkorange', 'mediumseagreen']
            fig, ax = plt.subplots(figsize=(6, 5))
            bars = ax.bar(metrics, values, color=colors, edgecolor='white')
            ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=11)
            ax.set_ylim(0, 1.1)
            ax.set_title('Precision / Recall / F1 Score', fontsize=13, fontweight='bold')
            ax.set_ylabel('Score')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'precision_recall_f1.png'), dpi=150)
            plt.close()
            print(f"  Saved: precision_recall_f1.png")

    def save_report(self, output_dir=None):
        if output_dir is None:
            output_dir = f"segformer_report_{time.strftime('%Y%m%d_%H%M%S')}"
        print(f"\nSaving report to: {output_dir}/")
        self.export_csv(output_dir)
        self.export_json(output_dir)
        self.generate_plots(output_dir)
        self.print_summary()
        print(f"\nReport complete → {output_dir}/")
        return output_dir


class SegFormerBottleDetector:
    """
    Bottle detection using SegFormer segmentation
    Converts pixel masks to bounding boxes
    """
    
    def __init__(
        self,
        model_path,
        confidence_threshold=0.5,
        min_area=500,  # Minimum blob area in pixels
        no_camera=False,
    ):
        """
        Args:
            model_path: Path to trained SegFormer model
            confidence_threshold: Confidence threshold for segmentation
            min_area: Minimum area for valid detection
            no_camera: Skip RealSense init (use for offline dataset evaluation)
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
        if not no_camera:
            self.initialize_camera()

        # Metrics
        self.metrics = MetricsTracker()

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
        self.metrics.update(inference_time, detections)

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

        self.metrics.save_report()

        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def _draw_metrics_overlay(self, image):
        """Draw live metrics panel in top-right corner (matches DETR style)."""
        h, w = image.shape[:2]
        panel_w, panel_h = 260, 160
        overlay = image.copy()
        cv2.rectangle(overlay, (w - panel_w, 0), (w, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)

        fps    = self.metrics.current_fps()
        avg_ms = self.metrics.avg_inference_ms()
        frames = len(self.metrics.frame_records)
        total  = self.metrics.total_detections()
        avg_conf  = round(float(np.mean(self.metrics.all_confidences)), 3) if self.metrics.all_confidences else 0.0
        avg_depth = round(float(np.mean(self.metrics.all_depths)),      2) if self.metrics.all_depths      else 0.0

        def put(text, row, color=(255, 255, 255)):
            cv2.putText(image, text, (w - panel_w + 8, 22 + row * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)

        put(f"FPS        : {fps:.1f}",      0, (0, 255, 0))
        put(f"Inference  : {avg_ms:.1f} ms", 1, (0, 255, 255))
        put(f"Frames     : {frames}",        2)
        put(f"Total dets : {total}",         3)
        put(f"Avg conf   : {avg_conf:.3f}",  4, (255, 200, 0))
        put(f"Avg depth  : {avg_depth:.2f}m",5, (200, 200, 255))
        return image

    def run_continuous(self):
        """Run continuous detection"""
        print("\nRunning continuous detection...")
        print("Controls: [q] quit+report  [s] screenshot  [r] save report  [m] print metrics")

        self.metrics.reset()
        frame_count    = 0
        last_detections = []
        last_mask       = None

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

                fps = self.metrics.current_fps()
                cv2.putText(annotated,
                            f"SegFormer | {len(last_detections)} bottles | {fps:.1f} FPS",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                annotated = self._draw_metrics_overlay(annotated)

                cv2.imshow('SegFormer Detection  [q=quit  r=report  s=screenshot]', annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename  = f"segformer_detection_{timestamp}.jpg"
                    cv2.imwrite(filename, annotated)
                    print(f"Saved: {filename}")
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
    
    def evaluate_dataset(self, annotations_json, images_dir, iou_threshold=0.5):
        """
        Evaluate Precision / Recall / F1 against COCO-format ground truth.

        Args:
            annotations_json : path to All-1.json (COCO polygon format)
            images_dir       : directory containing the RGB images
            iou_threshold    : IoU overlap threshold for a TP match (default 0.5)
        """
        print("\n" + "=" * 60)
        print("  DATASET EVALUATION  (no camera needed)")
        print("=" * 60)
        print(f"  Annotations : {annotations_json}")
        print(f"  Images dir  : {images_dir}")
        print(f"  IoU thresh  : {iou_threshold}")

        with open(annotations_json) as f:
            coco = json.load(f)

        # Build image_id → file_name map
        id2file = {img['id']: img['file_name'] for img in coco['images']}

        # Build image_id → list of GT bboxes ([x1,y1,x2,y2])
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
                # Try just the basename
                img_path = os.path.join(images_dir, os.path.basename(file_name))
            if not os.path.exists(img_path):
                print(f"  [{idx}/{total_images}] SKIP (not found): {file_name}")
                continue

            bgr = cv2.imread(img_path)
            if bgr is None:
                print(f"  [{idx}/{total_images}] SKIP (unreadable): {file_name}")
                continue

            h, w = bgr.shape[:2]
            depth_dummy = np.zeros((h, w), dtype=np.float32)

            detections, _ = self.detect(bgr, depth_dummy)

            ground_truths = gt_by_image.get(img_id, [])
            self.metrics.evaluate_frame(detections, ground_truths, iou_threshold)

            if idx % 50 == 0 or idx == total_images:
                print(f"  [{idx}/{total_images}]  "
                      f"P={self.metrics.get_precision():.3f}  "
                      f"R={self.metrics.get_recall():.3f}  "
                      f"F1={self.metrics.get_f1():.3f}")

        report_dir = self.metrics.save_report()
        print(f"\nEvaluation complete → {report_dir}/")

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
    MODEL_PATH = "/home/frauas/ODT6/ObjectDetectionWithTransformersODT6/ODTSegformer/ObjectDetectionWithTransformersODT6/water_bottle_segformer_model_v2/huggingface_model_best"
    
    import os
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at: {MODEL_PATH}")
        print("Please update MODEL_PATH in the script")
        return
    
    # Dataset paths used by mode 3
    ANNOTATIONS_JSON = "/home/frauas/ODT6/ObjectDetectionWithTransformersODT6/my_dataset/All-1.json"
    IMAGES_DIR       = "/home/frauas/ODT6/ObjectDetectionWithTransformersODT6/my_dataset/rgb"

    try:
        print("\nConfiguration:")
        print(f"  Model: {MODEL_PATH}")
        print(f"  Confidence Threshold: 0.5")
        print(f"  Min Area: 500 pixels")

        print("\nSelect mode:")
        print("1. Single frame  (camera)")
        print("2. Continuous detection  (camera)")
        print("3. Dataset evaluation  (Precision / Recall / F1  –  no camera)")

        choice = input("Enter choice (1, 2 or 3): ").strip()

        if choice == "3":
            detector = SegFormerBottleDetector(
                model_path=MODEL_PATH,
                confidence_threshold=0.5,
                min_area=500,
                no_camera=True,
            )
            detector.evaluate_dataset(
                annotations_json=ANNOTATIONS_JSON,
                images_dir=IMAGES_DIR,
                iou_threshold=0.5,
            )
        else:
            detector = SegFormerBottleDetector(
                model_path=MODEL_PATH,
                confidence_threshold=0.5,
                min_area=500,
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