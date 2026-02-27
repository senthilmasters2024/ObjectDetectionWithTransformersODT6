#!/usr/bin/env python3
"""
RealSense D435 + DUAL DETR Models - ENSEMBLE DETECTION
Uses BOTH positive-trained and hard-negative-trained models for better accuracy

UPDATED:
- Added shape-based post-filter to reduce false positives (boxes/chairs)
- Made detection faster (inference every 2 frames instead of 3)
- Added temporal hold (stickiness) so detection doesn't flicker when bottle rotates
- Relaxed filter thresholds so rotated bottle still gets detected
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import time
import os


class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.inference_times = []
        self.frame_count = 0
        self.total_detections = 0
        self.detections_per_class = {}
        self.model_agreement_count = 0
        self.model1_only = 0
        self.model2_only = 0

    def update_inference(self, inference_time, detections):
        self.inference_times.append(inference_time)
        self.frame_count += 1
        self.total_detections += len(detections)
        for det in detections:
            label = det['label']
            self.detections_per_class[label] = self.detections_per_class.get(label, 0) + 1

    def get_fps(self):
        if not self.inference_times:
            return 0.0
        recent = self.inference_times[-30:]
        avg_time = sum(recent) / len(recent)
        return 1.0 / avg_time if avg_time > 0 else 0.0

    def get_avg_inference_time(self):
        if not self.inference_times:
            return 0.0
        return (sum(self.inference_times) / len(self.inference_times)) * 1000

    def get_summary(self):
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
                 confidence_threshold_1=0.95,
                 confidence_threshold_2=0.95,
                 ensemble_strategy="weighted"):
        print("="*70)
        print("INITIALIZING DUAL-MODEL DETR DETECTION")
        print("="*70)

        self.confidence_threshold_1 = confidence_threshold_1
        self.confidence_threshold_2 = confidence_threshold_2
        self.ensemble_strategy = ensemble_strategy

        # ---- Shape filter knobs (relaxed for rotation) ----
        self.bottle_min_ar = 1.45     # allow wider view angles
        self.bottle_max_ar = 6.0
        self.bottle_min_h = 55        # allow smaller detections
        self.bottle_shape_thr = 0.18  # less strict, better recall

        # ---- Temporal hold (stickiness) ----
        self.last_good = []
        self.last_good_time = 0.0
        self.hold_seconds = 0.7

        print(f"\nEnsemble Strategy : {ensemble_strategy}")
        print(f"M1 Threshold      : {confidence_threshold_1}")
        print(f"M2 Threshold      : {confidence_threshold_2}")
        print(f"Shape Filter      : min_ar={self.bottle_min_ar}, min_h={self.bottle_min_h}, score_thr={self.bottle_shape_thr}")
        print(f"Hold Seconds      : {self.hold_seconds}")

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            try:
                test_tensor = torch.zeros(1).cuda()
                _ = test_tensor + 1
                del test_tensor
                torch.cuda.empty_cache()
                self.device = torch.device("cuda")
                print(f"\nUsing device: cuda ({torch.cuda.get_device_name(0)})")
            except RuntimeError as e:
                print(f"\nCUDA error: {e}, falling back to CPU")
        else:
            print("\nUsing device: cpu")

        model1_path = "/home/frauas/ODT6/ObjectDetectionWithTransformersODT6/water_bottle_model/huggingface_model"
        if not os.path.exists(model1_path):
            model1_path = "./water_bottle_model/huggingface_model"
        if not os.path.exists(model1_path):
            raise FileNotFoundError(f"Model 1 not found: {model1_path}")

        print(f"\nLoading M1 from: {model1_path}")
        self.processor_1 = DetrImageProcessor.from_pretrained(model1_path)
        self.model_1 = DetrForObjectDetection.from_pretrained(model1_path)
        self.model_1.to(self.device)
        self.model_1.eval()
        print(f"M1 loaded | labels: {self.model_1.config.id2label}")

        model2_path = "/home/frauas/ODT6/ObjectDetectionWithTransformersODT6/water_bottle_model_hard_negatives/huggingface_model"
        if not os.path.exists(model2_path):
            model2_path = "./water_bottle_model_hard_negatives/huggingface_model"
        if not os.path.exists(model2_path):
            raise FileNotFoundError(f"Model 2 not found: {model2_path}")

        print(f"\nLoading M2 from: {model2_path}")
        self.processor_2 = DetrImageProcessor.from_pretrained(model2_path)
        self.model_2 = DetrForObjectDetection.from_pretrained(model2_path)
        self.model_2.to(self.device)
        self.model_2.eval()
        print(f"M2 loaded | labels: {self.model_2.config.id2label}")

        self.pipeline = None
        self._cleaned_up = False
        self.initialize_camera()
        self.metrics = MetricsTracker()

    def initialize_camera(self):
        print("\n" + "="*70)
        print("INITIALIZING REALSENSE D435")
        print("="*70)
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError("No RealSense devices found!")
        dev = devices[0]
        print(f"Device: {dev.get_info(rs.camera_info.name)}")

        self.pipeline = rs.pipeline()
        camera_started = False
        for width, height, fps in [(640, 480, 30), (848, 480, 30)]:
            try:
                config = rs.config()
                config.enable_device(dev.get_info(rs.camera_info.serial_number))
                config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
                config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
                self.pipeline.start(config)
                print(f"Camera started at {width}x{height}")
                camera_started = True
                break
            except Exception as e:
                print(f"Failed {width}x{height}: {e}")

        if not camera_started:
            raise RuntimeError("Failed to start camera")

        print("Warming up...")
        for _ in range(30):
            try:
                self.pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError:
                continue
        print("RealSense ready!")

    def get_frame(self, timeout_ms=5000):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                return None, None
            return np.asanyarray(color_frame.get_data()), depth_frame
        except RuntimeError as e:
            print(f"Frame timeout: {e}")
            return None, None

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
                class_name = model.config.id2label.get(label.item(), "unknown")
                if class_name in ["no-object", "background"]:
                    continue
                detections.append({
                    "label": class_name,
                    "confidence": float(score.item()),
                    "bbox": box.detach().cpu().numpy()
                })

        del inputs, outputs, target_sizes, results
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return detections

    def compute_iou(self, box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)
        if xi_max < xi_min or yi_max < yi_min:
            return 0.0
        intersection = (xi_max - xi_min) * (yi_max - yi_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0

    def nms_detections(self, detections, iou_threshold=0.5):
        if len(detections) == 0:
            return []
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        keep = []
        while detections:
            best = detections[0]
            keep.append(best)
            detections = detections[1:]
            detections = [d for d in detections if self.compute_iou(best["bbox"], d["bbox"]) < iou_threshold]
        return keep

    def ensemble_detections(self, detections_1, detections_2):
        # Voting, union, intersection, weighted remain as in your original logic
        if self.ensemble_strategy == "union":
            return self.nms_detections(detections_1 + detections_2, iou_threshold=0.5)

        elif self.ensemble_strategy == "intersection":
            final_detections = []
            for det1 in detections_1:
                for det2 in detections_2:
                    if self.compute_iou(det1["bbox"], det2["bbox"]) > 0.5:
                        final_detections.append({
                            "label": det1["label"],
                            "confidence": (det1["confidence"] + det2["confidence"]) / 2,
                            "bbox": (det1["bbox"] + det2["bbox"]) / 2,
                            "source": "both"
                        })
                        self.metrics.model_agreement_count += 1
            return final_detections

        elif self.ensemble_strategy == "weighted":
            final_detections = []
            used_det2 = set()

            for det1 in detections_1:
                best_match = None
                best_iou = 0
                best_idx = -1

                for idx, det2 in enumerate(detections_2):
                    if idx in used_det2:
                        continue
                    iou = self.compute_iou(det1["bbox"], det2["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_match = det2
                        best_idx = idx

                if best_iou > 0.3 and best_match is not None:
                    final_detections.append({
                        "label": det1["label"],
                        "confidence": 0.4 * det1["confidence"] + 0.6 * best_match["confidence"],
                        "bbox": (det1["bbox"] + best_match["bbox"]) / 2,
                        "source": "both"
                    })
                    used_det2.add(best_idx)
                    self.metrics.model_agreement_count += 1
                else:
                    final_detections.append({
                        "label": det1["label"],
                        "confidence": det1["confidence"] * 0.8,
                        "bbox": det1["bbox"],
                        "source": "model1"
                    })
                    self.metrics.model1_only += 1

            for idx, det2 in enumerate(detections_2):
                if idx not in used_det2:
                    final_detections.append({
                        "label": det2["label"],
                        "confidence": det2["confidence"] * 0.9,
                        "bbox": det2["bbox"],
                        "source": "model2"
                    })
                    self.metrics.model2_only += 1

            return final_detections

        else:  # voting
            final_detections = []
            used_det2 = set()

            for det1 in detections_1:
                found_match = False
                for idx, det2 in enumerate(detections_2):
                    if idx in used_det2:
                        continue
                    if self.compute_iou(det1["bbox"], det2["bbox"]) > 0.4:
                        final_detections.append({
                            "label": det1["label"],
                            "confidence": max(det1["confidence"], det2["confidence"]),
                            "bbox": (det1["bbox"] + det2["bbox"]) / 2,
                            "source": "both"
                        })
                        used_det2.add(idx)
                        found_match = True
                        self.metrics.model_agreement_count += 1
                        break

                if not found_match and det1["confidence"] > 0.95:
                    final_detections.append({
                        "label": det1["label"],
                        "confidence": det1["confidence"],
                        "bbox": det1["bbox"],
                        "source": "model1"
                    })
                    self.metrics.model1_only += 1

            for idx, det2 in enumerate(detections_2):
                if idx not in used_det2 and det2["confidence"] > 0.98:
                    final_detections.append({
                        "label": det2["label"],
                        "confidence": det2["confidence"],
                        "bbox": det2["bbox"],
                        "source": "model2"
                    })
                    self.metrics.model2_only += 1

            return final_detections

    # -------- Shape filtering helpers --------
    def looks_like_bottle_bbox(self, bbox, min_ar, max_ar, min_h):
        xmin, ymin, xmax, ymax = bbox
        w = max(1.0, float(xmax - xmin))
        h = max(1.0, float(ymax - ymin))
        ar = h / w
        return (h >= min_h) and (min_ar <= ar <= max_ar)

    def bottle_shape_score(self, image_bgr, bbox):
        xmin, ymin, xmax, ymax = bbox.astype(int)
        ih, iw = image_bgr.shape[:2]
        xmin = max(0, xmin); ymin = max(0, ymin)
        xmax = min(iw - 1, xmax); ymax = min(ih - 1, ymax)
        if xmax <= xmin or ymax <= ymin:
            return 0.0

        roi = image_bgr[ymin:ymax, xmin:xmax]
        if roi.size == 0:
            return 0.0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(gray, 60, 150)
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return 0.0

        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area <= 0:
            return 0.0

        x, y, ww, hh = cv2.boundingRect(c)
        ar = hh / max(1.0, ww)
        fill = area / max(1.0, ww * hh)

        score = (min(ar, 6.0) / 6.0) * (1.0 - abs(fill - 0.35))
        return float(score)
    # ----------------------------------------

    def detect_objects(self, image):
        detections_1 = self.detect_with_model(image, self.model_1, self.processor_1, self.confidence_threshold_1)
        detections_2 = self.detect_with_model(image, self.model_2, self.processor_2, self.confidence_threshold_2)

        m1_summary = [(d["label"], round(d["confidence"], 3)) for d in detections_1]
        m2_summary = [(d["label"], round(d["confidence"], 3)) for d in detections_2]
        print(f"[M1 thr={self.confidence_threshold_1}] {m1_summary}")
        print(f"[M2 thr={self.confidence_threshold_2}] {m2_summary}")

        final_detections = self.ensemble_detections(detections_1, detections_2)

        # Filter: keep only bottle-shaped detections
        filtered = []
        for det in final_detections:
            if det.get("label") != "bottle":
                continue

            bbox = det.get("bbox")
            if bbox is None:
                continue

            if not self.looks_like_bottle_bbox(bbox, self.bottle_min_ar, self.bottle_max_ar, self.bottle_min_h):
                continue

            score = self.bottle_shape_score(image, bbox)
            if score < self.bottle_shape_thr:
                continue

            filtered.append(det)

        # ---- Temporal hold ----
        now = time.time()
        if len(filtered) > 0:
            self.last_good = filtered
            self.last_good_time = now
            return filtered

        if (now - self.last_good_time) < self.hold_seconds:
            return self.last_good

        return []

    def get_depth_at_bbox(self, depth_frame, bbox):
        xmin, ymin, xmax, ymax = bbox.astype(int)
        xmin = max(0, xmin); ymin = max(0, ymin)
        xmax = min(depth_frame.get_width(), xmax)
        ymax = min(depth_frame.get_height(), ymax)
        cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2
        depths = []
        for dx in range(-10, 10):
            for dy in range(-10, 10):
                x, y = cx + dx, cy + dy
                if 0 <= x < depth_frame.get_width() and 0 <= y < depth_frame.get_height():
                    d = depth_frame.get_distance(x, y)
                    if d > 0:
                        depths.append(d)
        return float(np.median(depths)) if depths else 0.0

    def draw_detections(self, image, detections, depth_frame=None):
        annotated = image.copy()
        for det in detections:
            label = det["label"]
            confidence = det["confidence"]
            bbox = det["bbox"]
            source = det.get("source", "unknown")

            xmin, ymin, xmax, ymax = bbox.astype(int)

            depth_text = ""
            if depth_frame is not None:
                depth = self.get_depth_at_bbox(depth_frame, bbox)
                if depth > 0:
                    depth_text = f" | {depth:.2f}m"

            if source == "both":
                color = (0, 255, 0); source_text = "both"
            elif source == "model1":
                color = (0, 255, 255); source_text = "M1"
            elif source == "model2":
                color = (255, 0, 255); source_text = "M2"
            else:
                color = (255, 255, 255); source_text = "?"

            cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), color, 2)
            text = f"{label}: {confidence:.2f} [{source_text}]{depth_text}"

            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (xmin, ymin - th - 10), (xmin + tw, ymin), color, -1)
            cv2.putText(annotated, text, (xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return annotated

    def draw_metrics_overlay(self, image):
        h, w = image.shape[:2]
        overlay = image.copy()
        cv2.rectangle(overlay, (w - 300, 0), (w, 185), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        items = [
            (f"FPS: {self.metrics.get_fps():.1f}", (0, 255, 0)),
            (f"Inference: {self.metrics.get_avg_inference_time():.1f}ms", (0, 255, 255)),
            (f"Frames: {self.metrics.frame_count}", (255, 255, 255)),
            (f"Detections: {self.metrics.total_detections}", (255, 255, 255)),
            (f"Strategy: {self.ensemble_strategy}", (200, 200, 200)),
            (f"M1 thr: {self.confidence_threshold_1}", (100, 255, 100)),
            (f"M2 thr: {self.confidence_threshold_2}", (100, 200, 255)),
            (f"Filter ar>={self.bottle_min_ar}", (200, 200, 200)),
            (f"Filter score>={self.bottle_shape_thr}", (200, 200, 200)),
            (f"Hold: {self.hold_seconds}s", (200, 200, 200)),
        ]

        for i, (text, color) in enumerate(items):
            cv2.putText(image, text, (w - 290, 22 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return image

    def run_continuous(self):
        print("\n" + "="*70)
        print("RUNNING DUAL-MODEL CONTINUOUS DETECTION")
        print("="*70)
        print("Controls: q=quit  s=screenshot  m=metrics")
        print("          v=voting  i=intersection  u=union  w=weighted")
        print()

        self.metrics.reset()
        frame_count = 0
        last_detections = []

        try:
            while True:
                color_image, depth_frame = self.get_frame(timeout_ms=5000)
                if color_image is None:
                    continue

                # Faster updates (every 2 frames instead of 3)
                if frame_count % 2 == 0:
                    start_time = time.time()
                    detections = self.detect_objects(color_image)
                    inference_time = time.time() - start_time
                    last_detections = detections
                    self.metrics.update_inference(inference_time, detections)

                annotated = self.draw_detections(color_image, last_detections, depth_frame)
                cv2.putText(
                    annotated,
                    f"DUAL-DETR ({self.ensemble_strategy}) | {len(last_detections)} objects | {self.metrics.get_fps():.1f} FPS",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2
                )

                annotated = self.draw_metrics_overlay(annotated)
                cv2.putText(
                    annotated,
                    "Green=Both | Yellow=M1 | Magenta=M2",
                    (10, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )
                cv2.imshow("RealSense + Dual DETR", annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    fname = f"dual_detection_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(fname, annotated)
                    print(f"Screenshot saved: {fname}")
                elif key == ord("m"):
                    self.metrics.print_summary()
                elif key == ord("v"):
                    self.ensemble_strategy = "voting"; print("→ voting")
                elif key == ord("i"):
                    self.ensemble_strategy = "intersection"; print("→ intersection")
                elif key == ord("u"):
                    self.ensemble_strategy = "union"; print("→ union")
                elif key == ord("w"):
                    self.ensemble_strategy = "weighted"; print("→ weighted")

                frame_count += 1

        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            self.metrics.print_summary()
            self.cleanup()

    def cleanup(self):
        if self._cleaned_up:
            return
        self._cleaned_up = True
        if self.pipeline:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass
        cv2.destroyAllWindows()


def main():
    print("="*70)
    print("RealSense D435 + DUAL DETR MODELS")
    print("="*70)
    print("\nSelect ensemble strategy:")
    print("  1. Voting       - both models agree (HIGH PRECISION)")
    print("  2. Intersection - very conservative")
    print("  3. Union        - combine all (HIGH RECALL)")
    print("  4. Weighted     - balanced (RECOMMENDED)")
    choice = input("Enter choice (1-4, default=4): ").strip() or "4"
    strategy = {"1": "voting", "2": "intersection", "3": "union", "4": "weighted"}.get(choice, "weighted")

    detector = DualModelRealsenseDETR(
        confidence_threshold_1=0.95,
        confidence_threshold_2=0.95,
        ensemble_strategy=strategy
    )
    detector.run_continuous()
    detector.cleanup()


if __name__ == "__main__":
    main()