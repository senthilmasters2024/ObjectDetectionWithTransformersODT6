#!/usr/bin/env python3
"""
Live RealSense D435 Detection using SegFormer M1 + Detection Head
-----------------------------------------------------------------
Loads the trained best_model.pt (RGBEncoder + SegFormerDetectionHead)
and runs continuous bottle detection on the RealSense D435 RGB stream.
Depth is used to show distance per detection.

Controls:
  q         - quit
  s         - save screenshot
  +/-       - raise/lower confidence threshold
"""

import sys
import os

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "ODTSegformer", "ObjectDetectionWithTransformersODT6")
)

import torch
import torch.nn as nn
import numpy as np
import cv2
import pyrealsense2 as rs
from PIL import Image
import time
from transformers import SegformerImageProcessor

from models.encoder_rgb import RGBEncoder
from models.detection_head import PositionEmbeddingSine, MLP

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH  = "./segformer_m1_detection_head/best_model.pt"

IMAGE_SIZE   = (512, 512)
FEATURE_DIMS = [32, 64, 160, 256]
HIDDEN_DIM   = 256
NUM_QUERIES  = 100
NUM_CLASSES  = 1          # bottle=0,  no-object=1

CONF_THRESH  = 0.3        # starting confidence threshold
INFER_EVERY  = 3          # run model every N frames (keeps display smooth)

SEGFORMER_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"

# ============================================================
# MODEL  (identical to TrainSegFormerM1_DetectionHead.py)
# ============================================================

class SegFormerDetectionHead(nn.Module):
    def __init__(self, feature_dims=FEATURE_DIMS, hidden_dim=HIDDEN_DIM,
                 num_queries=NUM_QUERIES, num_classes=NUM_CLASSES + 1,
                 num_decoder_layers=6):
        super().__init__()
        self.input_projections = nn.ModuleList([
            nn.Conv2d(dim, hidden_dim, kernel_size=1) for dim in feature_dims
        ])
        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2)
        self.query_embed        = nn.Embedding(num_queries, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=8,
            dim_feedforward=2048, dropout=0.1, batch_first=False,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed  = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

    def forward(self, features):
        B = features[0].shape[0]
        proj_list, pos_list = [], []
        for i, feat in enumerate(features):
            proj = self.input_projections[i](feat)
            pos  = self.position_embedding(proj)
            proj_list.append(proj.flatten(2).permute(2, 0, 1))
            pos_list.append(pos.flatten(2).permute(2, 0, 1))
        memory    = torch.cat(proj_list, dim=0)
        pos_embed = torch.cat(pos_list,  dim=0)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)
        tgt = torch.zeros_like(query_embed)
        hs  = self.transformer_decoder(tgt + query_embed, memory + pos_embed)
        hs  = hs.permute(1, 0, 2)
        return {
            "pred_logits": self.class_embed(hs),
            "pred_boxes":  self.bbox_embed(hs).sigmoid(),
        }


class SegFormerM1Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = RGBEncoder(model_name=SEGFORMER_MODEL)
        self.head    = SegFormerDetectionHead()

    def forward(self, pixel_values):
        return self.head(list(self.encoder(pixel_values)))


# ============================================================
# NMS
# ============================================================

def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(detections, iou_thresh=0.5):
    detections = sorted(detections, key=lambda d: d["conf"], reverse=True)
    keep = []
    for det in detections:
        if not any(_iou(det["box"], k["box"]) > iou_thresh for k in keep):
            keep.append(det)
    return keep


# ============================================================
# LIVE DETECTOR CLASS
# ============================================================

class SegFormerLiveDetector:

    def __init__(self, model_path=MODEL_PATH, conf_thresh=CONF_THRESH):
        self.conf_thresh = conf_thresh
        self.pipeline    = None
        self._cleaned_up = False

        # ---- device ----
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            try:
                torch.zeros(1).cuda()
                self.device = torch.device("cuda")
                print(f"GPU: {torch.cuda.get_device_name(0)}")
            except RuntimeError:
                print("CUDA not usable, falling back to CPU")
        print(f"Device: {self.device}")

        # ---- processor ----
        self.processor = SegformerImageProcessor.from_pretrained(SEGFORMER_MODEL)

        # ---- model ----
        print(f"\nLoading model: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = SegFormerM1Detector()
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        print(f"  Loaded from epoch {ckpt.get('epoch', 0)+1}  "
              f"val_loss={ckpt.get('val_loss', float('nan')):.4f}")

        # ---- camera ----
        self._init_camera()

        # ---- timing ----
        self.inference_times = []

    def _init_camera(self):
        print("\nInitializing RealSense D435...")
        ctx     = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError("No RealSense device found.")

        dev = devices[0]
        print(f"  Device  : {dev.get_info(rs.camera_info.name)}")
        print(f"  Serial  : {dev.get_info(rs.camera_info.serial_number)}")

        self.pipeline = rs.pipeline()
        started = False
        for w, h, fps in [(640, 480, 30), (848, 480, 30), (1280, 720, 15)]:
            try:
                cfg = rs.config()
                cfg.enable_device(dev.get_info(rs.camera_info.serial_number))
                cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
                cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
                self.pipeline.start(cfg)
                print(f"  Camera  : {w}x{h} @ {fps}fps")
                started = True
                break
            except Exception as e:
                print(f"  Failed {w}x{h}: {e}")

        if not started:
            raise RuntimeError("Could not start RealSense camera.")

        self.align = rs.align(rs.stream.color)

        # Warm-up
        print("  Warming up (30 frames)...")
        for _ in range(30):
            self.pipeline.wait_for_frames(timeout_ms=5000)
        print("  Camera ready.")

    def _get_frames(self):
        """Returns (bgr_image, depth_frame) or (None, None)."""
        try:
            frames        = self.pipeline.wait_for_frames(timeout_ms=5000)
            aligned       = self.align.process(frames)
            color_frame   = aligned.get_color_frame()
            depth_frame   = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                return None, None
            return np.asanyarray(color_frame.get_data()), depth_frame
        except Exception:
            return None, None

    def _infer(self, bgr):
        """Run model on one BGR frame. Returns list of detections."""
        h, w = bgr.shape[:2]
        pil  = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        pil_r = pil.resize(IMAGE_SIZE, Image.BILINEAR)

        enc = self.processor(pil_r, return_tensors="pt")
        pv  = enc["pixel_values"].to(self.device)

        t0 = time.time()
        with torch.no_grad():
            out = self.model(pv)
        elapsed = time.time() - t0
        self.inference_times.append(elapsed)

        logits = out["pred_logits"][0]   # (Q, C)
        boxes  = out["pred_boxes"][0]    # (Q, 4) norm cxcywh

        bottle_prob = logits.softmax(-1)[:, 0]

        detections = []
        for q in range(NUM_QUERIES):
            conf = bottle_prob[q].item()
            if conf < self.conf_thresh:
                continue
            cx, cy, bw, bh = boxes[q].cpu().tolist()
            x1 = max(0,  int((cx - bw / 2) * w))
            y1 = max(0,  int((cy - bh / 2) * h))
            x2 = min(w,  int((cx + bw / 2) * w))
            y2 = min(h,  int((cy + bh / 2) * h))
            if x2 > x1 and y2 > y1:
                detections.append({"conf": conf, "box": [x1, y1, x2, y2]})

        return nms(detections)

    def _depth_at(self, depth_frame, box):
        """Mean depth in metres at the centre third of the box."""
        x1, y1, x2, y2 = box
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        r  = max(1, min((x2 - x1), (y2 - y1)) // 6)
        samples = []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                d = depth_frame.get_distance(cx + dx, cy + dy)
                if d > 0:
                    samples.append(d)
        return float(np.median(samples)) if samples else 0.0

    def _draw(self, frame, detections, depth_frame, fps, thresh):
        annotated = frame.copy()
        GREEN = (0, 210, 0)
        DARK  = (0, 0, 0)

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            conf  = det["conf"]
            depth = self._depth_at(depth_frame, det["box"]) if depth_frame else 0.0

            cv2.rectangle(annotated, (x1, y1), (x2, y2), GREEN, 2)
            label = f"bottle {conf:.2f}"
            if depth > 0:
                label += f"  {depth:.2f}m"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), GREEN, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, DARK, 1)

        # HUD
        hud = [
            f"SegFormer M1  |  FPS: {fps:.1f}",
            f"Detections: {len(detections)}",
            f"Conf thresh: {thresh:.2f}  (+/- to adjust)",
            "Press Q to quit  |  S to save",
        ]
        for i, line in enumerate(hud):
            cv2.putText(annotated, line, (10, 28 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
        return annotated

    def run(self):
        print("\n" + "=" * 55)
        print("  LIVE DETECTION  |  SegFormer M1 + Detection Head")
        print("=" * 55)
        print(f"  Confidence threshold : {self.conf_thresh:.2f}")
        print(f"  Infer every          : {INFER_EVERY} frames")
        print("  Q = quit  |  S = screenshot  |  +/- = threshold")
        print("=" * 55 + "\n")

        frame_count   = 0
        last_dets     = []
        last_depth    = None
        fps           = 0.0
        fps_t         = time.time()
        fps_frames    = 0
        screenshot_n  = 0

        try:
            while True:
                bgr, depth_frame = self._get_frames()
                if bgr is None:
                    continue

                # Run inference every INFER_EVERY frames
                if frame_count % INFER_EVERY == 0:
                    last_dets  = self._infer(bgr)
                    last_depth = depth_frame

                # FPS
                fps_frames += 1
                now = time.time()
                if now - fps_t >= 1.0:
                    fps      = fps_frames / (now - fps_t)
                    fps_t    = now
                    fps_frames = 0

                annotated = self._draw(bgr, last_dets, last_depth, fps, self.conf_thresh)

                cv2.imshow("SegFormer M1 Live Detection", annotated)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                elif key == ord("s"):
                    screenshot_n += 1
                    fname = f"segformer_m1_live_{time.strftime('%Y%m%d_%H%M%S')}_{screenshot_n:03d}.jpg"
                    cv2.imwrite(fname, annotated)
                    print(f"  Screenshot saved: {fname}")
                elif key == ord("+") or key == ord("="):
                    self.conf_thresh = min(0.99, self.conf_thresh + 0.05)
                    print(f"  Conf threshold -> {self.conf_thresh:.2f}")
                elif key == ord("-"):
                    self.conf_thresh = max(0.05, self.conf_thresh - 0.05)
                    print(f"  Conf threshold -> {self.conf_thresh:.2f}")

                frame_count += 1

        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            self.cleanup()

        # Summary
        if self.inference_times:
            avg_ms = np.mean(self.inference_times) * 1000
            avg_fps = 1000 / avg_ms if avg_ms > 0 else 0
            print(f"\nSession summary:")
            print(f"  Frames processed   : {frame_count}")
            print(f"  Avg inference time : {avg_ms:.1f} ms")
            print(f"  Avg inference FPS  : {avg_fps:.1f}")

    def cleanup(self):
        if self._cleaned_up:
            return
        self._cleaned_up = True
        print("\nCleaning up...")
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        cv2.destroyAllWindows()
        print("Done.")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 55)
    print("  SegFormer M1 + Detection Head  |  Live RealSense")
    print("=" * 55)

    if not os.path.exists(MODEL_PATH):
        print(f"\nModel not found: {MODEL_PATH}")
        print("Run TrainSegFormerM1_DetectionHead.py first.")
        return

    detector = SegFormerLiveDetector(
        model_path=MODEL_PATH,
        conf_thresh=CONF_THRESH,
    )
    detector.run()


if __name__ == "__main__":
    main()
