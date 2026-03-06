#!/usr/bin/env python3
"""
Ensemble Bottle Detector with Depth Filtering

M1 = Bottle detector     -> proposes bottle regions
M2 = No-bottle detector  -> rejects non-bottle regions ("no-object" class)

Filter pipeline per M1 detection:
  Step 1: Min box size   -> reject tiny distant blobs
  Step 2: Aspect ratio   -> bottles are tall/narrow, chairs are wide
  Step 3: Depth check    -> reject anything further than MAX_DEPTH_M
  Step 4: M2 ensemble    -> reject if M2 overlaps
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import time

# =========================================================
# CONFIG
# =========================================================
MODEL1_PATH    = "./water_bottle_model/huggingface_model"
MODEL2_PATH    = "./water_bottle_model_hard_negatives/huggingface_model"

M1_THRESHOLD   = 0.50
M2_THRESHOLD   = 0.30
REJECT_IOU     = 0.50
DETECT_EVERY_N = 3

# Shape filters
MIN_BOX_HEIGHT = 40      # px  - ignore boxes shorter than this
MIN_BOX_WIDTH  = 15      # px  - ignore boxes narrower than this
# NOTE: No aspect ratio filter - bottles can be tilted/sideways so aspect is unreliable

# Depth filter
MAX_DEPTH_M    = 2.0     # metres - reject detections beyond this distance

M1_SKIP = {"background"}
M2_SKIP = {"background"}   # keep "no-object" - that IS M2s real trained class
# =========================================================


def load_model(path, device):
    proc  = DetrImageProcessor.from_pretrained(path)
    model = DetrForObjectDetection.from_pretrained(path)
    model.to(device).eval()
    return proc, model


def run_model(image_bgr, proc, model, threshold, device, skip_labels):
    rgb     = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    inputs  = proc(images=pil_img, return_tensors="pt")
    inputs  = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        sizes   = torch.tensor([pil_img.size[::-1]]).to(device)
        results = proc.post_process_object_detection(
                      outputs, target_sizes=sizes, threshold=threshold)[0]
    dets = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        name = model.config.id2label.get(label.item(), "unknown")
        if name in skip_labels:
            continue
        dets.append({"label": name, "confidence": float(score),
                     "bbox": box.detach().cpu().numpy()})
    return dets


def get_median_depth(depth_frame, bbox, frame_h, frame_w):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_w, x2), min(frame_h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    cx, cy    = (x1 + x2) // 2, (y1 + y2) // 2
    pad       = 10
    depth_img = np.asanyarray(depth_frame.get_data())
    patch     = depth_img[max(0, cy-pad):min(frame_h, cy+pad),
                          max(0, cx-pad):min(frame_w, cx+pad)].astype(float)
    patch = patch[patch > 0]
    if patch.size == 0:
        return None
    return float(np.median(patch)) / 1000.0


def filter_m1(dets, depth_frame, frame_h, frame_w):
    passed, dropped = [], []
    for det in dets:
        x1, y1, x2, y2 = det["bbox"]
        bw, bh = x2 - x1, y2 - y1

        if bh < MIN_BOX_HEIGHT or bw < MIN_BOX_WIDTH:
            dropped.append((det, f"too small ({bw:.0f}x{bh:.0f}px)"))
            continue

        depth_m = get_median_depth(depth_frame, det["bbox"], frame_h, frame_w)
        det["depth_m"] = round(depth_m, 2) if depth_m is not None else None
        if depth_m is not None and depth_m > MAX_DEPTH_M:
            dropped.append((det, f"too far {depth_m:.2f}m"))
            continue

        passed.append(det)
    return passed, dropped


def iou(b1, b2):
    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    inter = (xi2 - xi1) * (yi2 - yi1)
    a1    = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2    = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return float(inter / union) if union > 0 else 0.0


def ensemble(m1, m2):
    kept, rejected = [], []
    for d1 in m1:
        best, best_d2 = 0.0, None
        for d2 in m2:
            v = iou(d1["bbox"], d2["bbox"])
            if v > best:
                best, best_d2 = v, d2
        if best >= REJECT_IOU:
            rejected.append((d1, best, best_d2))
        else:
            kept.append(d1)
    return kept, rejected


def draw(frame, kept, m2_rejected, shape_dropped, fps):
    for det in kept:
        x1, y1, x2, y2 = det["bbox"].astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"bottle {det['confidence']:.2f}"
        if det.get("depth_m"):
            label += f" {det['depth_m']}m"
        cv2.putText(frame, label, (x1, max(0, y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    for det, ov, _ in m2_rejected:
        x1, y1, x2, y2 = det["bbox"].astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"M2_rej iou={ov:.2f}", (x1, max(0, y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    for det, reason in shape_dropped:
        x1, y1, x2, y2 = det["bbox"].astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 1)
        cv2.putText(frame, reason, (x1, max(0, y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1)

    cv2.putText(frame, f"FPS:{fps:.1f}  Bottles:{len(kept)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return frame


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print("Loading M1 (bottle)...")
    proc1, model1 = load_model(MODEL1_PATH, device)
    print("  M1 labels:", model1.config.id2label)

    print("Loading M2 (no-bottle)...")
    proc2, model2 = load_model(MODEL2_PATH, device)
    print("  M2 labels:", model2.config.id2label)

    pipeline = rs.pipeline()
    cfg      = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
    pipeline.start(cfg)

    # Align depth to colour so pixel coordinates match
    align = rs.align(rs.stream.color)

    print(f"\nFilters active:")
    print(f"  Depth  : reject beyond {MAX_DEPTH_M}m")
    print(f"  MinSize: {MIN_BOX_WIDTH}x{MIN_BOX_HEIGHT}px  (aspect ratio filter OFF - bottles can be tilted)")
    print("Press Q to quit.\n")

    frame_count  = 0
    last_kept    = []
    last_m2_rej  = []
    last_dropped = []
    fps_start    = time.time()

    while True:
        frames  = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_f = aligned.get_color_frame()
        depth_f = aligned.get_depth_frame()
        if not color_f or not depth_f:
            continue

        frame = np.asanyarray(color_f.get_data())
        h, w  = frame.shape[:2]

        if frame_count % DETECT_EVERY_N == 0:
            m1_raw  = run_model(frame, proc1, model1, M1_THRESHOLD, device, M1_SKIP)
            m2_dets = run_model(frame, proc2, model2, M2_THRESHOLD, device, M2_SKIP)

            m1_filtered, last_dropped = filter_m1(m1_raw, depth_f, h, w)
            last_kept, last_m2_rej    = ensemble(m1_filtered, m2_dets)

            print(f"\n[Frame {frame_count}]")
            print(f"  M1 raw        : {len(m1_raw)}")
            print(f"  After filters : {len(m1_filtered)} passed  {len(last_dropped)} dropped")
            for det, reason in last_dropped:
                print(f"    DROPPED  conf={det['confidence']:.3f}  -> {reason}")
            print(f"  M2 dets       : {len(m2_dets)}  {[(d['label'], round(d['confidence'],3)) for d in m2_dets]}")
            for det, ov, _ in last_m2_rej:
                print(f"    M2_REJECTED  conf={det['confidence']:.3f}  iou={ov:.3f}")
            for det in last_kept:
                print(f"    KEPT  conf={det['confidence']:.3f}  depth={det.get('depth_m')}m")
            print(f"  FINAL bottles : {len(last_kept)}")

        elapsed = time.time() - fps_start
        fps     = frame_count / elapsed if elapsed > 0 else 0.0
        out     = draw(frame.copy(), last_kept, last_m2_rej, last_dropped, fps)
        cv2.imshow("Ensemble Bottle Detector", out)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_count += 1

    pipeline.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()