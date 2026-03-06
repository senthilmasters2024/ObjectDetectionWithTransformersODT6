#!/usr/bin/env python3
"""
Standalone M2 No-Bottle Detector
Runs ONLY Model 2 to see what it detects and at what scores.
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
MODEL2_PATH = "./water_bottle_model_hard_negatives/huggingface_model"
CONFIDENCE_THRESHOLD = 0.10   # Start low, tune up based on what you see
DETECT_EVERY_N_FRAMES = 3
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

processor = DetrImageProcessor.from_pretrained(MODEL2_PATH)
model = DetrForObjectDetection.from_pretrained(MODEL2_PATH)
model.to(device).eval()
print("Model 2 labels:", model.config.id2label)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
print(f"\nRunning M2 only at threshold={CONFIDENCE_THRESHOLD}. Press Q to quit.\n")

frame_count = 0
fps_start = time.time()

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    frame = np.asanyarray(color_frame.get_data())

    if frame_count % DETECT_EVERY_N_FRAMES == 0:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        inputs = processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            target_sizes = torch.tensor([pil_image.size[::-1]]).to(device)

            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=CONFIDENCE_THRESHOLD
            )[0]

            raw_results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.0
            )[0]

        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            class_name = model.config.id2label.get(label.item(), "unknown")
            detections.append({
                "label": class_name,
                "confidence": float(score),
                "bbox": box.detach().cpu().numpy().astype(int)
            })

        raw_scores = sorted(
            [(float(s), model.config.id2label.get(l.item(), "unknown"))
             for s, l in zip(raw_results["scores"], raw_results["labels"])],
            reverse=True
        )[:5]

        print(f"[Frame {frame_count}] detections={len(detections)} | "
              f"top raw scores: {[(round(s,3), l) for s,l in raw_scores]}")

        for det in detections:
            xmin, ymin, xmax, ymax = det["bbox"]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
            cv2.putText(
                frame,
                f"{det['label']} {det['confidence']:.2f}",
                (xmin, max(0, ymin - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    elapsed = time.time() - fps_start
    fps = frame_count / elapsed if elapsed > 0 else 0.0
    cv2.putText(frame, f"FPS: {fps:.1f}  M2 only", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("M2 No-Bottle Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_count += 1

pipeline.stop()
cv2.destroyAllWindows()