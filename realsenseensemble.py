#!/usr/bin/env python3
"""
RealSense D435 + CASCADE DETR
Model 1: Bottle detector
Model 2: Hard-negative rejector
Final detection = M1 AND NOT(M2 overlap)
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import os


class CascadeBottleDetector:

    def __init__(self,
                 confidence_threshold_1=0.70,
                 confidence_threshold_2=0.85,
                 reject_iou_threshold=0.6):

        print("="*70)
        print("CASCADE DETR (Bottle + Hard Negative Rejector)")
        print("="*70)

        self.confidence_threshold_1 = confidence_threshold_1
        self.confidence_threshold_2 = confidence_threshold_2
        self.reject_iou_threshold = reject_iou_threshold

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # ------------------ LOAD MODEL 1 (Bottle) ------------------
        model1_path = "./water_bottle_model/huggingface_model"
        if not os.path.exists(model1_path):
            raise FileNotFoundError("Model 1 not found")

        self.processor_1 = DetrImageProcessor.from_pretrained(model1_path)
        self.model_1 = DetrForObjectDetection.from_pretrained(model1_path)
        self.model_1.to(self.device)
        self.model_1.eval()

        print("Model 1 labels:", self.model_1.config.id2label)

        # ------------------ LOAD MODEL 2 (Hard Negative) ------------------
        model2_path = "./water_bottle_model_hard_negatives/huggingface_model"
        if not os.path.exists(model2_path):
            raise FileNotFoundError("Model 2 not found")

        self.processor_2 = DetrImageProcessor.from_pretrained(model2_path)
        self.model_2 = DetrForObjectDetection.from_pretrained(model2_path)
        self.model_2.to(self.device)
        self.model_2.eval()

        print("Model 2 labels:", self.model_2.config.id2label)

        self.initialize_camera()

    # =========================================================
    # CAMERA
    # =========================================================

    def initialize_camera(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(config)
        print("RealSense started.")

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        return np.asanyarray(color_frame.get_data())

    # =========================================================
    # DETECTION
    # =========================================================

    def detect_with_model(self, image, model, processor, threshold):

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        inputs = processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)

            results = processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=threshold
            )[0]

        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            class_name = model.config.id2label[label.item()]
            if class_name in ["no-object", "background"]:
                continue

            detections.append({
                "confidence": score.item(),
                "bbox": box.cpu().numpy()
            })

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

    def detect_bottles(self, image):

        detections_1 = self.detect_with_model(
            image, self.model_1, self.processor_1, self.confidence_threshold_1
        )

        detections_2 = self.detect_with_model(
            image, self.model_2, self.processor_2, self.confidence_threshold_2
        )

        final = []

        for det1 in detections_1:
            reject = False
            for det2 in detections_2:
                iou = self.compute_iou(det1["bbox"], det2["bbox"])
                if iou > self.reject_iou_threshold:
                    reject = True
                    break

            if not reject:
                final.append(det1)

        return final

    # =========================================================
    # DRAWING
    # =========================================================

    def draw_detections(self, image, detections):

        h, w = image.shape[:2]

        for det in detections:
            xmin, ymin, xmax, ymax = det["bbox"]

            # Handle normalized coordinates if necessary
            if xmax <= 1.0 and ymax <= 1.0:
                xmin *= w
                xmax *= w
                ymin *= h
                ymax *= h

            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            cv2.putText(image,
                        f"{det['confidence']:.2f}",
                        (xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2)

        return image

    # =========================================================
    # MAIN LOOP
    # =========================================================

    def run(self):

        print("Running cascade detection. Press Q to quit.")

        while True:
            frame = self.get_frame()
            if frame is None:
                continue

            detections = self.detect_bottles(frame)
            annotated = self.draw_detections(frame.copy(), detections)

            cv2.imshow("Bottle Detection (Cascade DETR)", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.pipeline.stop()
        cv2.destroyAllWindows()


# =========================================================

def main():

    detector = CascadeBottleDetector(
        confidence_threshold_1=0.70,
        confidence_threshold_2=0.85,
        reject_iou_threshold=0.6
    )

    detector.run()


if __name__ == "__main__":
    main()