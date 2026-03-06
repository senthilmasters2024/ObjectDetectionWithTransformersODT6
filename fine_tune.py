#!/usr/bin/env python3
"""
DETR Single-Class Training
Model 1: Pure Bottle Detector
No explicit 'no-object' class
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import json
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path

# ==========================================================
# CONFIGURATION
# ==========================================================

IMAGES_DIR = "./my_dataset/rgb"
ANNOTATIONS_FILE = "./my_dataset/All-1.json"
SAVE_DIR = "./water_bottle_model"

EPOCHS = 50
BATCH_SIZE = 2
LEARNING_RATE = 1e-5
SAVE_EVERY = 10

# ==========================================================


class BottleDataset(Dataset):
    def __init__(self, images_dir, annotations_file, processor):
        self.images_dir = images_dir
        self.processor = processor

        print("\n📂 Loading annotations...")

        with open(annotations_file, "r") as f:
            coco = json.load(f)

        # Get actual image files
        actual_files = set()
        for ext in ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"]:
            actual_files.update([p.name for p in Path(images_dir).glob(ext)])

        # Map image id → info
        self.img_id_to_info = {
            img["id"]: img
            for img in coco["images"]
            if img["file_name"] in actual_files
        }

        # Map image id → annotations
        self.img_to_anns = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            if img_id in self.img_id_to_info:
                self.img_to_anns.setdefault(img_id, []).append(ann)

        self.image_ids = list(self.img_to_anns.keys())

        if len(self.image_ids) == 0:
            raise ValueError("❌ No valid training images found!")

        print(f"✅ Using {len(self.image_ids)} images")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img_id = self.image_ids[idx]
        img_info = self.img_id_to_info[img_id]
        img_path = os.path.join(self.images_dir, img_info["file_name"])

        image = Image.open(img_path).convert("RGB")
        anns = self.img_to_anns[img_id]

        img_w, img_h = image.size

        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann["bbox"]

            # Convert to normalized center format (DETR expects this)
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            norm_w = w / img_w
            norm_h = h / img_h

            # Skip invalid boxes
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and norm_w > 0 and norm_h > 0):
                continue

            boxes.append([x_center, y_center, norm_w, norm_h])
            labels.append(0)  # Only one class: bottle

        if len(boxes) == 0:
            # Skip problematic image
            return self.__getitem__((idx + 1) % len(self.image_ids))

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        encoding = self.processor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()

        target = {
            "boxes": boxes,
            "class_labels": labels
        }

        return pixel_values, target


def collate_fn(batch):
    pixel_values = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return pixel_values, targets


# ==========================================================
# TRAINING
# ==========================================================

def main():

    print("="*60)
    print("DETR SINGLE-CLASS TRAINING (BOTTLE ONLY)")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    # TRUE SINGLE CLASS
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=1,
        ignore_mismatched_sizes=True
    )

    model.config.id2label = {0: "bottle"}
    model.config.label2id = {"bottle": 0}

    model.to(device)

    print("Model configured for 1 class:")
    print(model.config.id2label)

    dataset = BottleDataset(IMAGES_DIR, ANNOTATIONS_FILE, processor)

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    os.makedirs(SAVE_DIR, exist_ok=True)

    best_loss = float("inf")

    print("\n🚀 Starting training...\n")

    for epoch in range(EPOCHS):

        model.train()
        losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for pixel_values, targets in pbar:

            pixel_values = pixel_values.to(device)

            batch_targets = []
            for t in targets:
                batch_targets.append({
                    "boxes": t["boxes"].to(device),
                    "class_labels": t["class_labels"].to(device)
                })

            outputs = model(pixel_values=pixel_values, labels=batch_targets)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            losses.append(loss.item())
            pbar.set_postfix(loss=loss.item())

        avg_loss = np.mean(losses)
        print(f"\nEpoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pt"))
            print("💾 Best model saved")

        if (epoch + 1) % SAVE_EVERY == 0:
            torch.save(model.state_dict(),
                       os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch+1}.pt"))

    print("\nSaving final HuggingFace model...")

    hf_save_dir = os.path.join(SAVE_DIR, "huggingface_model")
    model.save_pretrained(hf_save_dir)
    processor.save_pretrained(hf_save_dir)

    print("✅ Training complete")
    print("Best loss:", best_loss)
    print("="*60)


if __name__ == "__main__":
    main()