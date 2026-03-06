#!/usr/bin/env python3
"""
SegFormer M2 - Hard Negative (No-Object) Segmentation Training
Dataset: All-3_clean_bbox_only.json (bbox only, no-object class)
Mirrors the DETR M2 (trainhardnegatives.py) approach using SegFormer.

Since no polygon segmentation is available, bounding boxes are converted
to rectangular filled masks for training the segmentation model.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import json
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIGURATION
# ============================================================

IMAGES_DIR       = "./my_dataset/TrainingDatasetNoBottle/rgb"
ANNOTATIONS_FILE = "./my_dataset/All-3_clean_bbox_only.json"
SAVE_DIR         = "./segformer_m2_model"

EPOCHS        = 30
BATCH_SIZE    = 4
LEARNING_RATE = 6e-5
SAVE_EVERY    = 5
IMAGE_SIZE    = (512, 512)   # must be multiple of 32
TRAIN_SPLIT   = 0.85
RANDOM_SEED   = 42

# Label mapping: 0 = background, 1 = no-object (hard negative region)
ID2LABEL = {0: "background", 1: "no-object"}
LABEL2ID = {"background": 0, "no-object": 1}

# ============================================================


class HardNegativeSegDataset(Dataset):
    """
    Hard-negative segmentation dataset.
    No polygon annotations available -> rectangular masks are created from bboxes.
    """

    def __init__(self, images_dir, annotations_file, processor,
                 image_size=(512, 512), split="train",
                 train_ids=None, val_ids=None):
        self.images_dir = images_dir
        self.processor  = processor
        self.image_size = image_size
        self.split      = split

        print(f"\nLoading annotations for [{split}]...")
        with open(annotations_file) as f:
            coco = json.load(f)

        # Build case-insensitive filename lookup
        actual_files = {}
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            for p in Path(images_dir).glob(f"*{ext}"):
                actual_files[p.name.lower()] = p.name

        print(f"  Found {len(actual_files)} image files on disk")

        # Map image_id -> info
        self.img_id_to_info = {}
        for img in coco["images"]:
            key = img["file_name"].lower()
            if key in actual_files:
                img["file_name"] = actual_files[key]
                self.img_id_to_info[img["id"]] = img

        print(f"  Matched {len(self.img_id_to_info)} images from JSON")

        # Group bbox annotations by image
        self.img_to_anns = {}
        for ann in coco["annotations"]:
            if ann["image_id"] not in self.img_id_to_info:
                continue
            bbox = ann.get("bbox", [])
            if len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
                self.img_to_anns.setdefault(ann["image_id"], []).append(ann)

        all_ids = list(self.img_to_anns.keys())

        if train_ids is not None and val_ids is not None:
            self.image_ids = train_ids if split == "train" else val_ids
        else:
            if len(all_ids) >= 2:
                train_ids, val_ids = train_test_split(
                    all_ids, train_size=TRAIN_SPLIT, random_state=RANDOM_SEED
                )
            else:
                train_ids, val_ids = all_ids, all_ids
            self.image_ids = train_ids if split == "train" else val_ids

        if len(self.image_ids) == 0:
            raise ValueError(f"No images found for split '{split}'")

        print(f"  Using {len(self.image_ids)} images for [{split}]")

    def _bbox_to_mask(self, bboxes, orig_w, orig_h):
        """COCO bboxes (x,y,w,h) -> binary numpy mask (H x W)."""
        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for bbox in bboxes:
            x, y, w, h = bbox
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(orig_w, int(x + w))
            y2 = min(orig_h, int(y + h))
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 1
        return mask

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id   = self.image_ids[idx]
        img_info = self.img_id_to_info[img_id]
        img_path = os.path.join(self.images_dir, img_info["file_name"])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  Warning: could not open {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.image_ids))

        orig_w, orig_h = image.size
        anns  = self.img_to_anns[img_id]
        bboxes = [ann["bbox"] for ann in anns]

        mask = self._bbox_to_mask(bboxes, orig_w, orig_h)

        if mask.max() == 0:
            return self.__getitem__((idx + 1) % len(self.image_ids))

        # Resize
        image_r = image.resize(self.image_size, Image.BILINEAR)
        mask_r  = Image.fromarray(mask).resize(self.image_size, Image.NEAREST)
        mask_r  = np.array(mask_r, dtype=np.uint8)

        enc = self.processor(image_r, return_tensors="pt")
        pixel_values = enc["pixel_values"].squeeze(0)
        labels       = torch.from_numpy(mask_r).long()

        return {"pixel_values": pixel_values, "labels": labels}


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels":       torch.stack([b["labels"] for b in batch]),
    }


def compute_metrics(preds, targets, num_classes=2):
    preds   = preds.cpu().numpy().flatten()
    targets = targets.cpu().numpy().flatten()
    acc     = (preds == targets).mean()
    ious    = []
    for c in range(num_classes):
        p = preds == c
        t = targets == c
        inter = (p & t).sum()
        union = (p | t).sum()
        ious.append(inter / union if union > 0 else float("nan"))
    valid    = [v for v in ious if not np.isnan(v)]
    mean_iou = np.mean(valid) if valid else 0.0
    return acc, mean_iou, ious


def validate(model, loader, device):
    model.eval()
    losses, accs, ious = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Validating", leave=False):
            pv = batch["pixel_values"].to(device)
            lb = batch["labels"].to(device)
            out = model(pixel_values=pv, labels=lb)
            logits = torch.nn.functional.interpolate(
                out.logits, size=lb.shape[-2:], mode="bilinear", align_corners=False
            )
            preds = logits.argmax(dim=1)
            acc, miou, _ = compute_metrics(preds, lb)
            losses.append(out.loss.item())
            accs.append(acc)
            ious.append(miou)
    return {
        "loss": np.mean(losses),
        "accuracy": np.mean(accs),
        "mean_iou": np.mean(ious),
    }


def main():
    print("=" * 60)
    print("SEGFORMER M2 - HARD NEGATIVE (NO-OBJECT) SEGMENTATION")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
    print(f"\nLoading pretrained model: {model_name}")
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    model.to(device)
    print(f"  Num labels: {model.config.num_labels}")
    print(f"  Labels: {model.config.id2label}")

    # Build train/val split
    print("\nCreating train/val split from All-3_clean_bbox_only.json...")
    with open(ANNOTATIONS_FILE) as f:
        coco = json.load(f)

    actual_files = {}
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        for p in Path(IMAGES_DIR).glob(f"*{ext}"):
            actual_files[p.name.lower()] = p.name

    img_id_to_info = {
        img["id"]: img
        for img in coco["images"]
        if img["file_name"].lower() in actual_files
    }

    img_to_anns = {}
    for ann in coco["annotations"]:
        if ann["image_id"] not in img_id_to_info:
            continue
        bbox = ann.get("bbox", [])
        if len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
            img_to_anns.setdefault(ann["image_id"], []).append(ann)

    all_ids = list(img_to_anns.keys())
    if len(all_ids) >= 2:
        train_ids, val_ids = train_test_split(
            all_ids, train_size=TRAIN_SPLIT, random_state=RANDOM_SEED
        )
    else:
        train_ids, val_ids = all_ids, all_ids

    print(f"  Total: {len(all_ids)} | Train: {len(train_ids)} | Val: {len(val_ids)}")

    train_ds = HardNegativeSegDataset(
        IMAGES_DIR, ANNOTATIONS_FILE, processor,
        image_size=IMAGE_SIZE, split="train",
        train_ids=train_ids, val_ids=val_ids,
    )
    val_ds = HardNegativeSegDataset(
        IMAGES_DIR, ANNOTATIONS_FILE, processor,
        image_size=IMAGE_SIZE, split="val",
        train_ids=train_ids, val_ids=val_ids,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE * 0.1
    )

    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"\nStarting M2 training: {EPOCHS} epochs")
    print(f"  Train batches/epoch: {len(train_loader)}")

    best_iou = 0.0
    history  = {"train_loss": [], "train_iou": [], "val_loss": [], "val_iou": []}

    for epoch in range(EPOCHS):
        model.train()
        t_losses, t_ious = [], []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in pbar:
            pv = batch["pixel_values"].to(device)
            lb = batch["labels"].to(device)

            out  = model(pixel_values=pv, labels=lb)
            loss = out.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                logits = torch.nn.functional.interpolate(
                    out.logits, size=lb.shape[-2:], mode="bilinear", align_corners=False
                )
                preds = logits.argmax(dim=1)
                _, miou, _ = compute_metrics(preds, lb)

            t_losses.append(loss.item())
            t_ious.append(miou)
            pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{miou:.4f}")

        scheduler.step()

        val_metrics = validate(model, val_loader, device)
        avg_t_loss  = np.mean(t_losses)
        avg_t_iou   = np.mean(t_ious)

        history["train_loss"].append(avg_t_loss)
        history["train_iou"].append(avg_t_iou)
        history["val_loss"].append(val_metrics["loss"])
        history["val_iou"].append(val_metrics["mean_iou"])

        print(f"\nEpoch {epoch+1}/{EPOCHS}:")
        print(f"  Train  Loss={avg_t_loss:.4f}  IoU={avg_t_iou:.4f}")
        print(f"  Val    Loss={val_metrics['loss']:.4f}  IoU={val_metrics['mean_iou']:.4f}  Acc={val_metrics['accuracy']:.4f}")
        print(f"  LR={scheduler.get_last_lr()[0]:.6f}")

        if val_metrics["mean_iou"] > best_iou:
            best_iou = val_metrics["mean_iou"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_iou": best_iou,
                    "history": history,
                },
                os.path.join(SAVE_DIR, "best_model.pt"),
            )
            print(f"  ** New best model  Val IoU={best_iou:.4f} **")

        if (epoch + 1) % SAVE_EVERY == 0:
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "history": history},
                os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch+1}.pt"),
            )
            print(f"  Checkpoint saved")

    # Save final HuggingFace model
    hf_dir = os.path.join(SAVE_DIR, "huggingface_model")
    model.save_pretrained(hf_dir)
    processor.save_pretrained(hf_dir)
    print(f"\nM2 model saved to: {hf_dir}")

    import pickle
    with open(os.path.join(SAVE_DIR, "history.pkl"), "wb") as f:
        pickle.dump(history, f)

    print(f"\nTraining complete.")
    print(f"  Best Val IoU : {best_iou:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
