#!/usr/bin/env python3
"""
SegFormer Bottle Segmentation - Training V2
Uses:
  - All-1.json:                 672 positive images with polygon segmentation
  - All-3_clean_bbox_only.json: 77  negative images (no bottles, all-background)

Improvements over V1:
  - Combines positive + negative examples
  - Weighted CrossEntropy loss for class imbalance
  - Data augmentation (flip, brightness/contrast, random crop)
  - Saves best model by bottle-class IoU (not mean IoU)
  - Correct model path for SegFormer.py inference
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image, ImageDraw, ImageEnhance
import json
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import random
from sklearn.model_selection import train_test_split

# ============================================================================
# CONFIGURATION
# ============================================================================
POSITIVE_IMAGES_DIR   = "../../my_dataset/rgb"
POSITIVE_ANNOTATIONS  = "../../my_dataset/All-1.json"

NEGATIVE_IMAGES_DIR   = "../../my_dataset/TrainingDatasetNoBottle/rgb"
NEGATIVE_ANNOTATIONS  = "../../my_dataset/All-3_clean_bbox_only.json"

SAVE_DIR   = "./water_bottle_segformer_model_v2"
MODEL_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"  # b2 = better accuracy than b0

EPOCHS       = 80
BATCH_SIZE   = 4
LR           = 6e-5
SAVE_EVERY   = 20
IMAGE_SIZE   = (512, 512)
TRAIN_SPLIT  = 0.85
RANDOM_SEED  = 42

# Augmentation probabilities
AUG_HFLIP_PROB      = 0.5
AUG_BRIGHTNESS_PROB = 0.4
AUG_BRIGHTNESS_RANGE = (0.7, 1.3)
AUG_CROP_PROB       = 0.4
AUG_CROP_SCALE      = (0.75, 1.0)   # crop to 75–100% of image
# ============================================================================


def polygon_to_mask(polygons, image_wh):
    """Convert COCO polygon list to a binary numpy mask."""
    mask = Image.new('L', image_wh, 0)
    draw = ImageDraw.Draw(mask)
    for poly in polygons:
        if len(poly) >= 6:
            pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
            draw.polygon(pts, outline=1, fill=1)
    return np.array(mask, dtype=np.uint8)


def augment(image: Image.Image, mask: np.ndarray):
    """Apply identical spatial transforms to image and mask."""
    w, h = image.size

    # Random horizontal flip
    if random.random() < AUG_HFLIP_PROB:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        mask  = mask[:, ::-1].copy()

    # Random brightness / contrast (image only)
    if random.random() < AUG_BRIGHTNESS_PROB:
        factor = random.uniform(*AUG_BRIGHTNESS_RANGE)
        image  = ImageEnhance.Brightness(image).enhance(factor)
        factor = random.uniform(*AUG_BRIGHTNESS_RANGE)
        image  = ImageEnhance.Contrast(image).enhance(factor)

    # Random crop (same for image + mask)
    if random.random() < AUG_CROP_PROB:
        scale   = random.uniform(*AUG_CROP_SCALE)
        new_w   = int(w * scale)
        new_h   = int(h * scale)
        x0      = random.randint(0, w - new_w)
        y0      = random.randint(0, h - new_h)
        image   = image.crop((x0, y0, x0 + new_w, y0 + new_h))
        mask    = mask[y0:y0 + new_h, x0:x0 + new_w]

    return image, mask


# ---------------------------------------------------------------------------
class BottleSegDataset(Dataset):
    """
    Combines positive (bottle) and negative (no-bottle) images.
    Each sample: {'pixel_values': Tensor(3,H,W), 'labels': Tensor(H,W)}
    Labels: 0 = background, 1 = bottle
    """

    def __init__(self, processor, samples, augment_data=False):
        """
        Args:
            processor:     SegformerImageProcessor
            samples:       list of dicts {'image_path', 'mask_polygons'|None, 'original_wh'}
                           mask_polygons = list of polygon coords, or None for negatives
            augment_data:  apply augmentation
        """
        self.processor     = processor
        self.samples       = samples
        self.augment_data  = augment_data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        try:
            image = Image.open(s['image_path']).convert('RGB')
        except Exception as e:
            print(f"Warning: cannot open {s['image_path']}: {e}")
            return self.__getitem__((idx + 1) % len(self.samples))

        orig_w, orig_h = image.size

        # Build mask
        if s['polygons'] is not None:
            mask = polygon_to_mask(s['polygons'], (orig_w, orig_h))
        else:
            mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        # Skip if positive image has empty mask (data issue)
        if s['polygons'] is not None and mask.max() == 0:
            return self.__getitem__((idx + 1) % len(self.samples))

        # Augmentation
        if self.augment_data:
            image, mask = augment(image, mask)

        # Resize to model input size
        image = image.resize(IMAGE_SIZE, Image.BILINEAR)
        mask  = Image.fromarray(mask).resize(IMAGE_SIZE, Image.NEAREST)
        mask  = np.array(mask, dtype=np.uint8)

        # Process image with HuggingFace processor
        enc = self.processor(image, return_tensors="pt")
        pixel_values = enc['pixel_values'].squeeze(0)

        return {
            'pixel_values': pixel_values,
            'labels':       torch.from_numpy(mask).long()
        }


def collate_fn(batch):
    pixel_values = torch.stack([b['pixel_values'] for b in batch])
    labels       = torch.stack([b['labels']       for b in batch])
    return {'pixel_values': pixel_values, 'labels': labels}


# ---------------------------------------------------------------------------
def load_samples(pos_images_dir, pos_ann_file, neg_images_dir, neg_ann_file):
    """
    Returns a list of sample dicts for all positive + negative images.
    """
    samples = []

    # ---- Positives (All-1.json) ----
    with open(pos_ann_file) as f:
        pos_data = json.load(f)

    # Build filename → actual filename mapping (case-insensitive)
    pos_dir    = Path(pos_images_dir)
    actual_map = {}
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        for fp in pos_dir.glob(f'*{ext}'):
            actual_map[fp.name.lower()] = fp.name

    img_info = {}
    for img in pos_data['images']:
        fname_lower = img['file_name'].lower()
        if fname_lower in actual_map:
            img['file_name'] = actual_map[fname_lower]
            img_info[img['id']] = img

    # Group polygons by image
    img_polygons = {}
    for ann in pos_data['annotations']:
        iid = ann['image_id']
        if iid not in img_info:
            continue
        seg = ann.get('segmentation', [])
        if not isinstance(seg, list) or len(seg) == 0:
            continue
        if iid not in img_polygons:
            img_polygons[iid] = []
        img_polygons[iid].extend(seg)

    for iid, polys in img_polygons.items():
        info = img_info[iid]
        samples.append({
            'image_path': str(pos_dir / info['file_name']),
            'polygons':   polys,
            'is_positive': True
        })

    print(f"Positive samples loaded: {len(samples)}")

    # ---- Negatives (All-3) ----
    with open(neg_ann_file) as f:
        neg_data = json.load(f)

    neg_dir    = Path(neg_images_dir)
    neg_actual = {}
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        for fp in neg_dir.glob(f'*{ext}'):
            neg_actual[fp.name.lower()] = fp.name

    neg_added = 0
    for img in neg_data['images']:
        fname_lower = img['file_name'].lower()
        if fname_lower in neg_actual:
            samples.append({
                'image_path':  str(neg_dir / neg_actual[fname_lower]),
                'polygons':    None,
                'is_positive': False
            })
            neg_added += 1

    print(f"Negative samples loaded: {neg_added}")
    print(f"Total samples: {len(samples)}")
    return samples


# ---------------------------------------------------------------------------
def compute_iou(pred, target, num_classes=2):
    """Returns per-class IoU list and mean IoU (ignoring NaN)."""
    pred   = pred.cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()
    ious   = []
    for cls in range(num_classes):
        p = (pred == cls)
        t = (target == cls)
        inter = (p & t).sum()
        union = (p | t).sum()
        ious.append(inter / union if union > 0 else float('nan'))
    valid = [v for v in ious if not np.isnan(v)]
    mean  = float(np.mean(valid)) if valid else 0.0
    return ious, mean


@torch.no_grad()
def validate(model, loader, device, criterion):
    model.eval()
    losses, bottle_ious, mean_ious = [], [], []

    for batch in tqdm(loader, desc="  Val", leave=False):
        pv  = batch['pixel_values'].to(device)
        lbl = batch['labels'].to(device)

        out    = model(pixel_values=pv)
        logits = torch.nn.functional.interpolate(
            out.logits, size=lbl.shape[-2:], mode='bilinear', align_corners=False
        )

        loss = criterion(logits, lbl)
        losses.append(loss.item())

        preds = logits.argmax(dim=1)
        for p, t in zip(preds, lbl):
            ious, mean_iou = compute_iou(p, t)
            mean_ious.append(mean_iou)
            bottle_ious.append(ious[1] if not np.isnan(ious[1]) else 0.0)

    return {
        'loss':       float(np.mean(losses)),
        'bottle_iou': float(np.mean(bottle_ious)),
        'mean_iou':   float(np.mean(mean_ious))
    }


# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("SEGFORMER BOTTLE SEGMENTATION  –  Training V2")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load all samples
    samples = load_samples(
        POSITIVE_IMAGES_DIR, POSITIVE_ANNOTATIONS,
        NEGATIVE_IMAGES_DIR, NEGATIVE_ANNOTATIONS
    )

    # Stratified train/val split (keep positive/negative ratio)
    labels = [1 if s['is_positive'] else 0 for s in samples]
    train_idx, val_idx = train_test_split(
        range(len(samples)),
        train_size=TRAIN_SPLIT,
        stratify=labels,
        random_state=RANDOM_SEED
    )
    train_samples = [samples[i] for i in train_idx]
    val_samples   = [samples[i] for i in val_idx]

    pos_train = sum(s['is_positive'] for s in train_samples)
    neg_train = len(train_samples) - pos_train
    pos_val   = sum(s['is_positive'] for s in val_samples)
    neg_val   = len(val_samples) - pos_val

    print(f"\nTrain: {len(train_samples)} total  ({pos_train} positive, {neg_train} negative)")
    print(f"Val:   {len(val_samples)}   total  ({pos_val} positive, {neg_val} negative)")

    # Load model + processor
    print(f"\nLoading {MODEL_NAME} ...")
    processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "background", 1: "bottle"},
        label2id={"background": 0, "bottle": 1},
        ignore_mismatched_sizes=True,
        use_safetensors=True
    )
    model.to(device)
    print("Model ready.")

    # Datasets & loaders
    train_ds = BottleSegDataset(processor, train_samples, augment_data=True)
    val_ds   = BottleSegDataset(processor, val_samples,   augment_data=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)

    # Class-weighted loss
    # Approximate pixel ratio: bottle ~15%, background ~85%
    # Weights: background=1.0, bottle=5.0  (tune if needed)
    class_weights = torch.tensor([1.0, 5.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)

    # Optimizer + LR scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.1
    )

    os.makedirs(SAVE_DIR, exist_ok=True)
    best_bottle_iou = 0.0
    history = {k: [] for k in ['train_loss', 'val_loss', 'val_bottle_iou', 'val_mean_iou']}

    print(f"\nStarting training for {EPOCHS} epochs ...\n")

    for epoch in range(EPOCHS):
        model.train()
        epoch_losses   = []
        epoch_biou     = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{EPOCHS}")
        for batch in pbar:
            pv  = batch['pixel_values'].to(device)
            lbl = batch['labels'].to(device)

            out    = model(pixel_values=pv)
            logits = torch.nn.functional.interpolate(
                out.logits, size=lbl.shape[-2:], mode='bilinear', align_corners=False
            )
            loss = criterion(logits, lbl)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                preds = logits.argmax(dim=1)
                batch_biou = []
                for p, t in zip(preds, lbl):
                    ious, _ = compute_iou(p, t)
                    batch_biou.append(ious[1] if not np.isnan(ious[1]) else 0.0)
                mean_biou = float(np.mean(batch_biou))

            epoch_losses.append(loss.item())
            epoch_biou.append(mean_biou)
            pbar.set_postfix(loss=f"{loss.item():.4f}", bottle_iou=f"{mean_biou:.3f}")

        avg_loss  = float(np.mean(epoch_losses))
        avg_biou  = float(np.mean(epoch_biou))

        val_metrics = validate(model, val_loader, device, criterion)

        history['train_loss'].append(avg_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_bottle_iou'].append(val_metrics['bottle_iou'])
        history['val_mean_iou'].append(val_metrics['mean_iou'])

        print(
            f"  Train loss: {avg_loss:.4f}  bottle IoU: {avg_biou:.3f} | "
            f"Val loss: {val_metrics['loss']:.4f}  "
            f"bottle IoU: {val_metrics['bottle_iou']:.3f}  "
            f"mean IoU: {val_metrics['mean_iou']:.3f}"
        )

        # Save best model (keyed on bottle IoU)
        if val_metrics['bottle_iou'] > best_bottle_iou:
            best_bottle_iou = val_metrics['bottle_iou']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_bottle_iou': best_bottle_iou,
                'history': history
            }, os.path.join(SAVE_DIR, "best_model.pt"))
            print(f"  >> Best model saved  (val bottle IoU = {best_bottle_iou:.4f})")

        # Periodic checkpoint
        if (epoch + 1) % SAVE_EVERY == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'history': history
            }, os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch+1}.pt"))

    # Save final model in HuggingFace format (for SegFormer.py inference)
    print("\nSaving final model in HuggingFace format ...")
    hf_dir = os.path.join(SAVE_DIR, "huggingface_model")
    os.makedirs(hf_dir, exist_ok=True)
    model.save_pretrained(hf_dir)
    processor.save_pretrained(hf_dir)

    # Also load & re-save BEST weights into HF format
    print("Saving BEST weights into HuggingFace format ...")
    ckpt = torch.load(os.path.join(SAVE_DIR, "best_model.pt"), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    best_hf_dir = os.path.join(SAVE_DIR, "huggingface_model_best")
    os.makedirs(best_hf_dir, exist_ok=True)
    model.save_pretrained(best_hf_dir)
    processor.save_pretrained(best_hf_dir)

    import pickle
    with open(os.path.join(SAVE_DIR, "history.pkl"), 'wb') as f:
        pickle.dump(history, f)

    print(f"\nDone. Best val bottle IoU: {best_bottle_iou:.4f}")
    print(f"Models saved to: {SAVE_DIR}/")
    print(f"  huggingface_model/      <- final epoch")
    print(f"  huggingface_model_best/ <- best bottle IoU (use this for inference)")


if __name__ == "__main__":
    main()
