#!/usr/bin/env python3
"""
SegFormer M1 + Detection Head Training
----------------------------------------
Architecture:
  RGBEncoder  (SegFormer-b0 backbone, from ODTSegformer/models/encoder_rgb.py)
  +
  SegFormerDetectionHead  (same design as models/detection_head.py,
                           with the pos/query_pos kwargs fixed for standard
                           nn.TransformerDecoder compatibility)

Dataset:   All-1.json  (polygon annotations -> bbox used for detection)
Loss:      DETR-style Hungarian matching  (class CE + L1 + GIoU)
"""

import sys
import os

# Make ODTSegformer models importable
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "ODTSegformer", "ObjectDetectionWithTransformersODT6")
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerImageProcessor
from PIL import Image
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import train_test_split

# Reuse existing model components from ODTSegformer
from models.encoder_rgb import RGBEncoder
from models.detection_head import PositionEmbeddingSine, MLP

# ============================================================
# CONFIGURATION
# ============================================================

IMAGES_DIR       = "./my_dataset/rgb"
ANNOTATIONS_FILE = "./my_dataset/All-1.json"
SAVE_DIR         = "./segformer_m1_detection_head"

EPOCHS        = 50
BATCH_SIZE    = 2
LEARNING_RATE = 1e-5
SAVE_EVERY    = 10
IMAGE_SIZE    = (512, 512)
TRAIN_SPLIT   = 0.85
RANDOM_SEED   = 42

NUM_QUERIES  = 100
HIDDEN_DIM   = 256
NUM_CLASSES  = 1          # bottle  (no-object = index NUM_CLASSES)

# SegFormer-b0 encoder output channel dims per level
FEATURE_DIMS = [32, 64, 160, 256]

SEGFORMER_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"

# ============================================================
# MODEL
# ============================================================

class SegFormerDetectionHead(nn.Module):
    """
    Same design as models/detection_head.py.
    Fix: positional embeddings are added directly to tgt and memory
    before the standard nn.TransformerDecoder call, which does not
    accept pos/query_pos as keyword arguments.
    """

    def __init__(
        self,
        feature_dims=FEATURE_DIMS,
        hidden_dim=HIDDEN_DIM,
        num_queries=NUM_QUERIES,
        num_classes=NUM_CLASSES + 1,   # +1 for no-object
        num_decoder_layers=6,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim  = hidden_dim
        self.num_classes = num_classes

        # Project each encoder level to hidden_dim  (same as detection_head.py)
        self.input_projections = nn.ModuleList([
            nn.Conv2d(dim, hidden_dim, kernel_size=1)
            for dim in feature_dims
        ])

        # Sine positional encoding  (reused from detection_head.py)
        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2)

        # Learnable object queries  (same as detection_head.py)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Standard PyTorch transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=False,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # Prediction heads  (reused MLP from detection_head.py)
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed  = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

    def forward(self, multi_scale_features):
        """
        Args:
            multi_scale_features: list of 4 tensors from RGBEncoder,
                                  shapes (B, C_i, H_i, W_i)
        Returns:
            pred_logits : (B, num_queries, num_classes)
            pred_boxes  : (B, num_queries, 4)  normalized [cx,cy,w,h]
        """
        B = multi_scale_features[0].shape[0]

        proj_list = []
        pos_list  = []
        for i, feat in enumerate(multi_scale_features):
            proj = self.input_projections[i](feat)         # (B, D, H, W)
            pos  = self.position_embedding(proj)            # (B, D, H, W)
            proj_list.append(proj.flatten(2).permute(2, 0, 1))   # (H*W, B, D)
            pos_list.append(pos.flatten(2).permute(2, 0, 1))     # (H*W, B, D)

        memory    = torch.cat(proj_list, dim=0)   # (sum_HW, B, D)
        pos_embed = torch.cat(pos_list,  dim=0)   # (sum_HW, B, D)

        # Object queries
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # (Q, B, D)
        tgt         = torch.zeros_like(query_embed)

        # Add positional info before the standard decoder
        # (equivalent to what detection_head.py intends with pos/query_pos)
        hs = self.transformer_decoder(
            tgt + query_embed,       # (Q, B, D)  with query positional embed
            memory + pos_embed,      # (HW, B, D) with spatial positional embed
        )   # -> (Q, B, D)

        hs = hs.permute(1, 0, 2)    # (B, Q, D)

        pred_logits = self.class_embed(hs)           # (B, Q, num_classes)
        pred_boxes  = self.bbox_embed(hs).sigmoid()  # (B, Q, 4)

        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}


class SegFormerM1Detector(nn.Module):
    """
    Full M1 detector:
        RGBEncoder (SegFormer-b0) -> SegFormerDetectionHead
    """

    def __init__(self):
        super().__init__()
        self.encoder = RGBEncoder(model_name=SEGFORMER_MODEL)
        self.head    = SegFormerDetectionHead(
            feature_dims=FEATURE_DIMS,
            hidden_dim=HIDDEN_DIM,
            num_queries=NUM_QUERIES,
            num_classes=NUM_CLASSES + 1,
        )

    def forward(self, pixel_values):
        features = self.encoder(pixel_values)   # tuple of 4 hidden states
        return self.head(list(features))


# ============================================================
# DATASET
# ============================================================

class BottleDetectionDataset(Dataset):
    """
    Reads All-1.json and returns image tensors + normalized [cx,cy,w,h] boxes.
    """

    def __init__(self, images_dir, annotations_file, processor,
                 image_size=(512, 512), split="train",
                 train_ids=None, val_ids=None):
        self.images_dir = images_dir
        self.processor  = processor
        self.image_size = image_size

        with open(annotations_file) as f:
            coco = json.load(f)

        # Case-insensitive filename lookup
        actual_files = {}
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            for p in Path(images_dir).glob(f"*{ext}"):
                actual_files[p.name.lower()] = p.name

        self.img_id_to_info = {}
        for img in coco["images"]:
            key = img["file_name"].lower()
            if key in actual_files:
                img["file_name"] = actual_files[key]
                self.img_id_to_info[img["id"]] = img

        bottle_cat_ids = {
            c["id"] for c in coco["categories"]
            if "bottle" in c["name"].lower()
        }

        self.img_to_anns = {}
        for ann in coco["annotations"]:
            if ann["image_id"] not in self.img_id_to_info:
                continue
            if ann.get("category_id") not in bottle_cat_ids:
                continue
            bbox = ann.get("bbox", [])
            if len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
                self.img_to_anns.setdefault(ann["image_id"], []).append(ann)

        self.image_ids = train_ids if split == "train" else val_ids
        print(f"  [{split}] {len(self.image_ids)} images with bottle annotations")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id   = self.image_ids[idx]
        img_info = self.img_id_to_info[img_id]
        img_path = os.path.join(self.images_dir, img_info["file_name"])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  Warning: {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.image_ids))

        orig_w, orig_h = image.size
        boxes, labels  = [], []

        for ann in self.img_to_anns.get(img_id, []):
            x, y, w, h = ann["bbox"]
            cx = (x + w / 2) / orig_w
            cy = (y + h / 2) / orig_h
            nw = w / orig_w
            nh = h / orig_h
            if 0 < cx <= 1 and 0 < cy <= 1 and nw > 0 and nh > 0:
                boxes.append([cx, cy, nw, nh])
                labels.append(0)   # bottle = class 0

        if not boxes:
            return self.__getitem__((idx + 1) % len(self.image_ids))

        image_r = image.resize(self.image_size, Image.BILINEAR)
        enc = self.processor(image_r, return_tensors="pt")
        pixel_values = enc["pixel_values"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "boxes":  torch.tensor(boxes,  dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "boxes":        [b["boxes"]  for b in batch],
        "labels":       [b["labels"] for b in batch],
    }


# ============================================================
# DETR-STYLE LOSS
# ============================================================

def box_cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def generalized_iou(b1, b2):
    """GIoU for paired boxes in xyxy format. b1,b2: (N,4)"""
    x1 = torch.max(b1[:, 0], b2[:, 0])
    y1 = torch.max(b1[:, 1], b2[:, 1])
    x2 = torch.min(b1[:, 2], b2[:, 2])
    y2 = torch.min(b1[:, 3], b2[:, 3])

    inter  = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area1  = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2  = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    union  = area1 + area2 - inter
    iou    = inter / union.clamp(min=1e-6)

    enc_x1 = torch.min(b1[:, 0], b2[:, 0])
    enc_y1 = torch.min(b1[:, 1], b2[:, 1])
    enc_x2 = torch.max(b1[:, 2], b2[:, 2])
    enc_y2 = torch.max(b1[:, 3], b2[:, 3])
    enc_area = (enc_x2 - enc_x1).clamp(0) * (enc_y2 - enc_y1).clamp(0)

    return iou - (enc_area - union) / enc_area.clamp(min=1e-6)


def hungarian_match(pred_boxes, pred_logits, gt_boxes, gt_labels):
    """
    Args:
        pred_boxes   : (Q, 4) normalized cxcywh
        pred_logits  : (Q, C) raw logits
        gt_boxes     : (N, 4) normalized cxcywh
        gt_labels    : (N,)   int class indices
    Returns:
        pred_idx, gt_idx  (lists of matched indices)
    """
    N = len(gt_boxes)
    if N == 0:
        return [], []

    with torch.no_grad():
        probs      = pred_logits.softmax(-1)           # (Q, C)
        cls_cost   = -probs[:, gt_labels]              # (Q, N)
        l1_cost    = torch.cdist(pred_boxes, gt_boxes) # (Q, N)

        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)    # (Q, 4)
        gt_xyxy   = box_cxcywh_to_xyxy(gt_boxes)      # (N, 4)

        # GIoU cost: expand to (Q*N, 4) pairs
        Q = pred_boxes.shape[0]
        pred_exp = pred_xyxy.unsqueeze(1).expand(Q, N, 4).reshape(Q * N, 4)
        gt_exp   = gt_xyxy.unsqueeze(0).expand(Q, N, 4).reshape(Q * N, 4)
        giou_cost = -generalized_iou(pred_exp, gt_exp).reshape(Q, N)

        cost = cls_cost + 5.0 * l1_cost + 2.0 * giou_cost

    pi, gi = linear_sum_assignment(cost.cpu().numpy())
    return pi.tolist(), gi.tolist()


def compute_loss(outputs, gt_boxes_list, gt_labels_list, device):
    """
    DETR-style set prediction loss over a batch.
    """
    pred_logits = outputs["pred_logits"]   # (B, Q, C)
    pred_boxes  = outputs["pred_boxes"]    # (B, Q, 4)
    B, Q, C     = pred_logits.shape
    no_obj      = NUM_CLASSES              # last class index = no-object

    total_cls  = torch.tensor(0.0, device=device)
    total_box  = torch.tensor(0.0, device=device)
    total_giou = torch.tensor(0.0, device=device)

    for b in range(B):
        gt_boxes  = gt_boxes_list[b].to(device)    # (N, 4)
        gt_labels = gt_labels_list[b].to(device)   # (N,)

        # Default all queries to no-object
        tgt_cls = torch.full((Q,), no_obj, dtype=torch.long, device=device)

        if len(gt_boxes) == 0:
            total_cls = total_cls + F.cross_entropy(pred_logits[b], tgt_cls)
            continue

        pi, gi = hungarian_match(
            pred_boxes[b].detach(),
            pred_logits[b].detach(),
            gt_boxes, gt_labels
        )

        if not pi:
            total_cls = total_cls + F.cross_entropy(pred_logits[b], tgt_cls)
            continue

        pi_t = torch.tensor(pi, dtype=torch.long, device=device)
        gi_t = torch.tensor(gi, dtype=torch.long, device=device)

        tgt_cls[pi_t] = gt_labels[gi_t]
        total_cls = total_cls + F.cross_entropy(pred_logits[b], tgt_cls)

        matched_pred = pred_boxes[b][pi_t]    # (M, 4)
        matched_gt   = gt_boxes[gi_t]          # (M, 4)

        total_box  = total_box  + F.l1_loss(matched_pred, matched_gt)
        giou       = generalized_iou(
            box_cxcywh_to_xyxy(matched_pred),
            box_cxcywh_to_xyxy(matched_gt)
        )
        total_giou = total_giou + (1 - giou).mean()

    loss = (total_cls + 5.0 * total_box + 2.0 * total_giou) / B
    info = {
        "cls":  (total_cls  / B).item(),
        "box":  (total_box  / B).item(),
        "giou": (total_giou / B).item(),
    }
    return loss, info


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("SEGFORMER M1 + DETECTION HEAD  (All-1.json)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    processor = SegformerImageProcessor.from_pretrained(SEGFORMER_MODEL)

    print("\nBuilding model...")
    model = SegFormerM1Detector()
    model.to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params    : {n_total:,}")
    print(f"  Trainable       : {n_train:,}")

    # ---- Build train/val split ----
    print("\nBuilding split from All-1.json ...")
    with open(ANNOTATIONS_FILE) as f:
        coco = json.load(f)

    actual_files = {}
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        for p in Path(IMAGES_DIR).glob(f"*{ext}"):
            actual_files[p.name.lower()] = p.name

    img_id_to_info = {
        img["id"]: img for img in coco["images"]
        if img["file_name"].lower() in actual_files
    }
    bottle_cat_ids = {
        c["id"] for c in coco["categories"] if "bottle" in c["name"].lower()
    }
    img_to_anns = {}
    for ann in coco["annotations"]:
        if ann["image_id"] not in img_id_to_info:
            continue
        if ann.get("category_id") not in bottle_cat_ids:
            continue
        bbox = ann.get("bbox", [])
        if len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
            img_to_anns.setdefault(ann["image_id"], []).append(ann)

    all_ids = list(img_to_anns.keys())
    train_ids, val_ids = train_test_split(
        all_ids, train_size=TRAIN_SPLIT, random_state=RANDOM_SEED
    )
    print(f"  Total={len(all_ids)}  Train={len(train_ids)}  Val={len(val_ids)}")

    train_ds = BottleDetectionDataset(
        IMAGES_DIR, ANNOTATIONS_FILE, processor,
        image_size=IMAGE_SIZE, split="train",
        train_ids=train_ids, val_ids=val_ids,
    )
    val_ds = BottleDetectionDataset(
        IMAGES_DIR, ANNOTATIONS_FILE, processor,
        image_size=IMAGE_SIZE, split="val",
        train_ids=train_ids, val_ids=val_ids,
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # Encoder gets 10x lower LR (it's already pretrained)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": LEARNING_RATE * 0.1},
        {"params": model.head.parameters(),    "lr": LEARNING_RATE},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE * 0.01
    )

    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"\nTraining: {EPOCHS} epochs  |  batch={BATCH_SIZE}  |  lr={LEARNING_RATE}")
    print(f"  Encoder LR : {LEARNING_RATE * 0.1:.2e}  (pretrained, lower rate)")
    print(f"  Head LR    : {LEARNING_RATE:.2e}")

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [],
               "train_cls": [], "train_box": [], "train_giou": []}

    for epoch in range(EPOCHS):
        # ---------- TRAIN ----------
        model.train()
        t_loss, t_cls, t_box, t_giou = [], [], [], []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in pbar:
            pv     = batch["pixel_values"].to(device)
            boxes  = batch["boxes"]
            labels = batch["labels"]

            outputs      = model(pv)
            loss, info   = compute_loss(outputs, boxes, labels, device)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            t_loss.append(loss.item())
            t_cls.append(info["cls"])
            t_box.append(info["box"])
            t_giou.append(info["giou"])
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                cls=f"{info['cls']:.3f}",
                box=f"{info['box']:.3f}",
                giou=f"{info['giou']:.3f}",
            )

        scheduler.step()

        # ---------- VALIDATE ----------
        model.eval()
        v_loss = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="  Val", leave=False):
                pv    = batch["pixel_values"].to(device)
                loss, _ = compute_loss(
                    model(pv), batch["boxes"], batch["labels"], device
                )
                v_loss.append(loss.item())

        avg_t = float(np.mean(t_loss))
        avg_v = float(np.mean(v_loss))

        history["train_loss"].append(avg_t)
        history["val_loss"].append(avg_v)
        history["train_cls"].append(float(np.mean(t_cls)))
        history["train_box"].append(float(np.mean(t_box)))
        history["train_giou"].append(float(np.mean(t_giou)))

        print(f"\nEpoch {epoch+1}/{EPOCHS}:")
        print(f"  Train  loss={avg_t:.4f}  cls={np.mean(t_cls):.3f}  "
              f"box={np.mean(t_box):.3f}  giou={np.mean(t_giou):.3f}")
        print(f"  Val    loss={avg_v:.4f}")
        print(f"  LR={scheduler.get_last_lr()[0]:.2e}")

        if avg_v < best_val_loss:
            best_val_loss = avg_v
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "history": history,
                },
                os.path.join(SAVE_DIR, "best_model.pt"),
            )
            print(f"  ** Best model saved (val_loss={best_val_loss:.4f}) **")

        if (epoch + 1) % SAVE_EVERY == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "history": history,
                },
                os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch+1}.pt"),
            )
            print("  Checkpoint saved")

    # Save final weights
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "final_model.pt"))

    import pickle
    with open(os.path.join(SAVE_DIR, "history.pkl"), "wb") as f:
        pickle.dump(history, f)

    print(f"\nDone.  Best Val Loss = {best_val_loss:.4f}")
    print(f"Model saved to: {SAVE_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
