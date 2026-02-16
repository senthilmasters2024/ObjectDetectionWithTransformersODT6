#!/usr/bin/env python3
"""
Fine-tune DETR on Hard Negative Examples (Category 2: "No Object")
Filters annotations to use ONLY category_id=2 (objects that look like bottles but aren't)
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

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================================

# Hard negative dataset paths
NEGATIVE_IMAGES_DIR = "/home/frauas/ODT6/ObjectDetectionWithTransformersODT6/my_dataset/TrainingDatasetNoBottle/rgb"
NEGATIVE_ANNOTATIONS_FILE = '/home/frauas/ODT6/ObjectDetectionWithTransformersODT6/my_dataset/All-3.json'

# Pre-trained model from previous positive training
PRETRAINED_MODEL_DIR = "./water_bottle_model/huggingface_model"

SAVE_DIR = "./water_bottle_model_hard_negatives"

EPOCHS = 40
BATCH_SIZE = 2
LEARNING_RATE = 5e-6  # Lower LR for fine-tuning
SAVE_EVERY = 5

# Training strategy
HARD_NEGATIVE_STRATEGY = "suppress"  # "suppress" or "background"

# Category ID for hard negatives in All-3.json
HARD_NEGATIVE_CATEGORY_ID = 2  # "No Object" category

# ============================================================================

class HardNegativeDataset(Dataset):
    """
    Dataset for HARD negatives - Category 2 ("No Object")
    Objects that look like bottles but are explicitly labeled as NOT bottles
    """

    def __init__(self, images_dir, annotations_file, processor, 
                 hard_negative_cat_id=2, strategy="suppress"):
        self.images_dir = images_dir
        self.processor = processor
        self.strategy = strategy
        self.hard_negative_cat_id = hard_negative_cat_id
        self.dataset_type = "HARD_NEGATIVE"

        print(f"\n📂 Loading annotations from {annotations_file}...")
        print(f"   Filtering for category_id={hard_negative_cat_id} (hard negatives)")
        print(f"   Strategy: {strategy}")
        
        if not os.path.exists(annotations_file):
            raise FileNotFoundError(f"❌ Annotation file not found: {annotations_file}")
        
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"❌ Images directory not found: {images_dir}")
        
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)

        # Get actual image files
        actual_files = set()
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            actual_files.update([f.name for f in Path(images_dir).glob(f'*{ext}')])

        print(f"   Found {len(actual_files)} image files in directory")

        # Build image mapping
        self.img_id_to_info = {}
        for img in coco_data['images']:
            if img['file_name'] in actual_files:
                self.img_id_to_info[img['id']] = img

        print(f"   Matched {len(self.img_id_to_info)} images with annotations")

        # Build annotation mapping - FILTER FOR HARD NEGATIVES ONLY
        self.img_to_anns = {}
        total_annotations = len(coco_data.get('annotations', []))
        total_hard_negatives = 0
        total_bottles = 0
        
        for ann in coco_data.get('annotations', []):
            img_id = ann['image_id']
            cat_id = ann['category_id']
            
            # Count category distribution
            if cat_id == hard_negative_cat_id:
                total_hard_negatives += 1
            else:
                total_bottles += 1
            
            # ONLY include hard negative annotations (category_id=2)
            if img_id in self.img_id_to_info and cat_id == hard_negative_cat_id:
                if img_id not in self.img_to_anns:
                    self.img_to_anns[img_id] = []
                self.img_to_anns[img_id].append(ann)

        # Only use images that have hard negative annotations
        self.image_ids = [img_id for img_id in self.img_to_anns.keys() 
                         if len(self.img_to_anns[img_id]) > 0]
        
        print(f"\n📊 Annotation Statistics:")
        print(f"   Total annotations in file: {total_annotations}")
        print(f"   Bottle annotations (category ≠ {hard_negative_cat_id}): {total_bottles}")
        print(f"   Hard negative annotations (category = {hard_negative_cat_id}): {total_hard_negatives}")
        
        if len(self.image_ids) == 0:
            print(f"\n❌ No images with hard negative annotations found!")
            print(f"   Make sure category_id={hard_negative_cat_id} exists in your annotations")
            raise ValueError("No valid hard negative examples found!")
        
        hard_neg_count = sum(len(anns) for anns in self.img_to_anns.values())
        print(f"\n✅ HARD NEGATIVES DATASET:")
        print(f"   Images with hard negatives: {len(self.image_ids)}")
        print(f"   Total hard negative annotations: {hard_neg_count}")
        print(f"   Average per image: {hard_neg_count/len(self.image_ids):.1f}")
        
        # Show sample
        if len(self.image_ids) > 0:
            sample_id = self.image_ids[0]
            sample_anns = self.img_to_anns[sample_id]
            print(f"\n📋 Sample Image:")
            print(f"   Filename: {self.img_id_to_info[sample_id]['file_name']}")
            print(f"   Hard negatives in this image: {len(sample_anns)}")
            if len(sample_anns) > 0:
                print(f"   Sample bbox: {sample_anns[0]['bbox']}")

        # Show category info
        self.categories = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
        print(f"\n📚 All Categories in file:")
        for cat_id, cat_name in sorted(self.categories.items()):
            marker = "← USING THIS" if cat_id == hard_negative_cat_id else ""
            print(f"   Category {cat_id}: {cat_name} {marker}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.img_id_to_info[img_id]
        img_filename = img_info['file_name']

        img_path = os.path.join(self.images_dir, img_filename)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️  Error loading {img_filename}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        anns = self.img_to_anns[img_id]
        img_w, img_h = image.size

        if self.strategy == "suppress":
            # Strategy 1: SUPPRESS - No boxes (teach model to ignore)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            
        elif self.strategy == "background":
            # Strategy 2: BACKGROUND - Boxes with "no-object" label
            boxes = []
            labels = []
            
            for ann in anns:
                x, y, w, h = ann['bbox']
                
                # Normalize to [0, 1] range
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                norm_w = w / img_w
                norm_h = h / img_h
                
                # Skip invalid boxes
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and norm_w > 0 and norm_h > 0):
                    continue
                
                boxes.append([x_center, y_center, norm_w, norm_h])
                labels.append(1)  # Class 1 = "no-object"
            
            if len(boxes) == 0:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
            else:
                boxes = torch.tensor(boxes, dtype=torch.float32)
                labels = torch.tensor(labels, dtype=torch.int64)
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        encoding = self.processor(images=image, return_tensors="pt")
        pixel_values = encoding['pixel_values'].squeeze()

        target = {
            'boxes': boxes,
            'class_labels': labels
        }

        return pixel_values, target


def collate_fn(batch):
    pixel_values = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return pixel_values, targets


def main():
    print("="*70)
    print("DETR HARD NEGATIVE TRAINING")
    print("Training on Category 2 'No Object' annotations")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    print("\n📦 Loading pre-trained DETR model...")
    
    if os.path.exists(PRETRAINED_MODEL_DIR):
        print(f"   Loading from: {PRETRAINED_MODEL_DIR}")
        processor = DetrImageProcessor.from_pretrained(PRETRAINED_MODEL_DIR)
        model = DetrForObjectDetection.from_pretrained(PRETRAINED_MODEL_DIR)
        print("   ✅ Loaded pre-trained model")
    else:
        print(f"   ⚠️  Pre-trained model not found: {PRETRAINED_MODEL_DIR}")
        print(f"   Loading base DETR model (NOT recommended for fine-tuning)")
        
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=2,
            ignore_mismatched_sizes=True
        )
        model.config.num_labels = 2
        model.config.id2label = {0: "bottle", 1: "no-object"}
        model.config.label2id = {"bottle": 0, "no-object": 1}
    
    model.to(device)
    
    if hasattr(model, 'criterion'):
        empty_weight = torch.ones(model.config.num_labels + 1).to(device)
        empty_weight[-1] = 0.1
        model.criterion.empty_weight = empty_weight
        print(f"   🔧 Loss weights: {empty_weight.cpu().numpy()}")
    
    print(f"   Label mapping: {model.config.id2label}")

    print("\n" + "="*70)
    print("LOADING HARD NEGATIVE DATASET")
    print("="*70)
    
    try:
        dataset = HardNegativeDataset(
            NEGATIVE_IMAGES_DIR,
            NEGATIVE_ANNOTATIONS_FILE,
            processor,
            hard_negative_cat_id=HARD_NEGATIVE_CATEGORY_ID,
            strategy=HARD_NEGATIVE_STRATEGY
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"\n⚙️  Training Configuration:")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Strategy: {HARD_NEGATIVE_STRATEGY}")
    print(f"   Batches per epoch: {len(train_loader)}")

    print(f"\n🚀 Starting training...\n")

    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch_idx, (pixel_values, targets) in enumerate(pbar):
            pixel_values = pixel_values.to(device)

            batch_targets = []
            for t in targets:
                batch_targets.append({
                    'boxes': t['boxes'].to(device),
                    'class_labels': t['class_labels'].to(device)
                })

            try:
                outputs = model(pixel_values=pixel_values, labels=batch_targets)
                loss = outputs.loss

                if torch.isnan(loss):
                    print(f"\n⚠️  NaN loss at epoch {epoch+1}, batch {batch_idx}")
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()

                losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                
            except RuntimeError as e:
                print(f"\n⚠️  Error in batch {batch_idx}: {e}")
                continue

        if len(losses) == 0:
            print(f"⚠️  Epoch {epoch+1}: No successful batches!")
            continue
            
        avg_loss = np.mean(losses)
        min_loss = np.min(losses)
        max_loss = np.max(losses)
        
        print(f"\n📊 Epoch {epoch+1}/{EPOCHS}:")
        print(f"   Avg Loss: {avg_loss:.4f} | Min: {min_loss:.4f} | Max: {max_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_checkpoint = os.path.join(SAVE_DIR, "best_model.pt")
            torch.save(model.state_dict(), best_checkpoint)
            print(f"   💾 Best model saved! (loss: {best_loss:.4f})")

        if (epoch + 1) % SAVE_EVERY == 0:
            checkpoint_path = os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"   💾 Checkpoint saved: epoch_{epoch+1}")

    print("\n" + "="*70)
    print("💾 Saving final model...")

    hf_save_dir = os.path.join(SAVE_DIR, "huggingface_model")
    model.save_pretrained(hf_save_dir)
    processor.save_pretrained(hf_save_dir)

    print(f"\n✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📈 Final Statistics:")
    print(f"   Best Loss: {best_loss:.4f}")
    print(f"   Final Loss: {avg_loss:.4f}")
    print(f"\n💾 Model saved to: {hf_save_dir}")
    print("\n🎯 Next Steps:")
    print("   1. Test on images with bottle-like objects")
    print("   2. Verify actual bottles are still detected")
    print("   3. Measure false positive reduction")
    print("="*70)


if __name__ == "__main__":
    main()