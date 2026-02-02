#!/usr/bin/env python3
"""
Fine-tune DETR on Water Bottle Dataset - WORKING VERSION
Properly handles loss weights for single class detection
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
# CONFIGURATION
# ============================================================================
IMAGES_DIR = "./my_dataset/rgb"
ANNOTATIONS_FILE = './my_dataset/All-1.json'
SAVE_DIR = "./water_bottle_model"

EPOCHS = 50
BATCH_SIZE = 2
LEARNING_RATE = 1e-5
SAVE_EVERY = 10

# ============================================================================

class WaterBottleDataset(Dataset):

    def __init__(self, images_dir, annotations_file, processor):
        self.images_dir = images_dir
        self.processor = processor

        print("\n📂 Loading annotations...")
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)

        actual_files = set()
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            actual_files.update([f.name for f in Path(images_dir).glob(f'*{ext}')])

        self.img_id_to_info = {}
        for img in coco_data['images']:
            if img['file_name'] in actual_files:
                self.img_id_to_info[img['id']] = img

        self.img_to_anns = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id in self.img_id_to_info:
                if img_id not in self.img_to_anns:
                    self.img_to_anns[img_id] = []
                self.img_to_anns[img_id].append(ann)

        self.image_ids = list(self.img_to_anns.keys())

        if len(self.image_ids) == 0:
            raise ValueError("❌ No valid images found!")

        print(f"✅ Using {len(self.image_ids)} images")
        
        # Print sample annotations for verification
        if len(self.image_ids) > 0:
            sample_id = self.image_ids[0]
            sample_anns = self.img_to_anns[sample_id]
            print(f"\n📋 Sample image has {len(sample_anns)} annotations")
            if len(sample_anns) > 0:
                print(f"   Sample bbox: {sample_anns[0]['bbox']}")

        # Map ALL categories → 0 (bottle)
        self.category_mapping = {}
        for cat in coco_data['categories']:
            self.category_mapping[cat['id']] = 0

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.img_id_to_info[img_id]

        img_path = os.path.join(self.images_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')

        anns = self.img_to_anns[img_id]

        boxes = []
        labels = []

        img_w, img_h = image.size

        for ann in anns:
            x, y, w, h = ann['bbox']

            # Normalize to [0, 1] range - DETR format
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            norm_w = w / img_w
            norm_h = h / img_h
            
            # Sanity check - skip invalid boxes
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and norm_w > 0 and norm_h > 0):
                print(f"⚠️  Warning: Invalid bbox in image {img_info['file_name']}")
                continue

            boxes.append([x_center, y_center, norm_w, norm_h])
            labels.append(0)  # Only class: bottle

        if len(boxes) == 0:
            # Skip images with no valid boxes
            print(f"⚠️  Skipping {img_info['file_name']} - no valid boxes")
            return self.__getitem__((idx + 1) % len(self.image_ids))

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

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
    print("="*60)
    print("DETR 1-CLASS (BOTTLE) TRAINING - WORKING VERSION")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n📦 Loading DETR model...")
    
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    
    # Load model with num_labels=2 (1 bottle + 1 no-object)
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    
    # Update config
    model.config.num_labels = 2
    model.config.id2label = {0: "bottle", 1: "no-object"}
    model.config.label2id = {"bottle": 0, "no-object": 1}
    
    model.to(device)
    
    # CRITICAL FIX: Configure the loss function weights
    # The criterion needs proper weight configuration for num_labels=2
    if hasattr(model, 'criterion'):
        # Create weight tensor: [bottle, no-object, background]
        # Shape should be [num_labels + 1]
        empty_weight = torch.ones(model.config.num_labels + 1).to(device)
        empty_weight[-1] = 0.1  # Lower weight for background class
        model.criterion.empty_weight = empty_weight
        print(f"\n🔧 Set loss weights: {empty_weight.cpu().numpy()}")
    
    print("\n✅ Model ready")
    print(f"Num labels: {model.config.num_labels}")
    print(f"Label mapping: {model.config.id2label}")
    print(f"Classifier output features: {model.class_labels_classifier.out_features}")

    dataset = WaterBottleDataset(IMAGES_DIR, ANNOTATIONS_FILE, processor)

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"\n🚀 Starting training...")
    print(f"Total images: {len(dataset)}")
    print(f"Total batches per epoch: {len(train_loader)}")
    print(f"Training for {EPOCHS} epochs\n")

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

                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"\n⚠️  NaN loss at epoch {epoch+1}, batch {batch_idx}")
                    continue

                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                
                optimizer.step()

                losses.append(loss.item())
                pbar.set_postfix(loss=loss.item())
                
            except RuntimeError as e:
                print(f"\n⚠️  Error in batch {batch_idx}: {e}")
                continue

        if len(losses) == 0:
            print(f"⚠️  Epoch {epoch+1}: No successful batches!")
            continue
            
        avg_loss = np.mean(losses)
        min_loss = np.min(losses)
        max_loss = np.max(losses)
        
        print(f"\nEpoch {epoch+1}/{EPOCHS} Summary:")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Min Loss: {min_loss:.4f}")
        print(f"  Max Loss: {max_loss:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_checkpoint = os.path.join(SAVE_DIR, "best_model.pt")
            torch.save(model.state_dict(), best_checkpoint)
            print(f"  💾 New best model saved! (loss: {best_loss:.4f})")

        # Save periodic checkpoints
        if (epoch + 1) % SAVE_EVERY == 0:
            checkpoint_path = os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  💾 Checkpoint saved: {checkpoint_path}")

    print("\n" + "="*60)
    print("💾 Saving final model...")

    hf_save_dir = os.path.join(SAVE_DIR, "huggingface_model")
    model.save_pretrained(hf_save_dir)
    processor.save_pretrained(hf_save_dir)

    print(f"✅ Model saved to: {hf_save_dir}")
    print(f"\nTraining Statistics:")
    print(f"  Best Loss: {best_loss:.4f}")
    print(f"  Final Loss: {avg_loss:.4f}")
    
    if avg_loss > 2.0:
        print("\n⚠️  Warning: High final loss - model may need:")
        print("   - More training epochs")
        print("   - Better data quality/diversity")
        print("   - Data augmentation")
    elif avg_loss < 1.0:
        print("\n✅ Good training! Loss is low.")
    else:
        print("\n✅ Reasonable training. Test the model to verify performance.")

    print("\nTraining complete 🎉")
    print("="*60)


if __name__ == "__main__":
    main()