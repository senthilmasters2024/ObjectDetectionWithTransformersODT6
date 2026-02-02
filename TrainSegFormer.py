#!/usr/bin/env python3
"""
Complete SegFormer Training for Water Bottle Segmentation
Uses polygon annotations from COCO format
Includes train/val split, metrics, and visualization
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image, ImageDraw
import json
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split

# ============================================================================
# CONFIGURATION
# ============================================================================
IMAGES_DIR = "./my_dataset/rgb"
ANNOTATIONS_FILE = './my_dataset/All-2.json'
SAVE_DIR = "./water_bottle_segformer_model"

EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 6e-5
SAVE_EVERY = 10

# Image size for SegFormer (must be multiple of 32)
IMAGE_SIZE = (512, 512)

# Train/Val split
TRAIN_SPLIT = 0.85
RANDOM_SEED = 42

# ============================================================================

class BottleSegmentationDataset(Dataset):
    """Dataset for bottle segmentation with train/val split"""

    def __init__(self, images_dir, annotations_file, processor, image_size=(512, 512), 
                 split='train', train_ids=None, val_ids=None):
        """
        Args:
            images_dir: Path to images directory
            annotations_file: Path to COCO JSON with segmentation
            processor: SegformerImageProcessor
            image_size: Tuple of (width, height) for resizing
            split: 'train' or 'val'
            train_ids: List of image IDs for training (if split already done)
            val_ids: List of image IDs for validation (if split already done)
        """
        self.images_dir = images_dir
        self.processor = processor
        self.image_size = image_size
        self.split = split

        print(f"\n📂 Loading segmentation annotations for {split}...")
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)

        # Get actual files (case-insensitive matching)
        actual_files_dict = {}
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            for f in Path(images_dir).glob(f'*{ext}'):
                actual_files_dict[f.name.lower()] = f.name

        print(f"📁 Found {len(actual_files_dict)} image files in directory")

        # Match images (case-insensitive)
        self.img_id_to_info = {}
        for img in coco_data['images']:
            fname = img['file_name']
            fname_lower = fname.lower()
            if fname_lower in actual_files_dict:
                img['file_name'] = actual_files_dict[fname_lower]
                self.img_id_to_info[img['id']] = img

        print(f"📋 Matched {len(self.img_id_to_info)} images from JSON")

        # Group annotations by image
        self.img_to_anns = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id in self.img_id_to_info:
                if img_id not in self.img_to_anns:
                    self.img_to_anns[img_id] = []
                self.img_to_anns[img_id].append(ann)

        # Get all image IDs with annotations
        all_image_ids = list(self.img_to_anns.keys())
        
        # Use provided split or create new one
        if train_ids is not None and val_ids is not None:
            if split == 'train':
                self.image_ids = train_ids
            else:
                self.image_ids = val_ids
        else:
            # Create split
            train_ids, val_ids = train_test_split(
                all_image_ids, 
                train_size=TRAIN_SPLIT, 
                random_state=RANDOM_SEED
            )
            if split == 'train':
                self.image_ids = train_ids
            else:
                self.image_ids = val_ids

        if len(self.image_ids) == 0:
            raise ValueError(f"❌ No valid images found for {split} split!")

        print(f"✅ Using {len(self.image_ids)} images for {split}")
        
        # Sample check
        if len(self.image_ids) > 0:
            sample_id = self.image_ids[0]
            sample_anns = self.img_to_anns[sample_id]
            print(f"   Sample: {len(sample_anns)} annotations")
            if len(sample_anns) > 0 and 'segmentation' in sample_anns[0]:
                seg = sample_anns[0]['segmentation']
                if isinstance(seg, list) and len(seg) > 0:
                    print(f"   Segmentation: {len(seg[0])} polygon points")

    def polygon_to_mask(self, polygons, image_size):
        """Convert COCO polygon to binary mask"""
        mask = Image.new('L', image_size, 0)
        draw = ImageDraw.Draw(mask)
        
        for polygon in polygons:
            if len(polygon) >= 6:  # At least 3 points
                points = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
                draw.polygon(points, outline=1, fill=1)
        
        return np.array(mask)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.img_id_to_info[img_id]

        img_path = os.path.join(self.images_dir, img_info['file_name'])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️ Error loading {img_path}: {e}")
            # Return next image
            return self.__getitem__((idx + 1) % len(self.image_ids))
        
        original_size = image.size  # (width, height)
        anns = self.img_to_anns[img_id]

        # Create segmentation mask
        # 0 = background, 1 = bottle
        mask = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
        
        for ann in anns:
            if 'segmentation' in ann:
                seg = ann['segmentation']
                if isinstance(seg, list) and len(seg) > 0:
                    try:
                        poly_mask = self.polygon_to_mask(seg, image.size)
                        mask = np.maximum(mask, poly_mask)
                    except Exception as e:
                        print(f"⚠️ Error creating mask for image {img_id}: {e}")
                        continue

        # Check if mask has any annotations
        if mask.max() == 0:
            print(f"⚠️ No valid segmentation for image {img_info['file_name']}")
            return self.__getitem__((idx + 1) % len(self.image_ids))

        # Resize image and mask to model input size
        image_resized = image.resize(self.image_size, Image.BILINEAR)
        mask_resized = Image.fromarray(mask).resize(self.image_size, Image.NEAREST)
        mask_resized = np.array(mask_resized)

        # Process with SegFormer processor
        encoding = self.processor(image_resized, return_tensors="pt")
        
        # Remove batch dimension
        pixel_values = encoding['pixel_values'].squeeze(0)
        
        # Convert mask to tensor
        segmentation_mask = torch.from_numpy(mask_resized).long()

        return {
            'pixel_values': pixel_values,
            'labels': segmentation_mask
        }


def collate_fn(batch):
    """Collate function for dataloader"""
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return {'pixel_values': pixel_values, 'labels': labels}


def compute_metrics(pred_masks, true_masks, num_classes=2):
    """
    Compute IoU and accuracy
    
    Args:
        pred_masks: Predicted segmentation (B, H, W)
        true_masks: Ground truth segmentation (B, H, W)
        num_classes: Number of classes
        
    Returns:
        accuracy, mean_iou, per_class_iou
    """
    pred_masks = pred_masks.cpu().numpy()
    true_masks = true_masks.cpu().numpy()
    
    # Flatten
    pred_flat = pred_masks.flatten()
    true_flat = true_masks.flatten()
    
    # Accuracy
    accuracy = (pred_flat == true_flat).mean()
    
    # Per-class IoU
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred_flat == cls)
        true_cls = (true_flat == cls)
        
        intersection = (pred_cls & true_cls).sum()
        union = (pred_cls | true_cls).sum()
        
        if union > 0:
            iou = intersection / union
            ious.append(iou)
        else:
            ious.append(float('nan'))
    
    # Mean IoU (excluding NaN values)
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    mean_iou = np.mean(valid_ious) if len(valid_ious) > 0 else 0.0
    
    return accuracy, mean_iou, ious


def validate(model, val_loader, device):
    """Run validation"""
    model.eval()
    
    val_losses = []
    val_accs = []
    val_ious = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            # Predictions
            logits = outputs.logits
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
            pred_masks = upsampled_logits.argmax(dim=1)
            
            # Metrics
            acc, mean_iou, _ = compute_metrics(pred_masks, labels)
            
            val_losses.append(loss.item())
            val_accs.append(acc)
            val_ious.append(mean_iou)
    
    return {
        'loss': np.mean(val_losses),
        'accuracy': np.mean(val_accs),
        'mean_iou': np.mean(val_ious)
    }


def main():
    print("="*60)
    print("SEGFORMER BOTTLE SEGMENTATION TRAINING")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n📦 Loading SegFormer model...")
    
    # Load pretrained SegFormer
    # Options: "nvidia/segformer-b0-finetuned-ade-512-512" (smallest, fastest)
    #          "nvidia/segformer-b1-finetuned-ade-512-512" (balanced)
    #          "nvidia/segformer-b2-finetuned-ade-512-512" (larger, slower)
    model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
    
    processor = SegformerImageProcessor.from_pretrained(model_name)
    
    # Load model with 2 classes: background (0), bottle (1)
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "background", 1: "bottle"},
        label2id={"background": 0, "bottle": 1},
        ignore_mismatched_sizes=True
    )
    
    model.to(device)
    
    print("✅ Model ready")
    print(f"Model: {model_name}")
    print(f"Num labels: {model.config.num_labels}")
    print(f"Label mapping: {model.config.id2label}")

    # Load datasets with train/val split
    # First, get all image IDs and create split
    print("\n📊 Creating train/val split...")
    
    # Load JSON to get image IDs
    with open(ANNOTATIONS_FILE, 'r') as f:
        coco_data = json.load(f)
    
    # Get actual files
    actual_files_dict = {}
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        for f in Path(IMAGES_DIR).glob(f'*{ext}'):
            actual_files_dict[f.name.lower()] = f.name
    
    # Match images
    img_id_to_info = {}
    for img in coco_data['images']:
        fname_lower = img['file_name'].lower()
        if fname_lower in actual_files_dict:
            img_id_to_info[img['id']] = img
    
    # Get images with annotations
    img_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id in img_id_to_info:
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)
    
    all_image_ids = list(img_to_anns.keys())
    
    # Create split
    train_ids, val_ids = train_test_split(
        all_image_ids, 
        train_size=TRAIN_SPLIT, 
        random_state=RANDOM_SEED
    )
    
    print(f"Total images: {len(all_image_ids)}")
    print(f"Train: {len(train_ids)} ({len(train_ids)/len(all_image_ids)*100:.1f}%)")
    print(f"Val: {len(val_ids)} ({len(val_ids)/len(all_image_ids)*100:.1f}%)")

    # Create datasets
    train_dataset = BottleSegmentationDataset(
        IMAGES_DIR, 
        ANNOTATIONS_FILE, 
        processor,
        image_size=IMAGE_SIZE,
        split='train',
        train_ids=train_ids,
        val_ids=val_ids
    )
    
    val_dataset = BottleSegmentationDataset(
        IMAGES_DIR, 
        ANNOTATIONS_FILE, 
        processor,
        image_size=IMAGE_SIZE,
        split='val',
        train_ids=train_ids,
        val_ids=val_ids
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler (optional but recommended)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=EPOCHS,
        eta_min=LEARNING_RATE * 0.1
    )

    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"\n🚀 Starting training...")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Batches per epoch: {len(train_loader)}")
    print(f"Training for {EPOCHS} epochs\n")

    best_iou = 0.0
    best_val_loss = float('inf')
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_iou': [],
        'val_loss': [],
        'val_acc': [],
        'val_iou': []
    }

    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_losses = []
        train_accs = []
        train_ious = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in pbar:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (optional but helps stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            # Metrics
            with torch.no_grad():
                logits = outputs.logits
                upsampled_logits = torch.nn.functional.interpolate(
                    logits,
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
                pred_masks = upsampled_logits.argmax(dim=1)
                
                acc, mean_iou, _ = compute_metrics(pred_masks, labels)
                train_accs.append(acc)
                train_ious.append(mean_iou)

            train_losses.append(loss.item())
            pbar.set_postfix(loss=loss.item(), acc=acc, iou=mean_iou)

        # Update learning rate
        scheduler.step()
        
        # Training metrics
        avg_train_loss = np.mean(train_losses)
        avg_train_acc = np.mean(train_accs)
        avg_train_iou = np.mean(train_ious)
        
        # Validation
        val_metrics = validate(model, val_loader, device)
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['train_iou'].append(avg_train_iou)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_iou'].append(val_metrics['mean_iou'])
        
        # Print summary
        print(f"\nEpoch {epoch+1}/{EPOCHS} Summary:")
        print(f"  Train - Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f}, IoU: {avg_train_iou:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, IoU: {val_metrics['mean_iou']:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model based on validation IoU
        if val_metrics['mean_iou'] > best_iou:
            best_iou = val_metrics['mean_iou']
            best_checkpoint = os.path.join(SAVE_DIR, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': best_iou,
                'history': history
            }, best_checkpoint)
            print(f"  💾 New best model! (Val IoU: {best_iou:.4f})")

        # Periodic checkpoint
        if (epoch + 1) % SAVE_EVERY == 0:
            checkpoint_path = os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, checkpoint_path)
            print(f"  💾 Checkpoint saved")

    print("\n" + "="*60)
    print("💾 Saving final model...")

    # Save final model in HuggingFace format
    hf_save_dir = os.path.join(SAVE_DIR, "huggingface_model")
    model.save_pretrained(hf_save_dir)
    processor.save_pretrained(hf_save_dir)
    
    # Save training history
    import pickle
    with open(os.path.join(SAVE_DIR, "training_history.pkl"), 'wb') as f:
        pickle.dump(history, f)

    print(f"✅ Model saved to: {hf_save_dir}")
    print(f"\nTraining Statistics:")
    print(f"  Best Val IoU: {best_iou:.4f}")
    print(f"  Final Train Loss: {avg_train_loss:.4f}")
    print(f"  Final Train IoU: {avg_train_iou:.4f}")
    print(f"  Final Val Loss: {val_metrics['loss']:.4f}")
    print(f"  Final Val IoU: {val_metrics['mean_iou']:.4f}")

    print("\nTraining complete 🎉")
    print("="*60)


if __name__ == "__main__":
    main()