#!/usr/bin/env python3
"""
PointTransformerV3 Training Script

Clean, standalone training script for PTv3 point cloud segmentation.
Usage: python train_ptv3.py --dataset-name your_dataset --epochs 100
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import h5py
import os
import sys
import time
import argparse
from pathlib import Path

from config import Config


def setup_pointcept():
    """Import PointTransformerV3 from Pointcept"""
    sys.path.append(str(Config.POINTCEPT_PATH))
    try:
        from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3 as PTv3M1
        return PTv3M1, "m1"
    except ImportError:
        try:
            sys.path.append(str(Config.POINTCEPT_PATH / "pointcept/models/point_transformer_v3"))
            from point_transformer_v3m1_base import PointTransformerV3 as PTv3M1
            return PTv3M1, "m2"
        except ImportError as e:
            raise ImportError(f"Could not import PointTransformerV3. Make sure Pointcept is installed: {e}")


def check_flash_attention():
    """Check if flash attention is available"""
    try:
        import flash_attn
        return True
    except ImportError:
        return False


class SegmentationDataset(Dataset):
    """HDF5 dataset loader for point cloud segmentation"""
    
    def __init__(self, base_path, base_name, grid_size=0.02):
        self.base_path = Path(base_path)
        self.base_name = base_name
        self.grid_size = grid_size
        
        # Find all chunk files
        self.chunk_files = sorted(list(self.base_path.glob(f"{base_name}_*chunk_*")))
        if not self.chunk_files:
            raise FileNotFoundError(f"No chunk files found for {base_name} in {base_path}")
        
        # Load first chunk to get info
        with h5py.File(self.chunk_files[0], "r") as f:
            self.samples_per_chunk = len(f["points"])
            sample_shape = f["points"][0].shape
            self.num_features = sample_shape[1] if len(sample_shape) == 2 else 6
        
        self.total_samples = len(self.chunk_files) * self.samples_per_chunk
        print(f"Dataset: {len(self.chunk_files)} chunks, {self.total_samples} samples")
    
    def __len__(self):
        return self.total_samples
    
    def normalize_coordinates(self, coord):
        """Normalize coordinates for PTv3"""
        coord = coord.astype(np.float32)
        coord_center = np.mean(coord, axis=0, keepdims=True)
        coord = coord - coord_center
        coord_scale = np.max(np.abs(coord)) + 1e-6
        coord = coord / coord_scale * 2.0
        return coord
    
    def __getitem__(self, idx):
        chunk_id = idx // self.samples_per_chunk
        pos_in_chunk = idx % self.samples_per_chunk
        
        # Load from appropriate chunk
        with h5py.File(self.chunk_files[chunk_id], "r") as f:
            point_data = np.array(f["points"][pos_in_chunk], dtype=np.float32)
            label_data = np.array(f["labels"][pos_in_chunk], dtype=np.int64)
        
        # Extract coordinates and features
        coord = point_data[:, :3]
        coord = np.nan_to_num(coord, nan=0.0, posinf=1.0, neginf=-1.0)
        coord = self.normalize_coordinates(coord)
        
        # Handle features (RGB or use coordinates)
        if point_data.shape[1] > 3:
            feat = point_data[:, 3:].astype(np.float32)
            feat = np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0)
            if feat.max() > 1.0:
                feat = feat / 255.0  # Normalize RGB to [0,1]
        else:
            feat = coord.copy()
        
        # Create batch tensor
        batch = np.zeros(len(coord), dtype=np.int64)
        
        # Ensure valid labels
        label_data = np.clip(label_data, 0, Config.NUM_CLASSES - 1)
        
        return {
            'coord': torch.tensor(coord, dtype=torch.float32),
            'feat': torch.tensor(feat, dtype=torch.float32),
            'batch': torch.tensor(batch, dtype=torch.long),
            'grid_size': self.grid_size,
            'segment': torch.tensor(label_data, dtype=torch.long)
        }


def collate_fn(batch):
    """Collate function for PTv3 data format"""
    coords, feats, segments, batches = [], [], [], []
    
    for i, item in enumerate(batch):
        coords.append(item['coord'])
        feats.append(item['feat'])
        segments.append(item['segment'])
        batches.append(item['batch'] + i)
    
    coord = torch.cat(coords, dim=0)
    feat = torch.cat(feats, dim=0)
    segment = torch.cat(segments, dim=0)
    batch_tensor = torch.cat(batches, dim=0)
    
    # Create offset tensor
    offset_list = [0]
    for item in batch:
        offset_list.append(offset_list[-1] + len(item['coord']))
    offset = torch.tensor(offset_list[1:], dtype=torch.long)
    
    data_dict = {
        'coord': coord,
        'feat': feat,
        'batch': batch_tensor,
        'offset': offset,
        'grid_size': batch[0]['grid_size']
    }
    
    return data_dict, segment


class SegmentationHead(nn.Module):
    """Simple segmentation head for PTv3"""
    
    def __init__(self, in_channels, num_classes, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.BatchNorm1d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_channels // 2, num_classes)
        )
    
    def forward(self, feat):
        return self.classifier(feat)


class PTv3SegmentationModel(nn.Module):
    """Complete PTv3 segmentation model"""
    
    def __init__(self, backbone, seg_head):
        super().__init__()
        self.backbone = backbone
        self.seg_head = seg_head
    
    def forward(self, data_dict):
        point = self.backbone(data_dict)
        feat = point.feat if hasattr(point, 'feat') else point
        return self.seg_head(feat)


def create_model(feat_dim, flash_available, PTv3Model):
    """Create PTv3 model"""
    config = {
        'in_channels': feat_dim,
        'order': ("z", "z-trans"),
        'stride': (2, 2, 2, 2),
        'enc_depths': (2, 6, 2, 6, 2),
        'enc_channels': (32, 64, 128, 256, 512),
        'enc_num_head': (2, 4, 8, 16, 32),
        'enc_patch_size': (48, 48, 48, 48, 48),
        'dec_depths': (2, 2, 2, 6),
        'dec_channels': (64, 64, 128, 256),
        'dec_num_head': (4, 4, 8, 16),
        'dec_patch_size': (48, 48, 48, 48),
        'mlp_ratio': 4,
        'qkv_bias': True,
        'qk_scale': None,
        'attn_drop': 0.0,
        'proj_drop': 0.1,
        'drop_path': 0.3,
        'pre_norm': True,
        'shuffle_orders': True,
        'enable_rpe': not flash_available,
        'enable_flash': flash_available,
        'upcast_attention': not flash_available,
        'upcast_softmax': not flash_available,
        'cls_mode': False,
        'pdnorm_bn': False,
        'pdnorm_ln': False,
        'pdnorm_decouple': True,
        'pdnorm_adaptive': False,
        'pdnorm_affine': True,
        'pdnorm_conditions': ("Custom",),
    }
    
    return PTv3Model(**config)


def compute_iou(pred, target, num_classes=2):
    """Compute IoU for segmentation"""
    iou_list = []
    for i in range(num_classes):
        pred_i = (pred == i)
        target_i = (target == i)
        intersection = (pred_i & target_i).sum().float()
        union = (pred_i | target_i).sum().float()
        
        if union == 0:
            iou_list.append(float('nan'))
        else:
            iou_list.append((intersection / union).item())
    
    valid_ious = [iou for iou in iou_list if not np.isnan(iou)]
    return np.mean(valid_ious) if valid_ious else 0.0


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_iou = 0.0
    num_batches = 0
    
    for data_dict, targets in dataloader:
        data_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in data_dict.items()}
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        output = model(data_dict)
        loss = criterion(output, targets)
        
        if torch.isnan(loss):
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        pred_choice = torch.argmax(output, dim=1)
        correct = pred_choice.eq(targets).sum().item()
        iou = compute_iou(pred_choice.cpu(), targets.cpu(), Config.NUM_CLASSES)
        
        total_loss += loss.item()
        total_correct += correct
        total_samples += targets.size(0)
        total_iou += iou
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_accuracy = total_correct / total_samples if total_samples > 0 else 0
    avg_iou = total_iou / num_batches if num_batches > 0 else 0
    
    return avg_loss, avg_accuracy, avg_iou


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_iou = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for data_dict, targets in dataloader:
            data_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in data_dict.items()}
            targets = targets.to(device)
            
            output = model(data_dict)
            loss = criterion(output, targets)
            
            if torch.isnan(loss):
                continue
            
            pred_choice = torch.argmax(output, dim=1)
            correct = pred_choice.eq(targets).sum().item()
            iou = compute_iou(pred_choice.cpu(), targets.cpu(), Config.NUM_CLASSES)
            
            total_loss += loss.item()
            total_correct += correct
            total_samples += targets.size(0)
            total_iou += iou
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_accuracy = total_correct / total_samples if total_samples > 0 else 0
    avg_iou = total_iou / num_batches if num_batches > 0 else 0
    
    return avg_loss, avg_accuracy, avg_iou


def main():
    parser = argparse.ArgumentParser(description='Train PTv3 for point cloud segmentation')
    parser.add_argument('--dataset-name', required=True, help='Dataset name (without _chunk_X)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=6e-4, help='Learning rate')
    parser.add_argument('--device', default='auto', help='Device (auto/cuda/cpu)')
    parser.add_argument('--dataset-path', default=None, help='Dataset path')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Setup paths
    if args.dataset_path is None:
        args.dataset_path = Config.DATASET_PATH / "segmentation_dataset"
    
    Config.create_directories()
    
    # Setup model components
    PTv3Model, mode = setup_pointcept()
    flash_available = check_flash_attention()
    
    print(f"PTv3 mode: {mode}")
    print(f"Flash attention: {'available' if flash_available else 'not available'}")
    
    # Load dataset
    dataset = SegmentationDataset(args.dataset_path, args.dataset_name, grid_size=0.02)
    
    # Split dataset
    total_samples = len(dataset)
    train_size = int(0.8 * total_samples)
    valid_size = total_samples - train_size
    
    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset, [train_size, valid_size]
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    
    # Get feature dimension
    sample_batch, _ = next(iter(train_loader))
    feat_dim = sample_batch['feat'].shape[1]
    
    print(f"Dataset: {total_samples} samples ({train_size} train, {valid_size} valid)")
    print(f"Feature dimension: {feat_dim}")
    
    # Create model
    backbone = create_model(feat_dim, flash_available, PTv3Model).to(device)
    
    # Get output channels
    with torch.no_grad():
        sample_data_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                           for k, v in sample_batch.items()}
        sample_output = backbone(sample_data_dict)
        output_channels = sample_output.feat.shape[-1] if hasattr(sample_output, 'feat') else sample_output.shape[-1]
    
    seg_head = SegmentationHead(output_channels, Config.NUM_CLASSES).to(device)
    model = PTv3SegmentationModel(backbone, seg_head).to(device)
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    
    # Training loop
    save_path = Config.RESULTS_PATH / "ptv3"
    save_path.mkdir(parents=True, exist_ok=True)
    
    best_iou = 0.0
    history = []
    
    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_iou = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        valid_loss, valid_acc, valid_iou = validate(
            model, valid_loader, criterion, device
        )
        
        scheduler.step()
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_iou': train_iou,
            'valid_loss': valid_loss,
            'valid_acc': valid_acc,
            'valid_iou': valid_iou,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        print(f"Epoch {epoch:3d}/{args.epochs}: "
              f"Train IoU: {train_iou:.4f}, Valid IoU: {valid_iou:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if valid_iou > best_iou:
            best_iou = valid_iou
            torch.save(model.state_dict(), save_path / f"best_model_iou_{best_iou:.4f}.pth")
    
    # Save final results
    torch.save(model.state_dict(), save_path / "final_model.pth")
    pd.DataFrame(history).to_csv(save_path / "training_metrics.csv", index=False)
    
    print(f"\nTraining completed!")
    print(f"Best validation IoU: {best_iou:.4f}")
    print(f"Models saved to: {save_path}")


if __name__ == "__main__":
    main()