#!/usr/bin/env python3
"""
PointTransformerV2 Training Script

Clean, standalone training script for PTv2 point cloud segmentation.
Usage: python train_ptv2.py --dataset-name your_dataset --epochs 100 --mode m2
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
    """Import PointTransformerV2 from Pointcept"""
    sys.path.append(str(Config.POINTCEPT_PATH))
    try:
        from pointcept.models.point_transformer_v2.point_transformer_v2m1_origin import PointTransformerV2 as PTv2M1
        from pointcept.models.point_transformer_v2.point_transformer_v2m2_base import PointTransformerV2 as PTv2M2
        from pointcept.models.point_transformer_v2.point_transformer_v2m3_pdnorm import PointTransformerV2 as PTv2M3
        return {"m1": PTv2M1, "m2": PTv2M2, "m3": PTv2M3}, "package"
    except ImportError:
        try:
            sys.path.append(str(Config.POINTCEPT_PATH / "pointcept/models/point_transformer_v2"))
            from point_transformer_v2m1_origin import PointTransformerV2 as PTv2M1
            from point_transformer_v2m2_base import PointTransformerV2 as PTv2M2
            from point_transformer_v2m3_pdnorm import PointTransformerV2 as PTv2M3
            return {"m1": PTv2M1, "m2": PTv2M2, "m3": PTv2M3}, "direct"
        except ImportError as e:
            raise ImportError(f"Could not import PointTransformerV2. Make sure Pointcept is installed: {e}")


def check_flash_attention():
    """Check if flash attention is available"""
    try:
        import flash_attn
        return True
    except ImportError:
        return False


class SegmentationDataset(Dataset):
    """HDF5 dataset loader for point cloud segmentation"""
    
    def __init__(self, base_path, base_name, grid_sizes=(0.06, 0.12, 0.24, 0.48)):
        self.base_path = Path(base_path)
        self.base_name = base_name
        self.grid_sizes = grid_sizes
        
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
        """Normalize coordinates for PTv2 hierarchical pooling"""
        coord = coord.astype(np.float32)
        coord = np.nan_to_num(coord, nan=0.0, posinf=1.0, neginf=-1.0)
        coord_center = np.mean(coord, axis=0, keepdims=True)
        coord_centered = coord - coord_center
        coord_scale = np.max(np.abs(coord_centered)) + 1e-6
        coord_normalized = coord_centered / coord_scale * 3.0
        return coord_normalized
    
    def __getitem__(self, idx):
        chunk_id = idx // self.samples_per_chunk
        pos_in_chunk = idx % self.samples_per_chunk
        
        try:
            with h5py.File(self.chunk_files[chunk_id], "r") as f:
                point_data = np.array(f["points"][pos_in_chunk], dtype=np.float32)
                label_data = np.array(f["labels"][pos_in_chunk], dtype=np.int64)
        except Exception as e:
            # Return dummy sample
            dummy_points = np.random.randn(1024, self.num_features).astype(np.float32)
            dummy_labels = np.zeros(1024, dtype=np.int64)
            return {
                'coord': torch.tensor(dummy_points[:, :3], dtype=torch.float32),
                'feat': torch.tensor(dummy_points[:, :min(3, self.num_features)], dtype=torch.float32),
                'segment': torch.tensor(dummy_labels, dtype=torch.long)
            }
        
        # Extract coordinates and features
        coord = point_data[:, :3]
        coord = self.normalize_coordinates(coord)
        
        # Handle features
        if point_data.shape[1] > 3:
            feat = point_data[:, 3:].astype(np.float32)
            feat = np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0)
            if feat.max() > 1.0:
                feat = feat / 255.0
        else:
            feat = coord.copy()
        
        # Ensure valid labels
        label_data = np.clip(label_data, 0, Config.NUM_CLASSES - 1)
        
        return {
            'coord': torch.tensor(coord, dtype=torch.float32),
            'feat': torch.tensor(feat, dtype=torch.float32),
            'segment': torch.tensor(label_data, dtype=torch.long)
        }


def collate_fn(batch):
    """Collate function for PTv2 data format"""
    coords, feats, segments = [], [], []
    
    for item in batch:
        coords.append(item['coord'])
        feats.append(item['feat'])
        segments.append(item['segment'])
    
    coord = torch.cat(coords, dim=0)
    feat = torch.cat(feats, dim=0)
    segment = torch.cat(segments, dim=0)
    
    # Create offset tensor
    offset_list = [0]
    for item in batch:
        offset_list.append(offset_list[-1] + len(item['coord']))
    offset = torch.tensor(offset_list[1:], dtype=torch.long)
    
    data_dict = {
        'coord': coord,
        'feat': feat,
        'offset': offset
    }
    
    return data_dict, segment


class SegmentationHead(nn.Module):
    """Simple segmentation head for PTv2"""
    
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


class PTv2SegmentationModel(nn.Module):
    """Complete PTv2 segmentation model"""
    
    def __init__(self, backbone, seg_head):
        super().__init__()
        self.backbone = backbone
        self.seg_head = seg_head
    
    def forward(self, data_dict):
        point = self.backbone(data_dict)
        feat = point.feat if hasattr(point, 'feat') else point
        return self.seg_head(feat)


def create_ptv2_model(feat_dim, mode, flash_available, PTv2Models):
    """Create PTv2 model with mode-specific configuration"""
    PTv2Model = PTv2Models[mode]
    
    base_config = {
        'in_channels': feat_dim,
        'num_classes': Config.NUM_CLASSES,
    }
    
    if mode == "m1":
        config = {
            **base_config,
            'patch_embed_depth': 1,
            'patch_embed_channels': 32,
            'patch_embed_groups': 4,
            'patch_embed_neighbours': 8,
            'enc_depths': (1, 1, 2, 1),
            'enc_channels': (64, 128, 256, 384),
            'enc_groups': (8, 16, 32, 48),
            'enc_neighbours': (16, 16, 16, 16),
            'dec_depths': (1, 1, 1, 1),
            'dec_channels': (32, 64, 128, 256),
            'dec_groups': (4, 8, 16, 32),
            'dec_neighbours': (16, 16, 16, 16),
            'grid_sizes': (0.06, 0.12, 0.24, 0.48),
            'attn_qkv_bias': True,
            'pe_multiplier': False,
            'pe_bias': True,
            'attn_drop_rate': 0.0,
            'drop_path_rate': 0.0,
            'enable_checkpoint': False,
            'unpool_backend': "map",
        }
    elif mode == "m2":
        config = {
            **base_config,
            'patch_embed_depth': 1,
            'patch_embed_channels': 48,
            'patch_embed_groups': 6,
            'patch_embed_neighbours': 16,
            'enc_depths': (2, 2, 6, 2),
            'enc_channels': (96, 192, 384, 512),
            'enc_groups': (12, 24, 48, 64),
            'enc_neighbours': (16, 16, 16, 16),
            'dec_depths': (1, 1, 1, 2),
            'dec_channels': (48, 96, 192, 384),
            'dec_groups': (6, 12, 24, 48),
            'dec_neighbours': (16, 16, 16, 16),
            'grid_sizes': (0.06, 0.12, 0.24, 0.48),
            'attn_qkv_bias': True,
            'pe_multiplier': False,
            'pe_bias': True,
            'attn_drop_rate': 0.0,
            'drop_path_rate': 0.1,
            'enable_checkpoint': False,
            'unpool_backend': "map",
        }
    elif mode == "m3":
        config = {
            **base_config,
            'patch_embed_depth': 1,
            'patch_embed_channels': 48,
            'patch_embed_groups': 6,
            'patch_embed_neighbours': 16,
            'enc_depths': (2, 2, 6, 2),
            'enc_channels': (96, 192, 384, 512),
            'enc_groups': (12, 24, 48, 64),
            'enc_neighbours': (16, 16, 16, 16),
            'dec_depths': (1, 1, 1, 2),
            'dec_channels': (48, 96, 192, 384),
            'dec_groups': (6, 12, 24, 48),
            'dec_neighbours': (16, 16, 16, 16),
            'grid_sizes': (0.06, 0.12, 0.24, 0.48),
            'attn_qkv_bias': True,
            'pe_multiplier': False,
            'pe_bias': True,
            'attn_drop_rate': 0.0,
            'drop_path_rate': 0.1,
            'enable_checkpoint': False,
            'unpool_backend': "map",
            'context_channels': 128,
            'conditions': ("Custom",),
            'norm_decouple': True,
            'norm_adaptive': False,
            'norm_affine': True,
        }
    
    return PTv2Model(**config)


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


def train_epoch(model, dataloader, optimizer, criterion, device, mode):
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
        
        if mode == "m3":
            data_dict['condition'] = ["Custom"] * len(data_dict.get('offset', [1]))
            data_dict['context'] = None
        
        optimizer.zero_grad()
        
        output = model(data_dict)
        loss = criterion(output, targets)
        
        if torch.isnan(loss):
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
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


def validate(model, dataloader, criterion, device, mode):
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
            
            if mode == "m3":
                data_dict['condition'] = ["Custom"] * len(data_dict.get('offset', [1]))
                data_dict['context'] = None
            
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
    parser = argparse.ArgumentParser(description='Train PTv2 for point cloud segmentation')
    parser.add_argument('--dataset-name', required=True, help='Dataset name (without _chunk_X)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--mode', default='m2', choices=['m1', 'm2', 'm3'], 
                       help='PTv2 mode: m1=origin, m2=base, m3=pdnorm')
    parser.add_argument('--device', default='auto', help='Device (auto/cuda/cpu)')
    parser.add_argument('--dataset-path', default=None, help='Dataset path')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Setup paths
    if args.dataset_path is None:
        args.dataset_path = Config.DATASET_PATH / "segmentation_dataset"
    
    Config.create_directories()
    
    # Setup model components
    PTv2Models, import_mode = setup_pointcept()
    flash_available = check_flash_attention()
    
    # Load dataset
    dataset = SegmentationDataset(args.dataset_path, args.dataset_name)
    
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
        collate_fn=collate_fn, num_workers=0, drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, drop_last=True
    )
    
    # Get feature dimension
    sample_batch, _ = next(iter(train_loader))
    feat_dim = sample_batch['feat'].shape[1]
    
    # Create model
    try:
        backbone = create_ptv2_model(feat_dim, args.mode, flash_available, PTv2Models).to(device)
        
        # Get output channels
        with torch.no_grad():
            sample_data_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                               for k, v in sample_batch.items()}
            
            if args.mode == "m3":
                sample_data_dict['condition'] = ["Custom"]
                sample_data_dict['context'] = None
            
            sample_output = backbone(sample_data_dict)
            if hasattr(sample_output, 'feat'):
                output_channels = sample_output.feat.shape[-1]
            elif isinstance(sample_output, dict) and 'feat' in sample_output:
                output_channels = sample_output['feat'].shape[-1]
            else:
                output_channels = sample_output.shape[-1]
        
    except Exception as e:
        # Minimal fallback configuration
        try:
            PTv2Model = PTv2Models[args.mode]
            backbone = PTv2Model(in_channels=feat_dim).to(device)
            output_channels = 512
        except Exception as e2:
            return
    
    seg_head = SegmentationHead(output_channels, Config.NUM_CLASSES).to(device)
    model = PTv2SegmentationModel(backbone, seg_head).to(device)
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    
    # Training loop
    save_path = Config.RESULTS_PATH / f"ptv2_{args.mode}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    best_iou = 0.0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_iou = train_epoch(
            model, train_loader, optimizer, criterion, device, args.mode
        )
        valid_loss, valid_acc, valid_iou = validate(
            model, valid_loader, criterion, device, args.mode
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
            torch.save(model.state_dict(), 
                      save_path / f"best_model_{args.mode}_iou_{best_iou:.4f}.pth")
    
    # Save final results
    torch.save(model.state_dict(), save_path / f"final_model_{args.mode}.pth")
    pd.DataFrame(history).to_csv(save_path / f"training_metrics_{args.mode}.csv", index=False)


if __name__ == "__main__":
    main()