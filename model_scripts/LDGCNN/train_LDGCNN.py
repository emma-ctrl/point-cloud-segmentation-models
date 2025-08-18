#!/usr/bin/env python3
"""
LDGCNN Training Script

Clean, standalone training script for Linked Dynamic Graph CNN point cloud segmentation.
Usage: python train_ldgcnn.py --dataset-name your_dataset --epochs 100
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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


# =============================================================================
# MEMORY-EFFICIENT K-NN AND GRAPH FUNCTIONS
# =============================================================================

def knn_memory_efficient(x, k):
    """
    Memory-efficient k-NN computation for large point clouds
    Args:
        x: input points, shape [B, N, C]
        k: number of neighbors
    Returns:
        idx: indices of k nearest neighbors, shape [B, N, k]
    """
    batch_size, num_points, num_dims = x.shape
    
    # Ensure k doesn't exceed available points
    if num_points <= 1:
        dummy_idx = torch.zeros((batch_size, num_points, 1), dtype=torch.long, device=x.device)
        return dummy_idx
    
    effective_k = min(k, num_points - 1)
    effective_k = max(1, effective_k)
    
    try:
        device = x.device
        
        # Memory-efficient approach: Process in chunks to avoid OOM
        chunk_size = min(512, num_points)
        all_indices = []
        
        for start_idx in range(0, num_points, chunk_size):
            end_idx = min(start_idx + chunk_size, num_points)
            chunk_points = x[:, start_idx:end_idx, :]
            
            # Compute squared distances efficiently
            chunk_norms = torch.sum(chunk_points ** 2, dim=2, keepdim=True)
            all_norms = torch.sum(x ** 2, dim=2, keepdim=True)
            dot_products = torch.einsum('bic,bjc->bij', chunk_points, x)
            squared_distances = chunk_norms + all_norms.transpose(1, 2) - 2 * dot_products
            
            # Convert to negative for topk (we want smallest distances)
            pairwise_distance = -squared_distances
            
            # Mask diagonal elements for self-comparison
            for i in range(start_idx, end_idx):
                if i < num_points:
                    pairwise_distance[:, i - start_idx, i] = float('-inf')
            
            # Get top-k neighbors for this chunk
            chunk_indices = pairwise_distance.topk(k=effective_k, dim=-1)[1]
            all_indices.append(chunk_indices)
            
            # Clear intermediate tensors
            del chunk_norms, dot_products, squared_distances, pairwise_distance, chunk_indices
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate all chunk results
        idx = torch.cat(all_indices, dim=1)
        return idx
        
    except Exception as e:
        print(f"Error in k-NN computation: {e}")
        
        # Fallback: return simple sequential indices
        device = x.device
        fallback_idx = torch.zeros((batch_size, num_points, effective_k), dtype=torch.long, device=device)
        
        for b in range(batch_size):
            for i in range(num_points):
                indices = torch.arange(num_points, device=device)
                mask = indices != i
                available_indices = indices[mask]
                
                if len(available_indices) >= effective_k:
                    selected = available_indices[:effective_k]
                else:
                    repeats = (effective_k + len(available_indices) - 1) // len(available_indices)
                    selected = available_indices.repeat(repeats)[:effective_k]
                
                fallback_idx[b, i, :] = selected
        
        return fallback_idx


def get_graph_feature_memory_efficient(x, k=10, idx=None, dim9=False):
    """
    Memory-efficient graph feature extraction with dynamic k adjustment
    Args:
        x: input features, shape [B, C, N]
        k: number of neighbors
        idx: neighbor indices (optional)
        dim9: whether to use 9-dimensional features
    Returns:
        feature: edge features, shape [B, 2*C, N, effective_k]
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    # Ensure k doesn't exceed available points
    effective_k = min(k, num_points - 1)
    effective_k = max(1, effective_k)
    
    if idx is None:
        # Use memory-efficient k-NN
        x_for_knn = x.transpose(2, 1).contiguous()
        try:
            if dim9 == False:
                idx = knn_memory_efficient(x_for_knn, k=effective_k)
            else:
                idx = knn_memory_efficient(x_for_knn[:, :, 6:], k=effective_k)
        except Exception as knn_error:
            print(f"Error in knn_memory_efficient: {knn_error}")
            raise knn_error
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    
    _, num_dims, _ = x.size()
    
    x = x.transpose(2, 1).contiguous()
    
    # Memory-efficient feature gathering
    try:
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, effective_k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, effective_k, 1)
        
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"Reducing k from {effective_k} to {effective_k//2} due to memory constraints")
            return get_graph_feature_memory_efficient(
                x.transpose(2, 1).contiguous(), k=effective_k//2, idx=None, dim9=dim9
            )
        else:
            print(f"Runtime error in feature gathering: {e}")
            raise e
    
    return feature


# =============================================================================
# LDGCNN MODEL COMPONENTS
# =============================================================================

class EdgeConv(nn.Module):
    """Memory-efficient Edge Convolution layer for dynamic graph CNN with adaptive k"""
    
    def __init__(self, in_channels, out_channels, k=10):
        super(EdgeConv, self).__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels*2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
    def forward(self, x):
        """
        Args:
            x: input features [B, C, N]
        Returns:
            x: output features [B, out_channels, N]
        """
        batch_size, channels, num_points = x.shape
        
        # Ensure we have enough points for k-NN
        if num_points < self.k:
            effective_k = max(1, num_points - 1)
        else:
            effective_k = self.k
        
        # Get graph features with adaptive k
        try:
            graph_features = get_graph_feature_memory_efficient(x, k=effective_k)
        except Exception as e:
            print(f"Error in get_graph_feature_memory_efficient: {e}")
            raise e
        
        # Apply convolution
        x = self.conv(graph_features)
        
        # Max pooling over neighbors
        x = x.max(dim=-1, keepdim=False)[0]
        return x


class LDGCNN(nn.Module):
    """
    Memory-optimized Linked Dynamic Graph CNN for point cloud segmentation
    Based on the paper: "Linked Dynamic Graph CNN: Learning on Point Cloud via Linking Hierarchical Features"
    """
    
    def __init__(self, args, output_channels=2):
        super(LDGCNN, self).__init__()
        self.args = args
        self.k = args.k
        
        # Feature extraction layers with hierarchical linking
        self.conv1 = EdgeConv(3, 32, k=self.k)
        self.conv2 = EdgeConv(32, 32, k=self.k)
        self.conv3 = EdgeConv(32, 64, k=self.k)
        self.conv4 = EdgeConv(64, 128, k=self.k)
        self.conv5 = EdgeConv(128, 256, k=self.k)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(256)
        
        # Segmentation head with feature linking
        # Total features: 32 + 32 + 64 + 128 + 256 = 512
        self.conv6 = nn.Conv1d(512, 256, 1)
        self.conv7 = nn.Conv1d(256, 128, 1)
        self.conv8 = nn.Conv1d(128, 64, 1)
        self.conv9 = nn.Conv1d(64, output_channels, 1)
        
        self.bn6 = nn.BatchNorm1d(256)
        self.bn7 = nn.BatchNorm1d(128)
        self.bn8 = nn.BatchNorm1d(64)
        
        self.dropout = nn.Dropout(0.3)
        
        # Enable gradient checkpointing for memory efficiency
        self.use_checkpoint = True
        
    def forward(self, x):
        """
        Args:
            x: input point cloud [B, C, N] where C=3 for XYZ
        Returns:
            x: segmentation logits [B, num_classes, N]
        """
        # Memory management
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Hierarchical feature extraction with linking and gradient checkpointing
        if self.use_checkpoint and self.training:
            # Use gradient checkpointing to save memory during training
            x1 = torch.utils.checkpoint.checkpoint(
                lambda inp: F.relu(self.bn1(self.conv1(inp))), x, use_reentrant=False
            )
            x2 = torch.utils.checkpoint.checkpoint(
                lambda inp: F.relu(self.bn2(self.conv2(inp))), x1, use_reentrant=False
            )
            x3 = torch.utils.checkpoint.checkpoint(
                lambda inp: F.relu(self.bn3(self.conv3(inp))), x2, use_reentrant=False
            )
            x4 = torch.utils.checkpoint.checkpoint(
                lambda inp: F.relu(self.bn4(self.conv4(inp))), x3, use_reentrant=False
            )
            x5 = torch.utils.checkpoint.checkpoint(
                lambda inp: F.relu(self.bn5(self.conv5(inp))), x4, use_reentrant=False
            )
        else:
            # Normal forward pass for inference
            x1 = F.relu(self.bn1(self.conv1(x)))
            x2 = F.relu(self.bn2(self.conv2(x1)))
            x3 = F.relu(self.bn3(self.conv3(x2)))
            x4 = F.relu(self.bn4(self.conv4(x3)))
            x5 = F.relu(self.bn5(self.conv5(x4)))
        
        # Link hierarchical features (key innovation of LDGCNN)
        x_linked = torch.cat((x1, x2, x3, x4, x5), dim=1)
        
        # Clear intermediate features to save memory
        if torch.cuda.is_available():
            del x1, x2, x3, x4, x5
            torch.cuda.empty_cache()
        
        # Segmentation head
        x = F.relu(self.bn6(self.conv6(x_linked)))
        x = self.dropout(x)
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.dropout(x)
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.conv9(x)
        
        return x


# =============================================================================
# DATASET
# =============================================================================

class SegmentationDataset(Dataset):
    """HDF5 dataset loader for LDGCNN point cloud segmentation"""
    
    def __init__(self, base_path, base_name, subsample_points=4096, grid_size=0.02):
        self.base_path = Path(base_path)
        self.base_name = base_name
        self.subsample_points = subsample_points
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
    
    def normalize_point_cloud(self, pc):
        """Normalize point cloud for LDGCNN"""
        # Center the point cloud
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        
        # Scale to unit sphere
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        if m > 0:
            pc = pc / m
        
        return pc
    
    def subsample_point_cloud(self, points, labels, target_size):
        """Subsample point cloud to target size for memory efficiency"""
        if len(points) <= target_size:
            return points, labels
        
        # Ensure minimum number of points for k-NN
        min_points_needed = max(10 + 1, 32)
        actual_target = max(target_size, min_points_needed)
        
        # Stratified sampling to preserve class balance
        unique_labels = np.unique(labels)
        selected_indices = []
        
        points_per_class = actual_target // len(unique_labels)
        remaining_points = actual_target % len(unique_labels)
        
        for i, label in enumerate(unique_labels):
            label_indices = np.where(labels == label)[0]
            
            n_samples = points_per_class
            if i < remaining_points:
                n_samples += 1
            
            # Ensure we have at least some points from each class
            n_samples = max(n_samples, min_points_needed // len(unique_labels))
            
            if len(label_indices) >= n_samples:
                sampled_indices = np.random.choice(label_indices, n_samples, replace=False)
            else:
                sampled_indices = label_indices
                if len(sampled_indices) < n_samples:
                    extra_needed = n_samples - len(sampled_indices)
                    extra_indices = np.random.choice(label_indices, extra_needed, replace=True)
                    sampled_indices = np.concatenate([sampled_indices, extra_indices])
            
            selected_indices.extend(sampled_indices)
        
        # Shuffle and finalize selection
        selected_indices = np.array(selected_indices)
        np.random.shuffle(selected_indices)
        
        if len(selected_indices) < actual_target:
            remaining = actual_target - len(selected_indices)
            all_indices = np.arange(len(points))
            extra_indices = np.random.choice(all_indices, remaining, replace=True)
            selected_indices = np.concatenate([selected_indices, extra_indices])
        
        selected_indices = selected_indices[:actual_target]
        
        return points[selected_indices], labels[selected_indices]
    
    def __getitem__(self, idx):
        try:
            if idx >= self.total_samples:
                raise IndexError(f"Index {idx} out of bounds")
            
            # Calculate which chunk and position
            chunk_id = idx // self.samples_per_chunk
            pos_in_chunk = idx % self.samples_per_chunk
            
            # Load data from chunk
            chunk_path = self.chunk_files[chunk_id]
            
            with h5py.File(chunk_path, "r") as f:
                if pos_in_chunk >= len(f["points"]):
                    pos_in_chunk = pos_in_chunk % len(f["points"])
                
                point_data = np.array(f["points"][pos_in_chunk], dtype=np.float32)
                label_data = np.array(f["labels"][pos_in_chunk], dtype=np.int64)
            
            # Extract coordinates (XYZ)
            if point_data.shape[1] < 3:
                coord = np.zeros((point_data.shape[0], 3), dtype=np.float32)
                coord[:, :point_data.shape[1]] = point_data
            else:
                coord = point_data[:, :3].astype(np.float32)
            
            # Clean invalid data
            if np.any(np.isnan(coord)) or np.any(np.isinf(coord)):
                coord = np.nan_to_num(coord, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Normalize coordinates for LDGCNN
            coord = self.normalize_point_cloud(coord)
            
            # Memory optimization: Subsample large point clouds
            if len(coord) > self.subsample_points:
                coord, label_data = self.subsample_point_cloud(coord, label_data, self.subsample_points)
            
            # Validate label range
            if label_data.min() < 0 or label_data.max() >= Config.NUM_CLASSES:
                label_data = np.clip(label_data, 0, Config.NUM_CLASSES-1)
            
            # Convert to tensors - LDGCNN expects [C, N] format
            coord_tensor = torch.tensor(coord.T, dtype=torch.float32)  # [3, N]
            label_tensor = torch.tensor(label_data, dtype=torch.long)  # [N]
            
            # Final validation
            if coord_tensor.shape[0] != 3:
                dummy_size = min(1024, self.subsample_points)
                dummy_coord = torch.zeros((3, dummy_size), dtype=torch.float32)
                dummy_labels = torch.zeros(dummy_size, dtype=torch.long)
                return dummy_coord, dummy_labels
            
            if coord_tensor.shape[1] != label_tensor.shape[0]:
                min_len = min(coord_tensor.shape[1], label_tensor.shape[0])
                coord_tensor = coord_tensor[:, :min_len]
                label_tensor = label_tensor[:min_len]
            
            return coord_tensor, label_tensor
            
        except Exception as e:
            print(f"Error in __getitem__ for index {idx}: {e}")
            
            # Return valid dummy data to prevent worker crashes
            dummy_size = min(1024, self.subsample_points)
            dummy_coord = torch.zeros((3, dummy_size), dtype=torch.float32)
            dummy_labels = torch.zeros(dummy_size, dtype=torch.long)
            
            return dummy_coord, dummy_labels


def collate_fn(batch):
    """Collate function for LDGCNN data format"""
    coords, labels = [], []
    
    for coord, label in batch:
        coords.append(coord)
        labels.append(label)
    
    # Stack tensors
    coord = torch.stack(coords, dim=0)  # [B, 3, N]
    label = torch.stack(labels, dim=0)  # [B, N]
    
    return coord, label


# =============================================================================
# CONFIGURATION
# =============================================================================

class LDGCNNArgs:
    """Configuration arguments for LDGCNN"""
    def __init__(self):
        self.k = 10  # Number of nearest neighbors
        self.emb_dims = 1024  # Embedding dimensions
        self.dropout = 0.3  # Dropout rate


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

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
    
    for data, labels in dataloader:
        try:
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(data)  # [B, num_classes, N]
            
            # Reshape for loss computation
            logits = logits.permute(0, 2, 1).contiguous()  # [B, N, num_classes]
            logits = logits.view(-1, Config.NUM_CLASSES)  # [B*N, num_classes]
            labels = labels.view(-1)  # [B*N]
            
            loss = criterion(logits, labels)
            
            if torch.isnan(loss):
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Metrics
            pred_choice = torch.argmax(logits, dim=1)
            correct = pred_choice.eq(labels).sum().item()
            iou = compute_iou(pred_choice.cpu(), labels.cpu(), Config.NUM_CLASSES)
            
            total_loss += loss.item()
            total_correct += correct
            total_samples += labels.size(0)
            total_iou += iou
            num_batches += 1
            
            # Clear memory
            del logits, pred_choice, data, labels
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error in training batch: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    
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
        for data, labels in dataloader:
            try:
                data, labels = data.to(device), labels.to(device)
                
                logits = model(data)
                
                # Reshape for loss computation
                logits = logits.permute(0, 2, 1).contiguous()
                logits = logits.view(-1, Config.NUM_CLASSES)
                labels = labels.view(-1)
                
                loss = criterion(logits, labels)
                
                if torch.isnan(loss):
                    continue
                
                pred_choice = torch.argmax(logits, dim=1)
                correct = pred_choice.eq(labels).sum().item()
                iou = compute_iou(pred_choice.cpu(), labels.cpu(), Config.NUM_CLASSES)
                
                total_loss += loss.item()
                total_correct += correct
                total_samples += labels.size(0)
                total_iou += iou
                num_batches += 1
                
                # Clear memory
                del logits, pred_choice, data, labels
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error in validation batch: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_accuracy = total_correct / total_samples if total_samples > 0 else 0
    avg_iou = total_iou / num_batches if num_batches > 0 else 0
    
    return avg_loss, avg_accuracy, avg_iou


def main():
    parser = argparse.ArgumentParser(description='Train LDGCNN for point cloud segmentation')
    parser.add_argument('--dataset-name', required=True, help='Dataset name (without _chunk_X)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', default='auto', help='Device (auto/cuda/cpu)')
    parser.add_argument('--dataset-path', default=None, help='Dataset path')
    parser.add_argument('--subsample-points', type=int, default=4096, help='Subsample to this many points')
    parser.add_argument('--k-neighbors', type=int, default=10, help='Number of k neighbors for graph construction')
    
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
    
    print(f"LDGCNN Configuration:")
    print(f"  K-neighbors: {args.k_neighbors}")
    print(f"  Subsample points: {args.subsample_points}")
    print(f"  Output classes: {Config.NUM_CLASSES}")
    
    # Load dataset
    dataset = SegmentationDataset(
        args.dataset_path, 
        args.dataset_name, 
        subsample_points=args.subsample_points
    )
    
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
    
    print(f"Dataset: {total_samples} samples ({train_size} train, {valid_size} valid)")
    
    # Create model
    ldgcnn_args = LDGCNNArgs()
    ldgcnn_args.k = args.k_neighbors
    
    model = LDGCNN(ldgcnn_args, output_channels=Config.NUM_CLASSES).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Training loop
    save_path = Config.RESULTS_PATH / "ldgcnn"
    save_path.mkdir(parents=True, exist_ok=True)
    
    best_iou = 0.0
    history = []
    
    print(f"\nStarting LDGCNN training for {args.epochs} epochs...")
    
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
    
    print(f"\nLDGCNN training completed!")
    print(f"Best validation IoU: {best_iou:.4f}")
    print(f"Models saved to: {save_path}")


if __name__ == "__main__":
    main()