#!/usr/bin/env python3
"""
GACNet Training Script

Clean, standalone training script for Graph Attention Convolution Network point cloud segmentation.
Usage: python train_gacnet.py --dataset-name your_dataset --epochs 100
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
# UTILITY FUNCTIONS
# =============================================================================

def farthest_point_sample(point, npoint):
    """
    Farthest point sampling for PointNet++
    Args:
        point: input points [B, N, C]
        npoint: number of samples
    Returns:
        centroids: sampled point indices [B, npoint]
    """
    device = point.device
    B, N, C = point.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = point[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((point - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Ball query for PointNet++
    Args:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points [B, N, 3]
        new_xyz: query points [B, S, 3]
    Returns:
        group_idx: grouped points index [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def square_distance(src, dst):
    """Calculate squared distance between points"""
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """Index points with given indices"""
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


# =============================================================================
# GRAPH ATTENTION MODULE
# =============================================================================

class GraphAttention(nn.Module):
    """Graph Attention Layer for GACNet"""
    
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, concat=True):
        super(GraphAttention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, h, adj):
        """
        Args:
            h: input features [B, N, in_features]
            adj: adjacency matrix [B, N, N]
        Returns:
            output: attention weighted features [B, N, out_features]
        """
        Wh = torch.matmul(h, self.W)  # [B, N, out_features]
        e = self._prepare_attentional_mechanism_input(Wh)
        
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
    
    def _prepare_attentional_mechanism_input(self, Wh):
        """Prepare attention mechanism input"""
        B, N, out_features = Wh.shape
        
        # Broadcast Wh for pairwise attention computation
        Wh1 = Wh.unsqueeze(2).repeat(1, 1, N, 1)  # [B, N, N, out_features]
        Wh2 = Wh.unsqueeze(1).repeat(1, N, 1, 1)  # [B, N, N, out_features]
        
        # Concatenate for attention computation
        concat_features = torch.cat([Wh1, Wh2], dim=3)  # [B, N, N, 2*out_features]
        
        # Apply attention weights
        e = torch.matmul(concat_features, self.a).squeeze(3)  # [B, N, N]
        return self.leakyrelu(e)


# =============================================================================
# POINTNET++ LAYERS WITH GRAPH ATTENTION
# =============================================================================

class PointNetSetAbstractionMsg(nn.Module):
    """PointNet++ Set Abstraction with multiple scales"""
    
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)
    
    def forward(self, xyz, points):
        """
        Args:
            xyz: input points position [B, C, N]
            points: input points features [B, D, N]
        Returns:
            new_xyz: sampled points position [B, C, S]
            new_points: sampled points features [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        
        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz
            
            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)
        
        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    """PointNet++ Feature Propagation"""
    
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
    
    def forward(self, xyz1, xyz2, points1, points2):
        """
        Args:
            xyz1: input points position [B, C, N]
            xyz2: sampled input points position [B, C, S]  
            points1: input points features [B, D, N]
            points2: input points features [B, D, S]
        Returns:
            new_points: upsampled points features [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        
        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape
        
        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
        
        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
        
        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        return new_points


# =============================================================================
# GACNET MODEL
# =============================================================================

class GACNet(nn.Module):
    """
    Graph Attention Convolution Network for point cloud segmentation
    Combines PointNet++ with Graph Attention mechanism
    """
    
    def __init__(self, num_classes=2, dropout=0.6, alpha=0.2):
        super(GACNet, self).__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        self.alpha = alpha
        
        # PointNet++ Set Abstraction layers
        self.sa1 = PointNetSetAbstractionMsg(
            1024, [0.1, 0.2, 0.4], [16, 32, 128], 0,
            [[32, 32, 64], [64, 64, 128], [64, 96, 128]]
        )
        self.sa2 = PointNetSetAbstractionMsg(
            256, [0.2, 0.4, 0.8], [32, 64, 128], 320,
            [[64, 64, 128], [128, 128, 256], [128, 128, 256]]
        )
        self.sa3 = PointNetSetAbstractionMsg(
            64, [0.4, 0.8, 1.6], [16, 32, 128], 640,
            [[128, 196, 256], [128, 196, 256], [128, 256, 256]]
        )
        self.sa4 = PointNetSetAbstractionMsg(
            16, [0.8, 1.2, 2.4], [16, 32, 128], 768,
            [[256, 256, 512], [256, 384, 512], [256, 384, 512]]
        )
        
        # Graph Attention layers
        self.gat1 = GraphAttention(320, 128, dropout, alpha)
        self.gat2 = GraphAttention(640, 256, dropout, alpha)
        self.gat3 = GraphAttention(768, 384, dropout, alpha)
        self.gat4 = GraphAttention(1536, 512, dropout, alpha)
        
        # Feature Propagation layers
        self.fp4 = PointNetFeaturePropagation(1536+512, [256, 256])
        self.fp3 = PointNetFeaturePropagation(768+256, [256, 256]) 
        self.fp2 = PointNetFeaturePropagation(640+256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128+6, [128, 128, 128])
        
        # Final classification layers
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
    
    def _build_adjacency_matrix(self, xyz, k=16):
        """Build adjacency matrix for graph attention"""
        B, C, N = xyz.shape
        xyz_t = xyz.transpose(2, 1)  # [B, N, 3]
        
        # Compute pairwise distances
        dist = square_distance(xyz_t, xyz_t)  # [B, N, N]
        
        # Get k nearest neighbors
        _, idx = torch.topk(-dist, k=k, dim=-1)  # [B, N, k]
        
        # Build adjacency matrix
        adj = torch.zeros(B, N, N, device=xyz.device)
        batch_idx = torch.arange(B).view(-1, 1, 1).expand(-1, N, k)
        node_idx = torch.arange(N).view(1, -1, 1).expand(B, -1, k)
        adj[batch_idx, node_idx, idx] = 1
        
        # Make symmetric
        adj = adj + adj.transpose(1, 2)
        adj = (adj > 0).float()
        
        return adj
    
    def forward(self, xyz):
        """
        Args:
            xyz: input point cloud [B, 3, N]
        Returns:
            output: segmentation logits [B, num_classes, N]
        """
        B, C, N = xyz.shape
        
        # Set Abstraction layers with Graph Attention
        l1_xyz, l1_points = self.sa1(xyz, None)
        adj1 = self._build_adjacency_matrix(l1_xyz)
        l1_points_gat = self.gat1(l1_points.transpose(2, 1), adj1).transpose(2, 1)
        
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points_gat)
        adj2 = self._build_adjacency_matrix(l2_xyz)
        l2_points_gat = self.gat2(l2_points.transpose(2, 1), adj2).transpose(2, 1)
        
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points_gat)
        adj3 = self._build_adjacency_matrix(l3_xyz)
        l3_points_gat = self.gat3(l3_points.transpose(2, 1), adj3).transpose(2, 1)
        
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points_gat)
        adj4 = self._build_adjacency_matrix(l4_xyz)
        l4_points_gat = self.gat4(l4_points.transpose(2, 1), adj4).transpose(2, 1)
        
        # Feature Propagation layers
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points_gat)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, torch.cat([xyz, xyz], 1), l1_points)
        
        # Final classification
        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        
        return x


# =============================================================================
# DATASET (Same as LDGCNN)
# =============================================================================

class SegmentationDataset(Dataset):
    """HDF5 dataset loader for GACNet point cloud segmentation"""
    
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
        """Normalize point cloud for GACNet"""
        # Center the point cloud
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        
        # Scale to unit sphere
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        if m > 0:
            pc = pc / m
        
        return pc
    
    def subsample_point_cloud(self, points, labels, target_size):
        """Subsample point cloud to target size"""
        if len(points) <= target_size:
            return points, labels
        
        # Stratified sampling to preserve class balance
        unique_labels = np.unique(labels)
        selected_indices = []
        
        points_per_class = target_size // len(unique_labels)
        remaining_points = target_size % len(unique_labels)
        
        for i, label in enumerate(unique_labels):
            label_indices = np.where(labels == label)[0]
            
            n_samples = points_per_class
            if i < remaining_points:
                n_samples += 1
            
            if len(label_indices) >= n_samples:
                sampled_indices = np.random.choice(label_indices, n_samples, replace=False)
            else:
                sampled_indices = label_indices
                if len(sampled_indices) < n_samples:
                    extra_needed = n_samples - len(sampled_indices)
                    extra_indices = np.random.choice(label_indices, extra_needed, replace=True)
                    sampled_indices = np.concatenate([sampled_indices, extra_indices])
            
            selected_indices.extend(sampled_indices)
        
        selected_indices = np.array(selected_indices)
        np.random.shuffle(selected_indices)
        selected_indices = selected_indices[:target_size]
        
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
            
            # Normalize coordinates
            coord = self.normalize_point_cloud(coord)
            
            # Subsample large point clouds
            if len(coord) > self.subsample_points:
                coord, label_data = self.subsample_point_cloud(coord, label_data, self.subsample_points)
            
            # Validate label range
            if label_data.min() < 0 or label_data.max() >= Config.NUM_CLASSES:
                label_data = np.clip(label_data, 0, Config.NUM_CLASSES-1)
            
            # Convert to tensors - GACNet expects [C, N] format
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
            
            # Return valid dummy data
            dummy_size = min(1024, self.subsample_points)
            dummy_coord = torch.zeros((3, dummy_size), dtype=torch.float32)
            dummy_labels = torch.zeros(dummy_size, dtype=torch.long)
            
            return dummy_coord, dummy_labels


def collate_fn(batch):
    """Collate function for GACNet data format"""
    coords, labels = [], []
    
    for coord, label in batch:
        coords.append(coord)
        labels.append(label)
    
    # Stack tensors
    coord = torch.stack(coords, dim=0)  # [B, 3, N]
    label = torch.stack(labels, dim=0)  # [B, N]
    
    return coord, label


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
    parser = argparse.ArgumentParser(description='Train GACNet for point cloud segmentation')
    parser.add_argument('--dataset-name', required=True, help='Dataset name (without _chunk_X)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', default='auto', help='Device (auto/cuda/cpu)')
    parser.add_argument('--dataset-path', default=None, help='Dataset path')
    parser.add_argument('--subsample-points', type=int, default=4096, help='Subsample to this many points')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate for GAT layers')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha parameter for GAT layers')
    
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
    
    print(f"GACNet Configuration:")
    print(f"  Dropout: {args.dropout}")
    print(f"  Alpha: {args.alpha}")
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
    model = GACNet(
        num_classes=Config.NUM_CLASSES, 
        dropout=args.dropout, 
        alpha=args.alpha
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    criterion = nn.NLLLoss().to(device)  # Use NLLLoss with log_softmax output
    
    # Training loop
    save_path = Config.RESULTS_PATH / "gacnet"
    save_path.mkdir(parents=True, exist_ok=True)
    
    best_iou = 0.0
    history = []
    
    print(f"\nStarting GACNet training for {args.epochs} epochs...")
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_iou = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        valid_loss, valid_acc, valid_iou = validate(
            model, valid_loader, criterion, device
        )
        
        scheduler.step()
        
        history.append({
            'Epoch': epoch,
            'Train_Loss': train_loss,
            'Train_Accuracy': train_acc,
            'Train_IoU': train_iou,
            'Valid_Loss': valid_loss,
            'Valid_Accuracy': valid_acc,
            'Valid_IoU': valid_iou,
            'Learning_Rate': optimizer.param_groups[0]['lr']
        })
        
        print(f"Epoch {epoch:3d}/{args.epochs}: "
              f"Train IoU: {train_iou:.4f}, Valid IoU: {valid_iou:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if valid_iou > best_iou:
            best_iou = valid_iou
            torch.save(model.state_dict(), save_path / f"gacnet_epoch_{epoch}_iou_{best_iou:.4f}.pth")
    
    # Save final results
    torch.save(model.state_dict(), save_path / "gacnet_final_model.pth")
    pd.DataFrame(history).to_csv(save_path / "gacnet_training_metrics.csv", index=False)
    
    print(f"\nGACNet training completed!")
    print(f"Best validation IoU: {best_iou:.4f}")
    print(f"Models saved to: {save_path}")


if __name__ == "__main__":
    main()