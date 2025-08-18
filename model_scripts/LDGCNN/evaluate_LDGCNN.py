#!/usr/bin/env python3
"""
LDGCNN Evaluation Script

Evaluation script for LDGCNN point cloud segmentation models.
Processes real-world scenes and generates segmentation results.

Usage:
    python evaluate_ldgcnn.py --model-path path/to/model.pth --scene-dir path/to/scenes --output-dir results/
"""

import open3d as o3d
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import os
import sys
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree


class LDGCNNSceneProcessor:
    """Point cloud scene processor for LDGCNN model inference"""
    
    def __init__(self, model, device, max_points_per_chunk=20480, overlap_ratio=0.3, 
                 grid_size=0.2, adaptive_chunking=False):
        self.model = model
        self.device = device
        self.max_points_per_chunk = max_points_per_chunk
        self.overlap_ratio = overlap_ratio
        self.grid_size = grid_size
        self.adaptive_chunking = adaptive_chunking
        self.model.eval()
        self.current_confidences = None
        
    def correct_scene_orientation(self, points):
        """Apply orientation correction (flip Y-axis)"""
        points_corrected = points.copy()
        points_corrected[:, 1] = -points_corrected[:, 1]
        return points_corrected
        
    def load_pcd_scene(self, pcd_path):
        """Load PCD file and extract points and colors"""
        try:
            cloud = o3d.io.read_point_cloud(pcd_path)
            points = np.asarray(cloud.points)
            colors = np.asarray(cloud.colors)
            
            if len(colors) == 0 or colors.shape[0] == 0:
                return self._manual_load_pcd(pcd_path)
            
            if len(points) == 0:
                return np.array([]), np.array([])
            
            # Clean NaN values
            nan_points = np.isnan(points).sum()
            if nan_points > 0:
                valid_mask = ~np.isnan(points).any(axis=1)
                points = points[valid_mask]
                colors = colors[valid_mask]
            
            return points, colors
            
        except Exception as e:
            return self._manual_load_pcd(pcd_path)
    
    def _manual_load_pcd(self, pcd_path):
        """Manual PCD parsing as fallback"""
        try:
            with open(pcd_path, 'r') as f:
                lines = f.readlines()
            
            data_start = 0
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith('DATA ascii'):
                    data_start = i + 1
                    break
            
            points = []
            colors = []
            
            for line in lines[data_start:]:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    values = line.split()
                    
                    if len(values) >= 6:
                        x, y, z = float(values[0]), float(values[1]), float(values[2])
                        r, g, b = int(values[3]), int(values[4]), int(values[5])
                        points.append([x, y, z])
                        colors.append([r/255.0, g/255.0, b/255.0])
                        
                    elif len(values) >= 4:
                        x, y, z = float(values[0]), float(values[1]), float(values[2])
                        rgb_packed = int(values[3])
                        
                        r = (rgb_packed >> 16) & 0xFF
                        g = (rgb_packed >> 8) & 0xFF
                        b = rgb_packed & 0xFF
                        
                        points.append([x, y, z])
                        colors.append([r/255.0, g/255.0, b/255.0])
                        
                except Exception:
                    continue
            
            points = np.array(points, dtype=np.float32)
            colors = np.array(colors, dtype=np.float32)
            
            return points, colors
            
        except Exception:
            return np.array([]), np.array([])
    
    def compute_scene_bounds(self, points):
        """Compute 2D bounding box of the scene"""
        min_bounds = np.min(points[:, :2], axis=0)
        max_bounds = np.max(points[:, :2], axis=0)
        scene_size = max_bounds - min_bounds
        return min_bounds, max_bounds, scene_size
    
    def generate_chunk_grid(self, points):
        """Generate 2D grid of overlapping chunks"""
        min_bounds, max_bounds, scene_size = self.compute_scene_bounds(points)
        
        step_size = self.grid_size * (1 - self.overlap_ratio)
        step_size = max(step_size, 0.01)
        
        chunks = []
        
        x_steps = int(np.ceil((max_bounds[0] - min_bounds[0]) / step_size)) + 1
        y_steps = int(np.ceil((max_bounds[1] - min_bounds[1]) / step_size)) + 1
        
        x_positions = np.linspace(min_bounds[0], max_bounds[0], x_steps)
        y_positions = np.linspace(min_bounds[1], max_bounds[1], y_steps)
        
        for x in x_positions:
            for y in y_positions:
                chunk_min = np.array([x, y])
                chunk_max = chunk_min + self.grid_size
                chunks.append((chunk_min, chunk_max))
        
        return chunks
    
    def extract_chunk_points(self, points, colors, chunk_min, chunk_max):
        """Extract points within a 2D chunk"""
        mask = ((points[:, 0] >= chunk_min[0]) & (points[:, 0] < chunk_max[0]) &
                (points[:, 1] >= chunk_min[1]) & (points[:, 1] < chunk_max[1]))
        chunk_points = points[mask]
        chunk_colors = colors[mask]
        point_indices = np.where(mask)[0]
        
        return chunk_points, chunk_colors, point_indices
    
    def normalize_coordinates(self, coord):
        """Normalize coordinates for LDGCNN"""
        coord = coord.astype(np.float32)
        coord_center = np.mean(coord, axis=0, keepdims=True)
        coord = coord - coord_center
        coord_scale = np.max(np.abs(coord)) + 1e-6
        coord = coord / coord_scale
        return coord

    def prepare_ldgcnn_data(self, chunk_points, chunk_colors):
        """Prepare chunk data for LDGCNN model input"""
        num_points = chunk_points.shape[0]
        
        if num_points == 0:
            return None
        
        if len(chunk_colors) == 0 or chunk_colors.shape[0] == 0:
            chunk_colors = np.ones((num_points, 3), dtype=np.float32) * 0.5
        
        # Sample or pad to target number of points
        if num_points >= self.max_points_per_chunk:
            indices = np.random.choice(num_points, self.max_points_per_chunk, replace=False)
            sampled_points = chunk_points[indices]
            sampled_colors = chunk_colors[indices]
            sampling_indices = indices
        else:
            # Pad by repeating points
            pad_amount = self.max_points_per_chunk - num_points
            pad_indices = np.random.choice(num_points, pad_amount, replace=True)
            sampled_points = np.concatenate([chunk_points, chunk_points[pad_indices]], axis=0)
            sampled_colors = np.concatenate([chunk_colors, chunk_colors[pad_indices]], axis=0)
            sampling_indices = np.arange(num_points)
            
        coord = self.normalize_coordinates(sampled_points)
        
        # Prepare features
        feat = sampled_colors.astype(np.float32)
        if feat.max() > 1.0:
            feat = feat / 255.0
        
        # Convert to tensor format [B, C, N]
        coord_tensor = torch.tensor(coord.T, dtype=torch.float32, device=self.device)
        coord_tensor = coord_tensor.unsqueeze(0)  # [1, 3, N]
        
        return coord_tensor, sampling_indices
    
    def run_inference_on_chunk(self, chunk_points, chunk_colors):
        """Run LDGCNN model inference on a single chunk"""
        result = self.prepare_ldgcnn_data(chunk_points, chunk_colors)
        
        if result is None:
            return None, None
            
        data_tensor, sampling_indices = result
        
        with torch.no_grad():
            logits = self.model(data_tensor)
            
            # Handle different output formats
            if len(logits.shape) == 3 and logits.shape[1] == 2:
                probabilities = torch.softmax(logits, dim=1)
                predicted_labels = torch.argmax(logits, dim=1)
                probabilities = probabilities.squeeze(0).permute(1, 0).cpu().numpy()
                predicted_labels = predicted_labels.squeeze(0).cpu().numpy()
            else:
                probabilities = torch.softmax(logits, dim=-1)
                predicted_labels = torch.argmax(logits, dim=-1)
                probabilities = probabilities.squeeze(0).cpu().numpy()
                predicted_labels = predicted_labels.squeeze(0).cpu().numpy()
        
        # Only return predictions for original points (not padded)
        if sampling_indices is not None:
            valid_predictions = predicted_labels[:len(sampling_indices)]
            valid_probabilities = probabilities[:len(sampling_indices)]
        else:
            valid_predictions = predicted_labels
            valid_probabilities = probabilities
        
        return valid_predictions, valid_probabilities
    
    def aggregate_overlapping_predictions(self, all_predictions, total_points):
        """Aggregate predictions from overlapping chunks using voting"""
        vote_counts = np.zeros((total_points, 2))
        probability_sums = np.zeros((total_points, 2))
        point_seen_count = np.zeros(total_points)
        
        for chunk_predictions, chunk_probabilities, chunk_indices in all_predictions:
            for i, global_idx in enumerate(chunk_indices):
                if i < len(chunk_predictions):
                    predicted_class = chunk_predictions[i]
                    probs = chunk_probabilities[i]
                    
                    vote_counts[global_idx, predicted_class] += 1
                    probability_sums[global_idx] += probs
                    point_seen_count[global_idx] += 1
        
        final_predictions = np.zeros(total_points, dtype=int)
        final_confidences = np.zeros(total_points)
        
        for i in range(total_points):
            if point_seen_count[i] > 0:
                final_predictions[i] = np.argmax(vote_counts[i])
                avg_probs = probability_sums[i] / point_seen_count[i]
                final_confidences[i] = np.max(avg_probs)
            else:
                final_predictions[i] = 0
                final_confidences[i] = 0.0
        
        covered_points = np.sum(point_seen_count > 0)
        coverage_percentage = covered_points / total_points * 100
        
        return final_predictions, final_confidences, point_seen_count
    
    def post_process_predictions(self, points, predictions, confidences, min_cluster_size=50):
        """Post-process predictions to remove noise and small clusters"""
        target_mask = predictions == 1
        target_points = points[target_mask]
        
        if len(target_points) == 0:
            return predictions, confidences
        
        clustering = DBSCAN(eps=0.05, min_samples=min_cluster_size)
        cluster_labels = clustering.fit_predict(target_points)
        
        unique_labels, counts = np.unique(cluster_labels[cluster_labels >= 0], return_counts=True)
        
        if len(unique_labels) > 0:
            largest_cluster_label = unique_labels[np.argmax(counts)]
            largest_cluster_mask = cluster_labels == largest_cluster_label
            
            refined_predictions = np.zeros_like(predictions)
            target_indices = np.where(target_mask)[0]
            refined_target_indices = target_indices[largest_cluster_mask]
            refined_predictions[refined_target_indices] = 1
            
            return refined_predictions, confidences
        
        return predictions, confidences
    
    def process_single_scene(self, pcd_path, scene_number, output_dir):
        """Process a single scene and generate segmentation results"""
        scene_name = f"Scene_{scene_number}"
        
        points, colors = self.load_pcd_scene(pcd_path)
        
        if len(points) == 0:
            print(f"Failed to load scene {scene_number}")
            return None
        
        if len(colors) == 0 or colors.shape[0] == 0:
            colors = np.ones((len(points), 3), dtype=np.float32) * 0.5
        
        min_len = min(len(points), len(colors))
        points = points[:min_len]
        colors = colors[:min_len]
        
        total_points = points.shape[0]
        
        chunks = self.generate_chunk_grid(points)
        
        all_predictions = []
        
        for i, (chunk_min, chunk_max) in enumerate(chunks):
            chunk_points, chunk_colors, point_indices = self.extract_chunk_points(
                points, colors, chunk_min, chunk_max
            )
            
            if len(chunk_points) == 0:
                continue
            
            predictions, probabilities = self.run_inference_on_chunk(chunk_points, chunk_colors)
            
            if predictions is not None:
                all_predictions.append((predictions, probabilities, point_indices))
            
            if i % 20 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        final_predictions, final_confidences, point_coverage = self.aggregate_overlapping_predictions(
            all_predictions, total_points
        )
        
        refined_predictions, refined_confidences = self.post_process_predictions(
            points, final_predictions, final_confidences
        )
        
        self.current_confidences = refined_confidences
        
        target_mask = refined_predictions == 1
        background_mask = refined_predictions == 0
        
        high_conf_bg = (refined_predictions == 0) & (refined_confidences > 0.8)
        alien_mask = high_conf_bg
        pure_background_mask = background_mask & ~alien_mask
        
        target_count = np.sum(target_mask)
        alien_count = np.sum(alien_mask)
        background_count = np.sum(pure_background_mask)
        
        target_detection_rate = target_count / total_points * 100
        alien_detection_rate = alien_count / total_points * 100
        
        results = {
            'points': points,
            'colors': colors,
            'predictions': refined_predictions,
            'confidences': refined_confidences,
            'point_coverage': point_coverage,
            'target_point_count': target_count,
            'alien_point_count': alien_count,
            'background_point_count': background_count,
            'total_point_count': total_points,
            'scene_name': scene_name,
            'target_detection_rate': target_detection_rate,
            'alien_detection_rate': alien_detection_rate
        }
        
        print(f"Scene {scene_number}: {total_points:,} points processed")
        print(f"  Target: {target_count:,} points ({target_detection_rate:.2f}%)")
        print(f"  Alien: {alien_count:,} points ({alien_detection_rate:.2f}%)")
        
        scene_output_dir = os.path.join(output_dir, scene_name)
        os.makedirs(scene_output_dir, exist_ok=True)
        
        results_file = os.path.join(scene_output_dir, f"{scene_name}_results.h5")
        self.save_results_data(results, results_file)
        
        self.create_segmentation_images(points, colors, refined_predictions, refined_confidences, scene_name, scene_output_dir)
        self.save_point_cloud_visualizations(points, colors, refined_predictions, scene_name, scene_output_dir)
        
        return results

    def create_segmentation_images(self, points, colors, predictions, confidences, scene_name, save_dir):
        """Create and save segmentation visualization images"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'LDGCNN Segmentation Results - {scene_name}', fontsize=16, fontweight='bold')
        
        background_mask = predictions == 0
        target_mask = predictions == 1
        
        high_conf_bg = (predictions == 0) & (confidences > 0.8)
        alien_mask = high_conf_bg
        pure_background_mask = background_mask & ~alien_mask
        
        # Apply orientation correction for visualization
        points_corrected = self.correct_scene_orientation(points)
        
        # Original scene
        ax1 = axes[0]
        ax1.scatter(points_corrected[:, 0], points_corrected[:, 1], c=colors, s=1, alpha=0.8)
        ax1.set_xlabel('X coordinate')
        ax1.set_ylabel('Y coordinate')
        ax1.set_title('Original Scene - Top View (XY)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Segmentation overlay
        ax2 = axes[1]
        ax2.scatter(points_corrected[:, 0], points_corrected[:, 1], c=colors, s=1, alpha=0.8)
        
        if np.sum(alien_mask) > 0:
            ax2.scatter(points_corrected[alien_mask, 0], points_corrected[alien_mask, 1], 
                    c='blue', s=8, alpha=0.9, edgecolors='darkblue', linewidth=0.5,
                    label=f'Alien Objects ({np.sum(alien_mask)} pts)')
        
        if np.sum(target_mask) > 0:
            ax2.scatter(points_corrected[target_mask, 0], points_corrected[target_mask, 1], 
                    c='red', s=10, alpha=0.9, edgecolors='darkred', linewidth=0.5,
                    label=f'Target Object ({np.sum(target_mask)} pts)')
        
        ax2.set_xlabel('X coordinate')
        ax2.set_ylabel('Y coordinate')
        ax2.set_title('Segmentation Overlay - Top View (XY)', fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        target_count = np.sum(target_mask)
        alien_count = np.sum(alien_mask)
        total_count = len(points)
        
        fig.text(0.5, 0.02, 
                f'Target Objects: {target_count:,} points | Alien Objects: {alien_count:,} points | Total: {total_count:,} points', 
                ha='center', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08)
        
        save_path = os.path.join(save_dir, f"{scene_name}_segmentation_analysis.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def save_point_cloud_visualizations(self, points, colors, predictions, scene_name, save_dir):
        """Save 3D point cloud visualizations as PCD files"""
        background_mask = predictions == 0
        target_mask = predictions == 1
        
        high_conf_bg = (predictions == 0) & (self.current_confidences > 0.8)
        alien_mask = high_conf_bg
        background_mask = (predictions == 0) & ~alien_mask
        
        # Original scene
        pcd_original = o3d.geometry.PointCloud()
        pcd_original.points = o3d.utility.Vector3dVector(points)
        pcd_original.colors = o3d.utility.Vector3dVector(colors)
        
        original_path = os.path.join(save_dir, f"{scene_name}_original.pcd")
        o3d.io.write_point_cloud(original_path, pcd_original)
        
        # Segmentation results
        pcd_segmented = o3d.geometry.PointCloud()
        pcd_segmented.points = o3d.utility.Vector3dVector(points)
        seg_colors = colors.copy()
        
        seg_colors[alien_mask] = [0.0, 0.0, 1.0]   # Blue for alien objects
        seg_colors[target_mask] = [1.0, 0.0, 0.0]  # Red for target objects
        
        pcd_segmented.colors = o3d.utility.Vector3dVector(seg_colors)
        
        segmented_path = os.path.join(save_dir, f"{scene_name}_segmented.pcd")
        o3d.io.write_point_cloud(segmented_path, pcd_segmented)
        
        # Target object only
        if np.sum(target_mask) > 0:
            target_points = points[target_mask]
            target_colors = colors[target_mask]
            
            pcd_target = o3d.geometry.PointCloud()
            pcd_target.points = o3d.utility.Vector3dVector(target_points)
            pcd_target.colors = o3d.utility.Vector3dVector(target_colors)
            
            target_path = os.path.join(save_dir, f"{scene_name}_target_only.pcd")
            o3d.io.write_point_cloud(target_path, pcd_target)
        
        # Alien objects only
        if np.sum(alien_mask) > 0:
            alien_points = points[alien_mask]
            alien_colors = colors[alien_mask]
            
            pcd_alien = o3d.geometry.PointCloud()
            pcd_alien.points = o3d.utility.Vector3dVector(alien_points)
            pcd_alien.colors = o3d.utility.Vector3dVector(alien_colors)
            
            alien_path = os.path.join(save_dir, f"{scene_name}_alien_only.pcd")
            o3d.io.write_point_cloud(alien_path, pcd_alien)
    
    def save_results_data(self, results, save_path):
        """Save results to HDF5 file"""
        with h5py.File(save_path, 'w') as f:
            f.create_dataset('points', data=results['points'])
            f.create_dataset('colors', data=results['colors'])
            f.create_dataset('predictions', data=results['predictions'])
            f.create_dataset('confidences', data=results['confidences'])
            f.create_dataset('point_coverage', data=results['point_coverage'])
            
            f.attrs['target_point_count'] = results['target_point_count']
            f.attrs['alien_point_count'] = results['alien_point_count']
            f.attrs['background_point_count'] = results['background_point_count']
            f.attrs['total_point_count'] = results['total_point_count']
            f.attrs['scene_name'] = results['scene_name']
            f.attrs['target_detection_rate'] = results['target_detection_rate']
            f.attrs['alien_detection_rate'] = results['alien_detection_rate']
    
    def create_batch_summary(self, all_results, output_dir):
        """Create overall summary for batch processing"""
        if not all_results:
            return
        
        scene_names = [r['scene_name'] for r in all_results]
        target_counts = [r['target_point_count'] for r in all_results]
        alien_counts = [r['alien_point_count'] for r in all_results]
        total_counts = [r['total_point_count'] for r in all_results]
        target_detection_rates = [r['target_detection_rate'] for r in all_results]
        alien_detection_rates = [r['alien_detection_rate'] for r in all_results]
        
        # Create summary figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('LDGCNN Batch Processing Summary', fontsize=16, fontweight='bold')
        
        # Detection counts per scene
        ax1 = axes[0, 0]
        x_pos = range(len(scene_names))
        width = 0.6
        
        bars1 = ax1.bar(x_pos, target_counts, width, label='Target Objects', color='red', alpha=0.8)
        bars2 = ax1.bar(x_pos, alien_counts, width, bottom=target_counts, label='Alien Objects', color='blue', alpha=0.8)
        
        ax1.set_xlabel('Scene Number')
        ax1.set_ylabel('Points Detected')
        ax1.set_title('Object Detection per Scene')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([name.replace('Scene_', '') for name in scene_names])
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Detection rates comparison
        ax2 = axes[0, 1]
        x_pos = np.arange(len(scene_names))
        width = 0.35
        
        bars1 = ax2.bar(x_pos - width/2, target_detection_rates, width, label='Target Rate', color='red', alpha=0.8)
        bars2 = ax2.bar(x_pos + width/2, alien_detection_rates, width, label='Alien Rate', color='blue', alpha=0.8)
        
        ax2.set_xlabel('Scene Number')
        ax2.set_ylabel('Detection Rate (%)')
        ax2.set_title('Detection Rates per Scene')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([name.replace('Scene_', '') for name in scene_names])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Overall statistics
        ax3 = axes[1, 0]
        target_success = sum(1 for count in target_counts if count > 0)
        alien_success = sum(1 for count in alien_counts if count > 0)
        total_scenes = len(all_results)
        
        summary_text = f"""TARGET OBJECTS:
Success Rate: {target_success}/{total_scenes} ({target_success/total_scenes*100:.1f}%)

ALIEN OBJECTS:
Success Rate: {alien_success}/{total_scenes} ({alien_success/total_scenes*100:.1f}%)

COMBINED:
Scenes with detection: {len(set([i for i, (t, a) in enumerate(zip(target_counts, alien_counts)) if t > 0 or a > 0]))}/{total_scenes}
        """
        
        ax3.text(0.1, 0.5, summary_text, transform=ax3.transAxes, ha='left', va='center', 
                fontsize=11, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax3.set_title('Detection Success Rates')
        ax3.axis('off')
        
        # Point count distribution
        ax4 = axes[1, 1]
        ax4.hist(total_counts, bins=min(10, len(total_counts)), alpha=0.7, color='gray', edgecolor='black')
        ax4.set_xlabel('Total Points per Scene')
        ax4.set_ylabel('Number of Scenes')
        ax4.set_title('Point Count Distribution')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        summary_path = os.path.join(output_dir, "batch_processing_summary.png")
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create detailed results CSV
        results_df = pd.DataFrame({
            'Scene': [name.replace('Scene_', '') for name in scene_names],
            'Total_Points': total_counts,
            'Target_Points': target_counts,
            'Alien_Points': alien_counts,
            'Target_Rate_Percent': target_detection_rates,
            'Alien_Rate_Percent': alien_detection_rates,
            'Target_Detected': [count > 0 for count in target_counts],
            'Alien_Detected': [count > 0 for count in alien_counts]
        })
        
        csv_path = os.path.join(output_dir, "batch_results_summary.csv")
        results_df.to_csv(csv_path, index=False)
        
        return summary_path, csv_path


# =============================================================================
# LDGCNN MODEL CLASSES
# =============================================================================

def knn_memory_efficient(x, k):
    """Memory-efficient k-NN computation for large point clouds"""
    batch_size, num_points, num_dims = x.shape
    
    if num_points <= 1:
        dummy_idx = torch.zeros((batch_size, num_points, 1), dtype=torch.long, device=x.device)
        return dummy_idx
    
    effective_k = min(k, num_points - 1)
    effective_k = max(1, effective_k)
    
    try:
        device = x.device
        chunk_size = min(512, num_points)
        all_indices = []
        
        for start_idx in range(0, num_points, chunk_size):
            end_idx = min(start_idx + chunk_size, num_points)
            chunk_points = x[:, start_idx:end_idx, :]
            
            chunk_norms = torch.sum(chunk_points ** 2, dim=2, keepdim=True)
            all_norms = torch.sum(x ** 2, dim=2, keepdim=True)
            
            dot_products = torch.einsum('bic,bjc->bij', chunk_points, x)
            squared_distances = chunk_norms + all_norms.transpose(1, 2) - 2 * dot_products
            pairwise_distance = -squared_distances
            
            for i in range(start_idx, end_idx):
                if i < num_points:
                    pairwise_distance[:, i - start_idx, i] = float('-inf')
            
            chunk_indices = pairwise_distance.topk(k=effective_k, dim=-1)[1]
            all_indices.append(chunk_indices)
            
            del chunk_norms, dot_products, squared_distances, pairwise_distance, chunk_indices
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        idx = torch.cat(all_indices, dim=1)
        return idx
        
    except Exception as e:
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
    """Memory-efficient graph feature extraction with dynamic k adjustment"""
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    effective_k = min(k, num_points - 1)
    effective_k = max(1, effective_k)
    
    if idx is None:
        x_for_knn = x.transpose(2, 1).contiguous()
        try:
            if dim9 == False:
                idx = knn_memory_efficient(x_for_knn, k=effective_k)
            else:
                idx = knn_memory_efficient(x_for_knn[:, :, 6:], k=effective_k)
        except Exception as knn_error:
            raise knn_error
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    
    _, num_dims, _ = x.size()
    
    x = x.transpose(2, 1).contiguous()
    
    try:
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, effective_k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, effective_k, 1)
        
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    except RuntimeError as e:
        if "out of memory" in str(e):
            return get_graph_feature_memory_efficient(x.transpose(2, 1).contiguous(), k=effective_k//2, idx=None, dim9=dim9)
        else:
            raise e
    
    return feature

class EdgeConv(nn.Module):
    """Memory-efficient Edge Convolution layer for dynamic graph CNN"""
    def __init__(self, in_channels, out_channels, k=10):
        super(EdgeConv, self).__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels*2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
    def forward(self, x):
        batch_size, channels, num_points = x.shape
        
        if num_points < self.k:
            effective_k = max(1, num_points - 1)
        else:
            effective_k = self.k
        
        try:
            graph_features = get_graph_feature_memory_efficient(x, k=effective_k)
        except Exception as e:
            raise e
        
        x = self.conv(graph_features)
        x = x.max(dim=-1, keepdim=False)[0]
        return x

class LDGCNN(nn.Module):
    """Linked Dynamic Graph CNN for point cloud segmentation"""
    def __init__(self, args, output_channels=2):
        super(LDGCNN, self).__init__()
        self.args = args
        self.k = args.k
        
        # Feature extraction layers
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
        self.conv6 = nn.Conv1d(512, 256, 1)
        self.conv7 = nn.Conv1d(256, 128, 1)
        self.conv8 = nn.Conv1d(128, 64, 1)
        self.conv9 = nn.Conv1d(64, output_channels, 1)
        
        self.bn6 = nn.BatchNorm1d(256)
        self.bn7 = nn.BatchNorm1d(128)
        self.bn8 = nn.BatchNorm1d(64)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Hierarchical feature extraction with linking
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x4 = F.relu(self.bn4(self.conv4(x3)))
        x5 = F.relu(self.bn5(self.conv5(x4)))
        
        # Link hierarchical features
        x_linked = torch.cat((x1, x2, x3, x4, x5), dim=1)
        
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

class Args:
    """Configuration arguments for LDGCNN"""
    def __init__(self):
        self.k = 10
        self.emb_dims = 1024
        self.dropout = 0.3


def create_model(device, model_path):
    """Create and load LDGCNN model"""
    try:
        args = Args()
        model = LDGCNN(args, output_channels=2)
        model = model.to(device)
        
        if os.path.exists(model_path):
            print("Loading model weights...")
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            
            print("LDGCNN model loaded successfully")
            return model
        else:
            print(f"Model file not found: {model_path}")
            return None
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate LDGCNN point cloud segmentation model')
    parser.add_argument('--model-path', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--scene-dir', required=True, help='Directory containing PCD scene files')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    parser.add_argument('--device', default='auto', help='Device (auto/cuda/cpu)')
    parser.add_argument('--max-points', type=int, default=20480, help='Maximum points per chunk')
    parser.add_argument('--grid-size', type=float, default=0.2, help='Grid size in meters')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Check directories
    if not os.path.exists(args.scene_dir):
        print(f"Scene directory not found: {args.scene_dir}")
        return
    
    # Find PCD files
    pcd_files = [f for f in os.listdir(args.scene_dir) if f.endswith('.pcd')]
    pcd_files.sort()
    
    if not pcd_files:
        print("No PCD files found")
        return
    
    print(f"Found {len(pcd_files)} PCD files")
    
    # Create model
    model = create_model(device, args.model_path)
    if model is None:
        return
    
    # Initialize processor
    processor = LDGCNNSceneProcessor(
        model=model,
        device=device,
        max_points_per_chunk=args.max_points,
        overlap_ratio=0.3,
        grid_size=args.grid_size,
        adaptive_chunking=False
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Starting evaluation on {len(pcd_files)} scenes...")
    
    # Process all scenes
    all_results = []
    failed_scenes = []
    
    for i, pcd_file in enumerate(pcd_files, 1):
        scene_number = pcd_file.replace('.pcd', '').replace('realworld_scene_', '').replace('_ascii_rgb', '')
        scene_path = os.path.join(args.scene_dir, pcd_file)
        
        print(f"\nProcessing Scene {i}/{len(pcd_files)}: {scene_number}")
        
        try:
            result = processor.process_single_scene(
                scene_path, scene_number, args.output_dir
            )
            
            if result:
                all_results.append(result)
            else:
                failed_scenes.append(scene_number)
                
        except Exception as e:
            print(f"Error processing Scene {scene_number}: {e}")
            failed_scenes.append(scene_number)
            continue
        
        # Clear GPU cache between scenes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Create batch summary
    if all_results:
        processor.create_batch_summary(all_results, args.output_dir)
    
    # Final summary
    print(f"\nEvaluation completed!")
    print(f"Successfully processed: {len(all_results)}/{len(pcd_files)} scenes")
    
    if failed_scenes:
        print(f"Failed scenes: {', '.join(failed_scenes)}")
    
    if all_results:
        target_detected = sum(1 for r in all_results if r['target_point_count'] > 0)
        alien_detected = sum(1 for r in all_results if r['alien_point_count'] > 0)
        
        print(f"Target detection success: {target_detected}/{len(all_results)} scenes")
        print(f"Alien detection success: {alien_detected}/{len(all_results)} scenes")
        
        total_points = sum(r['total_point_count'] for r in all_results)
        total_target_points = sum(r['target_point_count'] for r in all_results)
        total_alien_points = sum(r['alien_point_count'] for r in all_results)
        
        print(f"Total points processed: {total_points:,}")
        print(f"Total target points found: {total_target_points:,}")
        print(f"Total alien points found: {total_alien_points:,}")
        
        print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()