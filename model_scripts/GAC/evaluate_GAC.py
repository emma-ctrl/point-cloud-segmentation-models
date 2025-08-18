#!/usr/bin/env python3
"""
GACNet Evaluation Script

Evaluation script for GACNet point cloud segmentation models.
Processes real-world scenes and generates segmentation results.

Usage:
    python evaluate_gacnet.py --model-path path/to/model.pth --scene-dir path/to/scenes --output-dir results/
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


class GACNetSceneProcessor:
    """Point cloud scene processor for GACNet model inference"""
    
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
        """Normalize coordinates for GACNet"""
        coord = coord.astype(np.float32)
        coord_center = np.mean(coord, axis=0, keepdims=True)
        coord = coord - coord_center
        coord_scale = np.max(np.abs(coord)) + 1e-6
        coord = coord / coord_scale
        return coord

    def prepare_gacnet_data(self, chunk_points, chunk_colors):
        """Prepare chunk data for GACNet model input"""
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
        
        # Prepare features (same as training)
        if sampled_colors.max() > 1.0:
            sampled_colors = sampled_colors / 255.0
        
        # Compute additional features
        z_coords = coord[:, 2]
        height_feature = (z_coords - z_coords.min()) / (z_coords.max() - z_coords.min() + 1e-8)
        
        centroid = np.mean(coord, axis=0)
        distances_to_centroid = np.linalg.norm(coord - centroid, axis=1)
        distance_feature = distances_to_centroid / (distances_to_centroid.max() + 1e-8)
        
        # Simplified density feature
        if len(coord) > 10:
            k_neighbors = min(10, len(coord) - 1)
            density_feature = np.zeros(len(coord))
            
            for i in range(0, len(coord), max(1, len(coord) // 100)):
                dists = np.linalg.norm(coord - coord[i], axis=1)
                kth_dist = np.partition(dists, k_neighbors)[k_neighbors]
                nearby_mask = dists < kth_dist * 2
                density_feature[nearby_mask] = 1.0 / (kth_dist + 1e-8)
            
            if density_feature.max() > 0:
                density_feature = density_feature / density_feature.max()
        else:
            density_feature = np.ones(len(coord)) * 0.5
        
        # Combine features: RGB (3) + Height (1) + Distance (1) + Density (1) = 6 features
        features = np.column_stack([
            sampled_colors,
            height_feature,
            distance_feature,
            density_feature
        ])
        
        # Convert to tensor format [1, C, N]
        xyz_tensor = torch.tensor(coord.T, dtype=torch.float32, device=self.device)
        features_tensor = torch.tensor(features.T, dtype=torch.float32, device=self.device)
        
        xyz_tensor = xyz_tensor.unsqueeze(0)
        features_tensor = features_tensor.unsqueeze(0)
        
        return xyz_tensor, features_tensor, sampling_indices
    
    def run_inference_on_chunk(self, chunk_points, chunk_colors):
        """Run GACNet model inference on a single chunk"""
        result = self.prepare_gacnet_data(chunk_points, chunk_colors)
        
        if result is None:
            return None, None
            
        xyz_tensor, features_tensor, sampling_indices = result
        
        with torch.no_grad():
            output = self.model(xyz_tensor, features_tensor)
            probabilities = torch.exp(output)  # Convert from log_softmax
            predicted_labels = torch.argmax(output, dim=-1)
        
        predicted_labels = predicted_labels.squeeze(0).cpu().numpy()
        probabilities = probabilities.squeeze(0).cpu().numpy()
        
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
        fig.suptitle(f'GACNet Segmentation Results - {scene_name}', fontsize=16, fontweight='bold')
        
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
        fig.suptitle('GACNet Batch Processing Summary', fontsize=16, fontweight='bold')
        
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
# GACNET MODEL CLASSES
# =============================================================================

def square_distance(src, dst):
    """Calculate Euclidean distance between each two points"""
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """Index points data"""
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """Farthest point sampling"""
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """Query ball point"""
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

def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """Sample and group points"""
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        fps_points = index_points(points, fps_idx)
        fps_points = torch.cat([new_xyz, fps_points], dim=-1)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
        fps_points = new_xyz
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_points
    else:
        return new_xyz, new_points

def sample_and_group_all(xyz, points):
    """Sample and group all points"""
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

class GraphAttention(nn.Module):
    """Graph attention module"""
    def __init__(self, all_channel, feature_dim, dropout, alpha):
        super(GraphAttention, self).__init__()
        self.alpha = alpha
        self.a = nn.Parameter(torch.zeros(size=(all_channel, feature_dim)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, center_xyz, center_feature, grouped_xyz, grouped_feature):
        B, npoint, C = center_xyz.size()
        _, _, nsample, D = grouped_feature.size()
        delta_p = center_xyz.view(B, npoint, 1, C).expand(B, npoint, nsample, C) - grouped_xyz
        delta_h = center_feature.view(B, npoint, 1, D).expand(B, npoint, nsample, D) - grouped_feature
        delta_p_concat_h = torch.cat([delta_p, delta_h], dim=-1)
        e = self.leakyrelu(torch.matmul(delta_p_concat_h, self.a))
        attention = F.softmax(e, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        graph_pooling = torch.sum(torch.mul(attention, grouped_feature), dim=2)
        return graph_pooling

class GraphAttentionConvLayer(nn.Module):
    """Graph attention convolution layer"""
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, dropout=0.6, alpha=0.2):
        super(GraphAttentionConvLayer, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.dropout = dropout
        self.alpha = alpha
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all
        self.GAT = GraphAttention(3 + last_channel, last_channel, self.dropout, self.alpha)

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points, grouped_xyz, fps_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, True)
        
        new_points = new_points.permute(0, 3, 2, 1)
        fps_points = fps_points.unsqueeze(3).permute(0, 2, 3, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            fps_points = F.relu(bn(conv(fps_points)))
            new_points = F.relu(bn(conv(new_points)))
        
        new_points = self.GAT(center_xyz=new_xyz,
                              center_feature=fps_points.squeeze(2).permute(0, 2, 1),
                              grouped_xyz=grouped_xyz,
                              grouped_feature=new_points.permute(0, 3, 2, 1))
        new_xyz = new_xyz.permute(0, 2, 1)
        new_points = new_points.permute(0, 2, 1)
        return new_xyz, new_points

class PointNetFeaturePropagation(nn.Module):
    """PointNet feature propagation module"""
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
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            dists[dists < 1e-10] = 1e-10
            weight = 1.0 / dists
            weight = weight / torch.sum(weight, dim=-1).view(B, N, 1)
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

class GACNet(nn.Module):
    """Graph Attention Convolution Network"""
    def __init__(self, num_classes, dropout=0.6, alpha=0.2):
        super(GACNet, self).__init__()
        self.sa1 = GraphAttentionConvLayer(1024, 0.1, 32, 6 + 3, [32, 32, 64], False, dropout, alpha)
        self.sa2 = GraphAttentionConvLayer(256, 0.2, 32, 64 + 3, [64, 64, 128], False, dropout, alpha)
        self.sa3 = GraphAttentionConvLayer(64, 0.4, 32, 128 + 3, [128, 128, 256], False, dropout, alpha)
        self.sa4 = GraphAttentionConvLayer(16, 0.8, 32, 256 + 3, [256, 256, 512], False, dropout, alpha)
        
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, points):
        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x


def create_model(device, model_path):
    """Create and load GACNet model"""
    try:
        model = GACNet(num_classes=2, dropout=0.6, alpha=0.2)
        model = model.to(device)
        
        if os.path.exists(model_path):
            print("Loading model weights...")
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint, strict=False)
            model.eval()
            
            print("GACNet model loaded successfully")
            return model
        else:
            print(f"Model file not found: {model_path}")
            return None
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate GACNet point cloud segmentation model')
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
    processor = GACNetSceneProcessor(
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