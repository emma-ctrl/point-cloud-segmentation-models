#!/usr/bin/env python3
"""
PointTransformerV2 Evaluation Script

Evaluation script for PTv2 point cloud segmentation models.
Processes real-world scenes and generates segmentation results.

Usage:
    python PTv2/evaluate_ptv2.py --model-path path/to/model.pth --scene-dir path/to/scenes --output-dir results/
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

# Add Pointcept to path
sys.path.append('./Pointcept')


class PTv2SceneProcessor:
    """Point cloud scene processor for PTv2 model inference"""
    
    def __init__(self, model, device, max_points_per_chunk=None, overlap_ratio=0.3, 
                 grid_sizes=(0.06, 0.12, 0.24, 0.48), adaptive_chunking=True):
        self.model = model
        self.device = device
        self.max_points_per_chunk = max_points_per_chunk
        self.overlap_ratio = overlap_ratio
        self.grid_sizes = grid_sizes
        self.adaptive_chunking = adaptive_chunking
        self.model.eval()
        self.current_confidences = None
        
    def correct_scene_orientation(self, points):
        """Apply orientation correction (180-degree rotation around X-axis)"""
        points_corrected = points.copy()
        points_corrected[:, 1] = -points_corrected[:, 1]
        points_corrected[:, 2] = -points_corrected[:, 2]
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
            
            points = self.correct_scene_orientation(points)
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
            
            if len(points) > 0:
                points = self.correct_scene_orientation(points)
            
            return points, colors
            
        except Exception:
            return np.array([]), np.array([])
    
    def compute_scene_bounds(self, points):
        """Compute 3D bounding box of the scene"""
        min_bounds = np.min(points, axis=0)
        max_bounds = np.max(points, axis=0)
        scene_size = max_bounds - min_bounds
        return min_bounds, max_bounds, scene_size
    
    def adaptive_chunk_size_calculation(self, points, target_chunk_density=15000):
        """Calculate optimal chunk size based on point density"""
        min_bounds, max_bounds, scene_size = self.compute_scene_bounds(points)
        scene_volume = np.prod(scene_size)
        point_density = len(points) / scene_volume if scene_volume > 0 else 1000
        
        target_volume = target_chunk_density / point_density
        chunk_size = np.cbrt(target_volume)
        
        # Ensure minimum chunk size compatible with PTv2 grid sizes
        min_chunk_size = max(self.grid_sizes) * 10
        chunk_size = max(min_chunk_size, min(2.0, chunk_size))
        
        return chunk_size
    
    def generate_chunk_grid(self, points, chunk_size_meters=None):
        """Generate 3D grid of overlapping chunks"""
        min_bounds, max_bounds, scene_size = self.compute_scene_bounds(points)
        
        if self.adaptive_chunking and chunk_size_meters is None:
            chunk_size_meters = self.adaptive_chunk_size_calculation(points)
        elif chunk_size_meters is None:
            chunk_size_meters = max(self.grid_sizes) * 12
        
        step_size = chunk_size_meters * (1 - self.overlap_ratio)
        step_size = max(step_size, 0.01)
        
        chunks = []
        
        x_steps = int(np.ceil((max_bounds[0] - min_bounds[0]) / step_size)) + 1
        y_steps = int(np.ceil((max_bounds[1] - min_bounds[1]) / step_size)) + 1
        z_steps = int(np.ceil((max_bounds[2] - min_bounds[2]) / step_size)) + 1
        
        max_chunks_per_dim = 100
        x_steps = min(x_steps, max_chunks_per_dim)
        y_steps = min(y_steps, max_chunks_per_dim)
        z_steps = min(z_steps, max_chunks_per_dim)
        
        x_positions = np.linspace(min_bounds[0], max_bounds[0], x_steps)
        y_positions = np.linspace(min_bounds[1], max_bounds[1], y_steps)
        z_positions = np.linspace(min_bounds[2], max_bounds[2], z_steps)
        
        for x in x_positions:
            for y in y_positions:
                for z in z_positions:
                    chunk_min = np.array([x, y, z])
                    chunk_max = chunk_min + chunk_size_meters
                    chunks.append((chunk_min, chunk_max))
        
        return chunks
    
    def extract_chunk_points(self, points, colors, chunk_min, chunk_max):
        """Extract points within a 3D chunk"""
        mask = np.all((points >= chunk_min) & (points <= chunk_max), axis=1)
        chunk_points = points[mask]
        chunk_colors = colors[mask]
        point_indices = np.where(mask)[0]
        
        return chunk_points, chunk_colors, point_indices
    
    def normalize_coordinates(self, coord):
        """Normalize coordinates for PTv2"""
        coord = coord.astype(np.float32)
        coord_center = np.mean(coord, axis=0, keepdims=True)
        coord_centered = coord - coord_center
        coord_scale = np.max(np.abs(coord_centered)) + 1e-6
        coord_normalized = coord_centered / coord_scale * 3.0
        return coord_normalized

    def prepare_ptv2_data_dict(self, chunk_points, chunk_colors):
        """Prepare chunk data for PTv2 model input"""
        num_points = chunk_points.shape[0]
        
        if num_points == 0:
            return None
        
        if len(chunk_colors) == 0 or chunk_colors.shape[0] == 0:
            chunk_colors = np.ones((num_points, 3), dtype=np.float32) * 0.5
        
        if self.max_points_per_chunk is not None and num_points > self.max_points_per_chunk:
            indices = np.random.choice(num_points, self.max_points_per_chunk, replace=False)
            sampled_points = chunk_points[indices]
            sampled_colors = chunk_colors[indices]
            sampling_indices = indices
        else:
            sampled_points = chunk_points
            sampled_colors = chunk_colors
            sampling_indices = np.arange(num_points)
            
        coord = self.normalize_coordinates(sampled_points)
        
        # Use normalized XYZ as features for PTv2
        feat = coord.copy()
        
        if feat.shape[1] != 3:
            feat = feat[:, :3]
        
        coord_tensor = torch.tensor(coord, dtype=torch.float32, device=self.device)
        feat_tensor = torch.tensor(feat, dtype=torch.float32, device=self.device)
        
        offset = torch.tensor([len(coord)], dtype=torch.long, device=self.device)
        
        data_dict = {
            'coord': coord_tensor,
            'feat': feat_tensor,
            'offset': offset
        }
        
        return data_dict, sampling_indices
    
    def run_inference_on_chunk(self, chunk_points, chunk_colors):
        """Run PTv2 model inference on a single chunk"""
        result = self.prepare_ptv2_data_dict(chunk_points, chunk_colors)
        
        if result is None:
            return None, None
            
        data_dict, sampling_indices = result
        
        with torch.no_grad():
            logits = self.model(data_dict)
            probabilities = torch.softmax(logits, dim=-1)
            predicted_labels = torch.argmax(probabilities, dim=-1)
        
        predicted_labels = predicted_labels.cpu().numpy()
        probabilities = probabilities.cpu().numpy()
        
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
    
    def process_single_scene(self, pcd_path, scene_number, output_dir, chunk_size_meters=None):
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
        
        chunks = self.generate_chunk_grid(points, chunk_size_meters)
        
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
            
            if i % 10 == 0 and torch.cuda.is_available():
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
        fig.suptitle(f'PTv2 Segmentation Results - {scene_name}', fontsize=16, fontweight='bold')
        
        background_mask = predictions == 0
        target_mask = predictions == 1
        
        high_conf_bg = (predictions == 0) & (confidences > 0.8)
        alien_mask = high_conf_bg
        pure_background_mask = background_mask & ~alien_mask
        
        # Original scene
        ax1 = axes[0]
        ax1.scatter(points[:, 0], points[:, 1], c=colors, s=1, alpha=0.8)
        ax1.set_xlabel('X coordinate')
        ax1.set_ylabel('Y coordinate')
        ax1.set_title('Original Scene - Top View (XY)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Segmentation overlay
        ax2 = axes[1]
        ax2.scatter(points[:, 0], points[:, 1], c=colors, s=1, alpha=0.8)
        
        if np.sum(alien_mask) > 0:
            ax2.scatter(points[alien_mask, 0], points[alien_mask, 1], 
                    c='blue', s=8, alpha=0.9, edgecolors='darkblue', linewidth=0.5,
                    label=f'Alien Objects ({np.sum(alien_mask)} pts)')
        
        if np.sum(target_mask) > 0:
            ax2.scatter(points[target_mask, 0], points[target_mask, 1], 
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
        
        seg_colors[alien_mask] = [0.0, 0.0, 1.0]
        seg_colors[target_mask] = [1.0, 0.0, 0.0]
        
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
        fig.suptitle('PTv2 Batch Processing Summary', fontsize=16, fontweight='bold')
        
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


def create_model(device, model_path):
    """Create and load PTv2 model"""
    try:
        from pointcept.models.point_transformer_v2.point_transformer_v2m2_base import PointTransformerV2 as PTv2M2
        print("Successfully imported PointTransformerV2")
    except ImportError as e:
        print(f"Error importing PointTransformerV2: {e}")
        return None
    
    # Model configuration
    feat_dim = 3
    num_classes = 2
    grid_sizes = (0.06, 0.12, 0.24, 0.48)
    
    ptv2_config = {
        'in_channels': feat_dim,
        'num_classes': num_classes,
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
        'grid_sizes': grid_sizes,
        'attn_qkv_bias': True,
        'pe_multiplier': False,
        'pe_bias': True,
        'attn_drop_rate': 0.0,
        'drop_path_rate': 0.0,
        'enable_checkpoint': False,
        'unpool_backend': "map",
    }
    
    try:
        model = PTv2M2(**ptv2_config).to(device)
        print("PointTransformerV2 model initialized")
    except Exception as e:
        print(f"Error initializing PTv2: {e}")
        return None
    
    # Load trained weights
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            print("Model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    else:
        print(f"Model file not found: {model_path}")
        return None


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate PTv2 point cloud segmentation model')
    parser.add_argument('--model-path', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--scene-dir', required=True, help='Directory containing PCD scene files')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    parser.add_argument('--device', default='auto', help='Device (auto/cuda/cpu)')
    parser.add_argument('--max-points', type=int, default=None, help='Maximum points per chunk')
    parser.add_argument('--chunk-size', type=float, default=None, help='Chunk size in meters')
    
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
    processor = PTv2SceneProcessor(
        model=model,
        device=device,
        max_points_per_chunk=args.max_points,
        overlap_ratio=0.2,
        grid_sizes=(0.06, 0.12, 0.24, 0.48),
        adaptive_chunking=True
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
                scene_path, scene_number, args.output_dir, args.chunk_size
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