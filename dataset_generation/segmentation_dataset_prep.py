import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import h5py
import random
import glob
import math
import torch
import time
from tqdm import tqdm
import pickle
import gc
import psutil  # For monitoring memory usage---99-9

# Add this debug code at the top of your script (after imports) to diagnose GPU issues

print("=== GPU DIAGNOSTIC ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
else:
    print("CUDA not available - likely CPU-only PyTorch installation")

# Also check if NVIDIA driver/CUDA toolkit is available
try:
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("\n=== NVIDIA-SMI OUTPUT ===")
        print(result.stdout)
    else:
        print("nvidia-smi not found or failed")
except:
    print("Could not run nvidia-smi")

print("========================")

# Path variables
OBJECT_DATASET_PATH = "D:/Emma/MiniMarket_Emma/datasets/MiniMarket_processed_dataset/" #processed data directory
GENERATED_DATASET_PATH = "D:/Emma/MiniMarket_Emma/datasets/MiniMarket_segmentation_dataset"  #output directory
CACHE_DIR = "D:/Emma/MiniMarket_Emma/MiniMarket_cache"  # Cache directory
if not os.path.exists(GENERATED_DATASET_PATH):
    os.makedirs(GENERATED_DATASET_PATH)
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Configuration parameters - OPTIMIZED FOR SPEED
NUM_POINTS_PER_SEG_SAMPLE = 20480  # 10 objects max x 2048 points
MAX_ALIEN_OBJECTS = 9 
NUM_ALIEN_OBJECTS = 9  
NUM_TRANSFORMATIONS = 10
BATCH_SIZE = 32
MAX_CACHED_ALIEN_OBJECTS = 10
SAVE_CHECKPOINT_EVERY = 100
NUM_WORKERS = 1  # Number of parallel processes
MAX_SAMPLES_PER_OBJECT = 1200  
TARGET_OBJECT = "biscuit_spread_lotus_400gm"  # Target object name
SAVE_CHUNKS = True  # Save data in chunks instead of all at once
CHUNK_SIZE = 100

# Memory monitoring function
def print_memory_usage():
    """Print current memory usage of the process"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"CPU Memory usage: {mem_info.rss / (1024 * 1024):.2f} MB")
    
    # Print GPU memory if available
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / (1024 * 1024):.2f} MB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / (1024 * 1024):.2f} MB")

# Set default tensor type to float32
torch.set_default_tensor_type(torch.FloatTensor)

# Check for CUDA
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"CUDA detected: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("Using CPU")
    
print(f"Using device: {device}")

def Rotx(t, batch_size=1):
    """Create rotation matrix around x-axis using PyTorch"""
    if batch_size == 1:
        cos_t = torch.cos(t)
        sin_t = torch.sin(t)
        return torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cos_t, -sin_t, 0.0],
            [0.0, sin_t, cos_t, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], device=device)
    else:
        cos_t = torch.cos(t)
        sin_t = torch.sin(t)
        zeros = torch.zeros(batch_size, device=device)
        ones = torch.ones(batch_size, device=device)
        
        R = torch.zeros(batch_size, 4, 4, device=device)
        R[:, 0, 0] = ones
        R[:, 1, 1] = cos_t
        R[:, 1, 2] = -sin_t
        R[:, 2, 1] = sin_t
        R[:, 2, 2] = cos_t
        R[:, 3, 3] = ones
        return R

def Roty(t, batch_size=1):
    """Create rotation matrix around y-axis using PyTorch"""
    if batch_size == 1:
        cos_t = torch.cos(t)
        sin_t = torch.sin(t)
        return torch.tensor([
            [cos_t, 0.0, sin_t, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-sin_t, 0.0, cos_t, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], device=device)
    else:
        cos_t = torch.cos(t)
        sin_t = torch.sin(t)
        zeros = torch.zeros(batch_size, device=device)
        ones = torch.ones(batch_size, device=device)
        
        R = torch.zeros(batch_size, 4, 4, device=device)
        R[:, 0, 0] = cos_t
        R[:, 0, 2] = sin_t
        R[:, 1, 1] = ones
        R[:, 2, 0] = -sin_t
        R[:, 2, 2] = cos_t
        R[:, 3, 3] = ones
        return R

def Rotz(t, batch_size=1):
    """Create rotation matrix around z-axis using PyTorch"""
    if batch_size == 1:
        cos_t = torch.cos(t)
        sin_t = torch.sin(t)
        return torch.tensor([
            [cos_t, -sin_t, 0.0, 0.0],
            [sin_t, cos_t, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], device=device)
    else:
        cos_t = torch.cos(t)
        sin_t = torch.sin(t)
        zeros = torch.zeros(batch_size, device=device)
        ones = torch.ones(batch_size, device=device)
        
        R = torch.zeros(batch_size, 4, 4, device=device)
        R[:, 0, 0] = cos_t
        R[:, 0, 1] = -sin_t
        R[:, 1, 0] = sin_t
        R[:, 1, 1] = cos_t
        R[:, 2, 2] = ones
        R[:, 3, 3] = ones
        return R

def random_homogeneous_3d_rotation(x_min, x_max, y_min, y_max, z_min, z_max, batch_size=1):
    """Generate random 3D rotation matrices for a batch"""
    if batch_size == 1:
        angle_about_x = random.uniform(x_min, x_max) * math.pi / 180
        angle_about_y = random.uniform(y_min, y_max) * math.pi / 180
        angle_about_z = random.uniform(z_min, z_max) * math.pi / 180
        
        angle_about_x = torch.tensor(angle_about_x, device=device)
        angle_about_y = torch.tensor(angle_about_y, device=device)
        angle_about_z = torch.tensor(angle_about_z, device=device)
        
        return torch.matmul(torch.matmul(Rotx(angle_about_x), 
                                       Roty(angle_about_y)), 
                           Rotz(angle_about_z))
    else:
        # Generate random angles for the batch
        angles_x = torch.tensor([random.uniform(x_min, x_max) * math.pi / 180 for _ in range(batch_size)], device=device)
        angles_y = torch.tensor([random.uniform(y_min, y_max) * math.pi / 180 for _ in range(batch_size)], device=device)
        angles_z = torch.tensor([random.uniform(z_min, z_max) * math.pi / 180 for _ in range(batch_size)], device=device)
        
        # Get rotation matrices for each axis
        rx = Rotx(angles_x, batch_size)
        ry = Roty(angles_y, batch_size)
        rz = Rotz(angles_z, batch_size)
        
        # Matrix multiplication for batches
        rxy = torch.bmm(rx[:, :3, :3], ry[:, :3, :3])
        result = torch.bmm(rxy, rz[:, :3, :3])
        
        # Convert to homogeneous 4x4 matrices
        homogeneous = torch.zeros(batch_size, 4, 4, device=device)
        homogeneous[:, :3, :3] = result
        homogeneous[:, 3, 3] = 1.0
        
        return homogeneous

def random_homogeneous_3d_translation(x_min, x_max, y_min, y_max, z_min, z_max, batch_size=1):
    """Generate random 3D translation matrices for a batch"""
    if batch_size == 1:
        translation = torch.tensor([
            [1.0, 0.0, 0.0, random.uniform(x_min, x_max)],
            [0.0, 1.0, 0.0, random.uniform(y_min, y_max)],
            [0.0, 0.0, 1.0, random.uniform(z_min, z_max)],
            [0.0, 0.0, 0.0, 1.0]
        ], device=device)
        return translation
    else:
        # For batch processing
        translations = torch.zeros(batch_size, 4, 4, device=device)
        
        # Set diagonal to ones
        translations[:, 0, 0] = 1.0
        translations[:, 1, 1] = 1.0
        translations[:, 2, 2] = 1.0
        translations[:, 3, 3] = 1.0
        
        # Set random translations
        translations[:, 0, 3] = torch.tensor([random.uniform(x_min, x_max) for _ in range(batch_size)], device=device)
        translations[:, 1, 3] = torch.tensor([random.uniform(y_min, y_max) for _ in range(batch_size)], device=device)
        translations[:, 2, 3] = torch.tensor([random.uniform(z_min, z_max) for _ in range(batch_size)], device=device)
        
        return translations

def load_and_cache_alien_objects(all_object_files, target_hdf5_file, max_cached=MAX_CACHED_ALIEN_OBJECTS):
    """Pre-load alien objects into memory for fast access"""
    alien_files = [f for f in all_object_files if f != target_hdf5_file]
    
    # Make sure we have at least NUM_ALIEN_OBJECTS different files
    min_required = min(NUM_ALIEN_OBJECTS, len(alien_files))
    
    if len(alien_files) > max_cached:
        alien_files = random.sample(alien_files, max(max_cached, min_required))
    
    print(f"Pre-loading {len(alien_files)} alien objects into memory...")
    start_time = time.time()
    
    alien_data_cache = {}
    for i, alien_file in enumerate(alien_files):
        print(f"  Loading {i+1}/{len(alien_files)}: {os.path.basename(alien_file)}")
        try:
            with h5py.File(alien_file, "r") as f:
                alien_data_cache[alien_file] = {
                    'points': f["point_clouds"][()],
                    'colors': f["color_clouds"][()]
                }
        except Exception as e:
            print(f"  ERROR loading {alien_file}: {e}")
            continue
    
    load_time = time.time() - start_time
    total_size_mb = sum(
        (data['points'].nbytes + data['colors'].nbytes) / (1024*1024) 
        for data in alien_data_cache.values()
    )
    
    print(f"Successfully cached {len(alien_data_cache)} alien objects")
    print(f"Total cache size: {total_size_mb:.1f} MB")
    print(f"Loading time: {load_time:.1f} seconds")
    
    return list(alien_data_cache.keys()), alien_data_cache

def process_sample_fast(target_points, target_colors, alien_files, alien_data_cache, object_data_size, sample_num=0):
    """Process a single sample using pre-loaded alien data (MUCH FASTER)"""
    
    start_time = time.time()
    
    # Convert target points and colors to tensors
    target_points_tensor = torch.tensor(target_points, device=device, dtype=torch.float32)
    
    # Add homogeneous coordinate
    ones = torch.ones(object_data_size, 1, device=device, dtype=torch.float32)
    target_points_homogeneous = torch.cat([target_points_tensor, ones], dim=1)
    
    # Generate random transformation
    translation = random_homogeneous_3d_translation(-0.25, 0.25, -0.25, 0.25, 0.0, 0.25)
    rotation = random_homogeneous_3d_rotation(-180, 180, -180, 180, -180, 180)
    transformation = torch.matmul(translation, rotation)
    
    # Apply transformation to target points
    transformed_points = torch.matmul(transformation, target_points_homogeneous.t()).t()
    
    # Remove homogeneous coordinate
    seg_sample_point = transformed_points[:, :3].cpu().numpy()
    seg_sample_color = target_colors
    
    # Use integer labels (1 for target object)
    seg_sample_label = np.ones(object_data_size, dtype=np.int32)
    
    # Always use exactly 9 alien objects
    num_alien_objects = 9
    
    # Make sure we have enough alien objects (reuse if necessary)
    copy_alien_files = alien_files.copy()
    
    # If we don't have enough unique alien files, reuse some
    if len(copy_alien_files) < num_alien_objects:
        while len(copy_alien_files) < num_alien_objects:
            copy_alien_files.append(random.choice(alien_files))
    
    # Shuffle and select exactly 9
    random.shuffle(copy_alien_files)
    alien_files_to_use = copy_alien_files[:num_alien_objects]
    
    # Process alien objects using cached data (FAST!)
    for i, alien_file in enumerate(alien_files_to_use):
        # Get cached data instead of loading from disk
        cached_data = alien_data_cache[alien_file]
        alien_sample_index = random.randrange(cached_data['points'].shape[0])
        alien_points = cached_data['points'][alien_sample_index]
        alien_colors = cached_data['colors'][alien_sample_index]
        
        # Create PyTorch tensor for alien points
        alien_points_tensor = torch.tensor(alien_points, device=device, dtype=torch.float32)
        
        # Add homogeneous coordinate
        alien_ones = torch.ones(object_data_size, 1, device=device, dtype=torch.float32)
        alien_points_homogeneous = torch.cat([alien_points_tensor, alien_ones], dim=1)
        
        # Generate random transformation for alien object
        alien_translation = random_homogeneous_3d_translation(-0.25, 0.25, -0.25, 0.25, 0.0, 0.25)
        alien_rotation = random_homogeneous_3d_rotation(-180, 180, -180, 180, -180, 180)
        alien_transformation = torch.matmul(alien_translation, alien_rotation)
        
        # Apply transformation
        transformed_alien = torch.matmul(alien_transformation, alien_points_homogeneous.t()).t()
        
        # Remove homogeneous coordinate
        transformed_alien_points = transformed_alien[:, :3].cpu().numpy()
        
        # Add to segmentation sample
        seg_sample_point = np.concatenate((seg_sample_point, transformed_alien_points), axis=0)
        seg_sample_color = np.concatenate((seg_sample_color, alien_colors), axis=0)
        
        # Use integer labels (0 for alien objects)
        alien_label = np.zeros(object_data_size, dtype=np.int32)
        seg_sample_label = np.concatenate((seg_sample_label, alien_label), axis=0)
    
    # Pad the remaining sample size with zeros
    total_points = seg_sample_point.shape[0]
    padding_needed = NUM_POINTS_PER_SEG_SAMPLE - total_points
    
    if padding_needed > 0:
        seg_sample_point = np.concatenate((seg_sample_point, np.zeros((padding_needed, 3))), axis=0)
        seg_sample_color = np.concatenate((seg_sample_color, np.zeros((padding_needed, 3))), axis=0)
        
        # Use integer labels (0 for padding - same as alien)
        padding_labels = np.zeros(padding_needed, dtype=np.int32)
        seg_sample_label = np.concatenate((seg_sample_label, padding_labels), axis=0)
    
    # Combine points and colors into single array [x, y, z, r, g, b]
    seg_sample_combined = np.concatenate((seg_sample_point, seg_sample_color), axis=1)
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    
    # Only print detailed info every 100 samples
    if sample_num % 100 == 1:
        print(f"Sample {sample_num} processed in {total_time:.2f}s")
    
    return seg_sample_combined, seg_sample_label

def save_chunks(points_colors_combined, labels, output_prefix, chunk_index):
    """Save a chunk of data to disk in PointNet format (NO .h5 extension)"""
    chunk_file = f"{output_prefix}_chunk_{chunk_index}"  # NO .h5 extension!
    print(f"Saving chunk {chunk_index} to {chunk_file}...")
    try:
        with h5py.File(chunk_file, 'w') as f:
            f.create_dataset("points", data=np.array(points_colors_combined))  # [x,y,z,r,g,b]
            f.create_dataset("labels", data=np.array(labels))  # Single integer per point
        print(f"Successfully saved chunk {chunk_index} with {len(points_colors_combined)} samples")
        return True
    except Exception as e:
        print(f"Error saving chunk {chunk_index}: {e}")
        return False

def generate_object_segmentation_dataset():
    """Generate segmentation dataset for the target object"""
    start_time = time.time()
    
    # Get the target file path
    target_hdf5_file = None
    all_object_files = sorted(glob.glob(OBJECT_DATASET_PATH + "/*_2048_*"))
    
    for file_path in all_object_files:
        if TARGET_OBJECT in file_path:
            target_hdf5_file = file_path
            break
    
    if not target_hdf5_file:
        print(f"Target object {TARGET_OBJECT} not found!")
        return
    
    print(f"Using target file: {target_hdf5_file}")
    print(f"IMPORTANT: Each sample will include EXACTLY 9 alien objects")
    print(f"Data will be formatted for PointNet: (N, {NUM_POINTS_PER_SEG_SAMPLE}, 6) for points+colors and (N, {NUM_POINTS_PER_SEG_SAMPLE}) for labels")
    print(f"OPTIMIZED SETTINGS:")
    print(f"  - MAX_SAMPLES_PER_OBJECT: {MAX_SAMPLES_PER_OBJECT}")
    print(f"  - NUM_TRANSFORMATIONS: {NUM_TRANSFORMATIONS}")
    print(f"  - CHUNK_SIZE: {CHUNK_SIZE}")
    print(f"  - Files will be saved WITHOUT .h5 extension")
    
    # Create output path
    target_object_name = os.path.basename(target_hdf5_file)
    output_prefix = os.path.join(GENERATED_DATASET_PATH, f"{target_object_name}_segmentation_{NUM_POINTS_PER_SEG_SAMPLE}")
    
    # PRE-LOAD ALIEN OBJECTS (BIG SPEED IMPROVEMENT!)
    alien_files, alien_data_cache = load_and_cache_alien_objects(all_object_files, target_hdf5_file)
    
    if len(alien_files) < 9:
        print(f"WARNING: Only {len(alien_files)} unique alien objects available, but will still use 9 total per sample (with duplicates)")
    else:
        print(f"Using {len(alien_files)} unique alien objects to select 9 for each sample")
    
    # Load target object data
    print("Loading target object data...")
    with h5py.File(target_hdf5_file, "r") as f:
        target_points = f["point_clouds"][()]
        target_colors = f["color_clouds"][()]
    
    object_data_size = target_points.shape[1]  # Should be 2048
    print(f"Target object has {target_points.shape[0]} samples, each with {object_data_size} points")
    
    # Track total samples and chunks
    total_samples = 0
    chunks_saved = 0
    
    # Current chunk data
    current_points_colors = []  # Combined [x,y,z,r,g,b]
    current_labels = []
    
    # Process each original sample
    num_samples_to_process = min(MAX_SAMPLES_PER_OBJECT, target_points.shape[0])
    total_samples_to_generate = num_samples_to_process * NUM_TRANSFORMATIONS
    print(f"Processing {num_samples_to_process} samples with {NUM_TRANSFORMATIONS} transformations each")
    print(f"Total samples to generate: {total_samples_to_generate}")
    
    # Progress tracking with timing
    pbar = tqdm(total=total_samples_to_generate, desc="Processing")
    last_time = time.time()
    samples_since_last = 0
    
    try:
        for sample_idx in range(num_samples_to_process):
            # Select one original sample
            sample_point = target_points[sample_idx,:,:]
            sample_color = target_colors[sample_idx,:,:]
            
            # Apply multiple transformations
            for transform_idx in range(NUM_TRANSFORMATIONS):
                # Process this sample using cached alien data (FAST!)
                points_colors_combined, labels = process_sample_fast(
                    sample_point, sample_color, alien_files, alien_data_cache, 
                    object_data_size, sample_num=total_samples+1
                )
                
                # Add to current chunk
                current_points_colors.append(points_colors_combined)
                current_labels.append(labels)
                
                total_samples += 1
                samples_since_last += 1
                pbar.update(1)
                
                # Calculate and print processing rate every 50 samples
                current_time = time.time()
                time_elapsed = current_time - last_time
                
                if samples_since_last >= 50:
                    rate = samples_since_last / time_elapsed if time_elapsed > 0 else 0
                    estimated_remaining = (total_samples_to_generate - total_samples) / rate if rate > 0 else 0
                    print(f"\n*** PROGRESS UPDATE ***")
                    print(f"Completed {total_samples}/{total_samples_to_generate} samples ({100*total_samples/total_samples_to_generate:.1f}%)")
                    print(f"Processing rate: {rate:.2f} samples/sec")
                    print(f"Estimated time remaining: {estimated_remaining/60:.1f} minutes")
                    print_memory_usage()
                    print("*" * 25)
                    
                    # Reset timing counters
                    last_time = current_time
                    samples_since_last = 0
                
                # Save chunk if needed
                if SAVE_CHUNKS and len(current_points_colors) >= CHUNK_SIZE:
                    success = save_chunks(current_points_colors, current_labels, 
                                          output_prefix, chunks_saved)
                    if success:
                        chunks_saved += 1
                        # Clear current chunk
                        current_points_colors = []
                        current_labels = []
                        
                        # Force garbage collection
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
            
            # Free memory after each original sample
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    except KeyboardInterrupt:
        print(f"\n*** INTERRUPTED BY USER ***")
        print(f"Processed {total_samples} samples so far")
        # Try to save the current chunk
        if current_points_colors:
            print("Saving current progress...")
            save_chunks(current_points_colors, current_labels, 
                        output_prefix, chunks_saved)
        pbar.close()
        return
    
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        # Try to save the current chunk
        if current_points_colors:
            save_chunks(current_points_colors, current_labels, 
                        output_prefix, chunks_saved)
        pbar.close()
        return
    
    pbar.close()
    
    # Save any remaining chunk
    if current_points_colors:
        print(f"\n=== SAVING FINAL CHUNK ===")
        save_chunks(current_points_colors, current_labels, 
                    output_prefix, chunks_saved)
        chunks_saved += 1
    
    total_time = time.time() - start_time
    print(f"\n=== PROCESSING COMPLETE ===")
    print(f"Total samples generated: {total_samples}")
    print(f"Total chunks saved: {chunks_saved}")
    print(f"Total processing time: {total_time/60:.1f} minutes")
    print(f"Average rate: {total_samples/total_time:.2f} samples/sec")
    
    # If SAVE_CHUNKS is False, save all data at once (NO .h5 extension)
    if not SAVE_CHUNKS and current_points_colors:
        try:
            print("Saving all data at once...")
            final_output = output_prefix  # NO .h5 extension!
            with h5py.File(final_output, 'w') as f:
                f.create_dataset("points", data=np.array(current_points_colors))
                f.create_dataset("labels", data=np.array(current_labels))
            print(f"All data saved successfully to {final_output}")
        except Exception as e:
            print(f"Error saving all data: {e}")

if __name__ == "__main__":
    # Try to install psutil if needed
    try:
        import psutil
    except ImportError:
        print("psutil not found, trying to install...")
        import subprocess
        subprocess.call(['pip', 'install', 'psutil'])
        try:
            import psutil
        except ImportError:
            print("Could not install psutil. Will continue without memory monitoring.")
    
    print("Starting dataset generation...")
    generate_object_segmentation_dataset()
    print("Dataset generation complete!")