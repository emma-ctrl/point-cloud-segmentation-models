# Point Cloud Segmentation Models

Collection of 4 state-of-the-art point cloud segmentation architectures with unified dataset generation pipeline.

## Architectures

- **PointTransformerV3 (PTv3)**
- **PointTransformerV2 (PTv2)**
- **LDGCNN** - Locally Dynamic Graph CNN
- **GAC** - Graph Attention Convolution

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/emma-ctrl/point-cloud-segmentation-models.git
cd point-cloud-segmentation-models
chmod +x setup.sh && ./setup.sh

# 2. Activate environment
conda activate pc-segmentation

# 3. Generate dataset from PCD files
python dataset_generation/generate_hdf5_dataset_with_padding.py

# 4. Create segmentation dataset
python dataset_generation/segmentation_dataset_prep.py

# 5. Train any architecture
python PTv3/train_ptv3.py --dataset-name your_dataset --epochs 100
python PTv2/train_ptv2.py --dataset-name your_dataset --epochs 100
python LDGCNN/train_ldgcnn.py --dataset-name your_dataset --epochs 100
python GAC/train_gac.py --dataset-name your_dataset --epochs 100
```

## Repository Structure

```
point-cloud-segmentation-models/
├── README.md                              # This file
├── environment.yml                        # Conda environment
├── setup.sh                               # Automated setup
├── config.py                              # Shared configuration
├── LICENSE                                # License file
├── dataset_generation/                    # Dataset creation tools
│   ├── generate_hdf5_dataset_with_padding.py    # PCD → HDF5 converter
│   └── segmentation_dataset_prep.py             # Segmentation dataset generator
├── PTv3/                                  # PointTransformerV3
│   ├── train_ptv3.py                      # Training script
│   └── evaluate_ptv3.py                   # Evaluation script
├── PTv2/                                  # PointTransformerV2
│   ├── train_ptv2.py                      # Training script
│   └── evaluate_ptv2.py                   # Evaluation script
├── LDGCNN/                                # Locally Dynamic Graph CNN
│   ├── train_ldgcnn.py                    # Training script
│   └── evaluate_ldgcnn.py                 # Evaluation script
├── GAC/                                   # Graph Attention Convolution
│   ├── train_gac.py                       # Training script
│   └── evaluate_gac.py                    # Evaluation script
├── datasets/                              # Generated datasets
└── results/                              # Training results
    ├── ptv3/                              # PTv3 results
    ├── ptv2/                              # PTv2 results
    ├── ldgcnn/                            # LDGCNN results
    └── gac/                               # GAC results
```

## Installation

### Automated Setup
```bash
chmod +x setup.sh
./setup.sh
conda activate pc-segmentation
```

### Manual Setup
```bash
# Create environment
conda env create -f environment.yml
conda activate pc-segmentation

# Install architecture-specific dependencies
# For PTv3 and PTv2 - requires Pointcept
git clone https://github.com/Pointcept/Pointcept.git
cd Pointcept && pip install -e . && cd ..
```

## Dataset Pipeline

### Step 1: Convert PCD to HDF5
Your PCD files → HDF5 format for training
```bash
python dataset_generation/generate_hdf5_dataset_with_padding.py
```

### Step 2: Generate Segmentation Dataset
Create point cloud scenes with target + alien objects
```bash
python dataset_generation/segmentation_dataset_prep.py
```

### Step 3: Train Models
Each architecture has its own optimized training script:
```bash
python PTv3/train_ptv3.py --dataset-name your_dataset
python PTv2/train_ptv2.py --dataset-name your_dataset
python LDGCNN/train_ldgcnn.py --dataset-name your_dataset
python GAC/train_gac.py --dataset-name your_dataset
```

## Dataset Format

### Input: PCD Files
- Raw point cloud data with RGB
- Multiple objects scanned from different angles
- Organised in directories by object type

### Intermediate: HDF5 Object Files
- `point_clouds`: (N, 2048, 3) - XYZ coordinates
- `color_clouds`: (N, 2048, 3) - RGB colors  
- Fixed 2048 points per object via downsampling/padding

### Final: Segmentation Dataset
- `points`: (N, 20480, 6) - Combined XYZ+RGB
- `labels`: (N, 20480) - Binary labels (0=alien, 1=target)
- Multiple objects per scene for segmentation training