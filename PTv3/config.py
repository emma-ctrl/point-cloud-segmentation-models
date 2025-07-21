"""
Configuration for PTv3 Point Cloud Segmentation
"""

from pathlib import Path


class Config:
    # Paths
    PROJECT_ROOT = Path(__file__).parent
    DATASET_PATH = PROJECT_ROOT / "datasets"
    RESULTS_PATH = PROJECT_ROOT / "results"
    POINTCEPT_PATH = PROJECT_ROOT / "Pointcept"
    
    # Dataset settings
    NUM_CLASSES = 2
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        cls.DATASET_PATH.mkdir(parents=True, exist_ok=True)
        cls.RESULTS_PATH.mkdir(parents=True, exist_ok=True)
        (cls.RESULTS_PATH / "ptv3").mkdir(parents=True, exist_ok=True)