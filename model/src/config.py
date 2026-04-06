import os
import torch
import platform

class Config:
    # System
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    
    # Model Architecture
    IMAGE_SIZE = 256
    NUM_CLASSES = 1  # Logic: 0=Real, 1=Fake (Sigmoid output)
    
    # Component Flags
    USE_RGB = True
    USE_FREQ = True
    USE_PATCH = True
    USE_VIT = True
    
    # Training Hyperparameters
    BATCH_SIZE = 32  # Optimized for Mac M4 (Unified Memory)
    EPOCHS = 3
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    NUM_WORKERS = 8  # Leverage M4 Performance Cores
    
    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Paths - Use default or override with environment variable
    # Format: <user's data dir>/Dataset/{Train/Fake, Validation/Real, etc.}
    DATA_DIR_ENV = os.environ.get("DEEPFAKE_DATA_DIR")
    if DATA_DIR_ENV:
        DATA_DIR = os.path.abspath(DATA_DIR_ENV)
    else:
        # Default to project data directory if it exists
        default_data = os.path.join(PROJECT_ROOT, "data")
        if os.path.exists(default_data) and os.listdir(default_data):
            DATA_DIR = default_data
        else:
            # Fallback to creating a data dir in project root
            DATA_DIR = default_data
    
    # Since we are using the root folder, the script will recursively find ALL images
    # in all sub-datasets and split them 80/20 for training/validation.
    TRAIN_DATA_PATH = DATA_DIR 
    TEST_DATA_PATH = DATA_DIR 
    CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
    ACTIVE_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "Mark-V.safetensors")

    @classmethod
    def setup(cls):
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        if os.environ.get("DEEPFAKE_DETECTION_SYSTEM_VERBOSE") == "1":
            print(f"Project initialized at {cls.PROJECT_ROOT}")
            print(f"Using device: {cls.DEVICE}")

if __name__ == "__main__":
    Config.setup()



