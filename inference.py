#!/usr/bin/env python3
"""
Deepfake Detection System Simple Inference Script
Quick command-line tool for testing deepfake detection on images.
"""

import argparse
import sys
import os
from pathlib import Path

# Add model directory to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "model"))

import torch
import cv2
import numpy as np
import io
import base64
from PIL import Image

# Import model components
from src.models import DeepfakeDetector
from src.config import Config
import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    from safetensors.torch import load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("âš ï¸  Warning: safetensors not available, will use PyTorch format")

# Color codes
colors = {
    'GREEN': '\033[92m',
    'RED': '\033[91m',
    'YELLOW': '\033[93m',
    'BLUE': '\033[94m',
    'RESET': '\033[0m'
}

def print_success(msg):
    print(f"{colors['GREEN']}âœ… {msg}{colors['RESET']}")

def print_error(msg):
    print(f"{colors['RED']}âŒ {msg}{colors['RESET']}")

def print_warning(msg):
    print(f"{colors['YELLOW']}âš ï¸  {msg}{colors['RESET']}")

def print_info(msg):
    print(f"{colors['BLUE']}â„¹ï¸  {msg}{colors['RESET']}")

def get_transform():
    """Create image transformation pipeline"""
    return A.Compose([
        A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def load_model(checkpoint_path, device):
    """Load DeepfakeDetector model"""
    print_info(f"Loading model architecture...")
    model = DeepfakeDetector(pretrained=False)
    model = model.to(device)
    model.eval()
    
    if not os.path.exists(checkpoint_path):
        print_warning(f"Checkpoint not found: {checkpoint_path}")
        print_warning("Using random initialization (model may not work properly)")
        return model

    # Detect Git LFS pointer files (common when downloading source ZIP).
    try:
        if os.path.getsize(checkpoint_path) < 1024 * 1024:
            with open(checkpoint_path, "r", encoding="utf-8", errors="ignore") as f:
                header = f.read(256)
            if "git-lfs.github.com/spec/v1" in header:
                print_error("Checkpoint is a Git LFS pointer, not actual model weights.")
                print_error("Please re-download real Mark-V.safetensors (~193 MB) and replace this file.")
                return model
    except Exception:
        pass
    
    print_info(f"Loading weights from: {checkpoint_path}")
    
    try:
        state_dict = None
        load_errors = []

        if checkpoint_path.endswith(".safetensors") and SAFETENSORS_AVAILABLE:
            try:
                print_info("Trying safetensors format...")
                state_dict = load_file(checkpoint_path)
            except Exception as e:
                load_errors.append(f"safetensors failed: {e}")

        if state_dict is None:
            try:
                print_info("Trying PyTorch format...")
                state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
            except Exception as e:
                load_errors.append(f"torch.load failed: {e}")

        if state_dict is None:
            raise RuntimeError("; ".join(load_errors) if load_errors else "unknown checkpoint load error")
        
        model.load_state_dict(state_dict, strict=True)
        print_success("Model weights loaded successfully")
        
    except RuntimeError as e:
        # Try with key remapping for backward compatibility
        print_warning(f"Direct load failed, attempting key remapping...")
        try:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('rgb_branch.features.'):
                    new_k = k.replace('rgb_branch.features.', 'rgb_branch.net.features.')
                    new_state_dict[new_k] = v
                elif k.startswith('rgb_branch.avgpool.'):
                    new_k = k.replace('rgb_branch.avgpool.', 'rgb_branch.net.avgpool.')
                    new_state_dict[new_k] = v
                else:
                    new_state_dict[k] = v
            
            model.load_state_dict(new_state_dict, strict=False)
            print_success("Model weights loaded (with key remapping)")
        except Exception as e2:
            print_error(f"Failed to load weights: {e}")
            print_error(f"Remapping also failed: {e2}")
    
    return model

def infer_image(model, image_path, device, transform):
    """Run inference on single image"""
    try:
        image_path = Path(image_path).expanduser().resolve()

        # Read image
        print_info(f"Reading image: {image_path}")
        image = cv2.imread(str(image_path))
        
        if image is None:
            try:
                pil_img = Image.open(str(image_path)).convert("RGB")
                image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                print_warning("OpenCV could not read image directly; loaded via PIL fallback.")
            except Exception:
                print_error(f"Could not read image. Check file path and format: {image_path}")
                return None
        
        print_info(f"Image shape: {image.shape}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Transform
        print_info("Applying transforms...")
        augmented = transform(image=image)
        image_tensor = augmented['image'].unsqueeze(0).to(device)
        
        # Inference
        print_info("Running inference...")
        with torch.no_grad():
            logits = model(image_tensor)
            prob = torch.sigmoid(logits).item()
        
        # Result
        is_fake = prob > 0.5
        label = "FAKE" if is_fake else "REAL"
        confidence = prob if is_fake else 1 - prob
        
        return {
            'prediction': label,
            'probability_fake': prob,
            'probability_real': 1 - prob,
            'confidence': confidence,
        }
        
    except Exception as e:
        print_error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def print_results(result):
    """Print formatted results"""
    if result is None:
        print_error("No results to display")
        return
    
    print(f"""
{colors['BLUE']}â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”{colors['RESET']}

{colors['YELLOW']}DETECTION RESULTS:{colors['RESET']}

  Prediction:            {colors['YELLOW'] if result['prediction'] == 'FAKE' else colors['GREEN']}{result['prediction']}{colors['RESET']}
  Confidence:            {result['confidence']*100:.2f}%
  
  Fake Probability:      {result['probability_fake']*100:.2f}%
  Real Probability:      {result['probability_real']*100:.2f}%

{colors['BLUE']}â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”{colors['RESET']}
    """)

def main():
    parser = argparse.ArgumentParser(
        description="Deepfake Detection System Simple Inference - Test deepfake detection on images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py --image img.jpg
  python inference.py --image img.jpg --checkpoint model/results/checkpoints/Mark-V.safetensors
  python inference.py --image img.jpg --device cpu
  python inference.py --image img.jpg --device cuda
        """
    )
    
    parser.add_argument(
        "--image", 
        type=str, 
        required=True,
        help="Path to image file (JPG, PNG, WebP)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(project_root / "model" / "results" / "checkpoints" / "Mark-V.safetensors"),
        help="Path to model checkpoint (default: Mark-V.safetensors)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device to use (auto=detect, default: auto)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    
    # Check image exists
    if not image_path.exists():
        print_error(f"Image not found: {image_path}")
        return 1
    
    # Select device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print_success("Using CUDA GPU")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print_success("Using Apple Metal Performance Shaders (MPS)")
        else:
            device = torch.device("cpu")
            print_warning("Using CPU (slower inference)")
    else:
        device = torch.device(args.device)
        print_info(f"Using device: {device}")
    
    print(f"\n{colors['BLUE']}Deepfake Detection System Inference Tool{colors['RESET']}\n")
    
    # Load model
    print_info(f"Model config: IMAGE_SIZE={Config.IMAGE_SIZE}, DEVICE={device}")
    model = load_model(args.checkpoint, device)
    
    if model is None:
        print_error("Failed to load model")
        return 1
    
    # Get transform
    transform = get_transform()
    
    # Run inference
    print()
    result = infer_image(model, image_path, device, transform)
    
    if result:
        print_results(result)
        return 0
    else:
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


