#!/usr/bin/env python3
"""
Deepfake Detection System Automated Setup Script
Handles environment setup, dependency installation, and system verification.
"""

import os
import sys
import subprocess
import shutil
import platform
import argparse
from pathlib import Path

# Colors for terminal output
colors = {
    'GREEN': '\033[92m',
    'RED': '\033[91m',
    'YELLOW': '\033[93m',
    'BLUE': '\033[94m',
    'CYAN': '\033[96m',
    'RESET': '\033[0m'
}

def print_step(message: str, step: int = None):
    """Print a formatted step message"""
    if step:
        print(f"\n{colors['CYAN']}[Step {step}] {message}{colors['RESET']}")
    else:
        print(f"\n{colors['CYAN']}{message}{colors['RESET']}")

def print_success(message: str):
    """Print a success message"""
    print(f"{colors['GREEN']}âœ… {message}{colors['RESET']}")

def print_error(message: str):
    """Print an error message"""
    print(f"{colors['RED']}âŒ {message}{colors['RESET']}")

def print_warning(message: str):
    """Print a warning message"""
    print(f"{colors['YELLOW']}âš ï¸  {message}{colors['RESET']}")

def print_info(message: str):
    """Print an info message"""
    print(f"{colors['BLUE']}â„¹ï¸  {message}{colors['RESET']}")

def check_python_version():
    """Check if Python version is compatible"""
    print_step("Checking Python Version", step=1)
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    print_info(f"Current Python: {version_str}")
    
    if version.major != 3 or version.minor < 9:
        print_error(f"Python 3.9+ required (you have {version.major}.{version.minor})")
        print_warning("Please install Python 3.9 or 3.10 from https://www.python.org/")
        return False
    
    if version.minor >= 12:
        print_warning(f"Python {version.major}.{version.minor} may have compatibility issues")
        print_warning("Recommended: Python 3.9 or 3.10")
    
    print_success(f"Python {version_str} is compatible")
    return True

def check_gpu():
    """Check for GPU availability"""
    print_step("Checking for GPU", step=2)
    
    try:
        import torch
        if torch.cuda.is_available():
            print_success(f"CUDA available: {torch.cuda.get_device_name(0)}")
            print_info(f"CUDA version: {torch.version.cuda}")
            return True
        else:
            print_warning("No CUDA GPU detected. Will use CPU (slower inference)")
            return False
    except ImportError:
        print_warning("PyTorch not installed yet. GPU check will run after installation.")
        return None

def create_venv(force_recreate: bool = False):
    """Create virtual environment"""
    print_step("Creating Virtual Environment", step=3)
    
    venv_path = Path("deepguard_env")
    
    if venv_path.exists():
        print_warning(f"Virtual environment already exists at {venv_path}")
        if force_recreate:
            shutil.rmtree(venv_path)
            print_info("Removed existing virtual environment")
        else:
            print_info("Using existing virtual environment")
            return venv_path
    
    print_info("Creating new virtual environment...")
    try:
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
        print_success(f"Virtual environment created at {venv_path}")
        return venv_path
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to create virtual environment: {e}")
        return None

def get_venv_python(venv_path: Path):
    """Get path to Python executable in venv"""
    if platform.system() == "Windows":
        return (venv_path / "Scripts" / "python.exe").resolve()
    else:
        return (venv_path / "bin" / "python").resolve()

def install_dependencies(venv_python: Path):
    """Install project dependencies"""
    print_step("Installing Dependencies", step=4)
    
    req_file = Path("backend") / "requirements_web.txt"
    if not req_file.exists():
        print_error(f"Requirements file not found: {req_file}")
        return False
    
    print_info(f"Installing from {req_file}...")
    try:
        # First upgrade pip
        subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
        print_success("Upgraded pip")
        
        # Install requirements
        subprocess.run([str(venv_python), "-m", "pip", "install", "-r", str(req_file)], check=True)
        print_success("All dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {e}")
        return False

def initialize_database(venv_python: Path):
    """Initialize SQLite database"""
    print_step("Initializing Database", step=5)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        result = subprocess.run(
            [str(venv_python), "-c", "import database; database.init_db()"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(Path("backend").resolve()),
            check=True
        )
        print(result.stdout)
        print_success("Database initialized")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to initialize database: {e}")
        print_error(f"Details: {e.stderr}")
        return False

def verify_model_weights():
    """Verify Mark-V model weights exist"""
    print_step("Verifying Model Weights", step=6)
    
    model_path = Path("model/results/checkpoints/Mark-V.safetensors")
    
    if model_path.exists():
        size_bytes = model_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)

        if size_bytes < 1024 * 1024:
            try:
                header = model_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                header = ""

            if "git-lfs.github.com/spec/v1" in header:
                print_error(f"Found Git LFS pointer instead of real weights: {model_path}")
                print_warning("This usually happens when downloading ZIP without LFS objects.")
                print_warning("Re-clone with Git LFS or replace file with real Mark-V.safetensors (~193 MB).")
                return False

        print_success(f"Model weights found: {model_path}")
        print_info(f"Model size: {size_mb:.1f} MB")
        return True
    else:
        print_warning(f"Model weights not found at {model_path}")
        print_warning("Will attempt to download or use random initialization on first run")
        return False

def test_model_loading(venv_python: Path):
    """Test if model can be loaded"""
    print_step("Testing Model Loading", step=7)
    
    test_script = """
import sys
import os
sys.path.insert(0, os.path.abspath('../model'))

import torch
from src.models import DeepfakeDetector
from src.config import Config

try:
    print(f"Device: {Config.DEVICE}")
    model = DeepfakeDetector(pretrained=False)
    model.eval()
    print("Model architecture loaded successfully")
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model(dummy_input)
    print("Model forward pass successful")
    print(f"Output shape: {output.shape}")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
"""
    
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        result = subprocess.run(
            [str(venv_python), "-c", test_script],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(Path("backend").resolve()),
            check=True
        )
        print(result.stdout)
        print_success("Model loading test passed")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Model loading test failed: {e}")
        print_error(f"Details: {e.stderr}")
        return False

def print_next_steps(venv_path: Path):
    """Print next steps for user"""
    print_step("Setup Complete! ðŸŽ‰", step=None)
    
    if platform.system() == "Windows":
        activate_cmd = f"deepguard_env\\Scripts\\Activate.ps1"
    else:
        activate_cmd = f"source deepguard_env/bin/activate"
    
    print(f"""
{colors['CYAN']}â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”{colors['RESET']}

{colors['GREEN']}âœ… Deepfake Detection System is ready to run!{colors['RESET']}

To start the server, run these commands:

1. Activate virtual environment:
   {activate_cmd}

2. Start the Flask server:
   cd backend
   python app.py

3. Open your browser:
   http://localhost:7860

â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”

{colors['YELLOW']}Optional: Test with Command-Line Inference{colors['RESET']}

cd model
python src/inference.py --source /path/to/image.jpg --checkpoints results/checkpoints/Mark-V.safetensors

â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”

For detailed setup guide, see: SETUP_GUIDE.md
    """)

def main():
    """Run all setup steps"""
    parser = argparse.ArgumentParser(description="Deepfake Detection System setup helper")
    parser.add_argument("--yes", action="store_true", help="Run non-interactively and reuse existing venv")
    parser.add_argument("--recreate-venv", action="store_true", help="Delete and recreate the virtual environment")
    args = parser.parse_args()

    print(f"""
{colors['CYAN']}
â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
â•‘           Deepfake Detection System Automated Setup Script                    â•‘
â•‘          AI-Powered Deepfake Detection System                 â•‘
â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
{colors['RESET']}
    """)
    
    # Step 1: Check Python version
    if not check_python_version():
        print_error("Python version check failed. Exiting.")
        return False
    
    # Step 2: Check GPU
    check_gpu()
    
    # Step 3: Create venv
    venv_path = create_venv(force_recreate=args.recreate_venv)
    if not venv_path:
        print_error("Failed to create virtual environment. Exiting.")
        return False
    
    venv_python = get_venv_python(venv_path)
    print_info(f"Using Python: {venv_python}")
    
    # Step 4: Install dependencies
    if not install_dependencies(venv_python):
        print_error("Failed to install dependencies. Exiting.")
        return False
    
    # Step 5: Initialize database
    if not initialize_database(venv_python):
        print_warning("Database initialization had issues, but setup will continue")
    
    # Step 6: Verify model weights
    verify_model_weights()
    
    # Step 7: Test model loading
    if not test_model_loading(venv_python):
        print_warning("Model loading test failed")
    
    # Print next steps
    print_next_steps(venv_path)
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print_error("\nSetup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)



