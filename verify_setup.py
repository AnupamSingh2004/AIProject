"""
Setup Verification and Model Training Guide
Checks that everything is properly configured before training.
"""

import sys
import os
from pathlib import Path
import subprocess


def check_conda_environment():
    """Check if AI conda environment exists and has required packages."""
    print("\n" + "=" * 70)
    print("1. Checking Conda Environment")
    print("=" * 70)
    
    try:
        # Check if we're in the AI environment
        env_name = os.environ.get('CONDA_DEFAULT_ENV', '')
        if env_name == 'AI':
            print(f"✅ Running in AI conda environment")
        else:
            print(f"⚠️  Not in AI environment (current: {env_name})")
            print("   Run: conda activate AI")
            return False
    except Exception as e:
        print(f"❌ Error checking conda environment: {e}")
        return False
    
    return True


def check_required_packages():
    """Check if all required packages are installed."""
    print("\n" + "=" * 70)
    print("2. Checking Required Packages")
    print("=" * 70)
    
    required_packages = {
        'tensorflow': 'TensorFlow',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'PIL': 'Pillow',
        'mediapipe': 'MediaPipe',
        'streamlit': 'Streamlit',
        'matplotlib': 'Matplotlib',
    }
    
    all_installed = True
    
    for package, name in required_packages.items():
        try:
            if package == 'cv2':
                import cv2
                version = cv2.__version__
            elif package == 'sklearn':
                import sklearn
                version = sklearn.__version__
            elif package == 'PIL':
                from PIL import Image
                version = 'installed'
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
            
            print(f"✅ {name}: {version}")
        except ImportError:
            print(f"❌ {name}: NOT INSTALLED")
            all_installed = False
    
    if not all_installed:
        print("\n⚠️  Missing packages detected!")
        print("   Run: pip install -r requirements.txt")
    
    return all_installed


def check_dataset():
    """Check if dataset is downloaded and preprocessed."""
    print("\n" + "=" * 70)
    print("3. Checking Dataset")
    print("=" * 70)
    
    data_dir = Path('data/raw')
    processed_dir = Path('data/processed')
    
    # Check raw data
    styles_csv = data_dir / 'styles.csv'
    images_dir = data_dir / 'images'
    
    if styles_csv.exists():
        print(f"✅ styles.csv found at {styles_csv}")
        
        # Count images
        if images_dir.exists():
            num_images = len(list(images_dir.glob('*.jpg')))
            print(f"✅ Images directory found with {num_images} images")
        else:
            print(f"⚠️  Images directory not found at {images_dir}")
            return False
    else:
        print(f"❌ styles.csv not found at {styles_csv}")
        print("   Run: python scripts/download_dataset.py")
        return False
    
    # Check processed data
    train_csv = processed_dir / 'train.csv'
    if train_csv.exists():
        print(f"✅ Preprocessed data found at {processed_dir}")
    else:
        print(f"⚠️  Preprocessed data not found")
        print("   Run: python scripts/preprocess_data.py")
        return False
    
    return True


def check_project_structure():
    """Check if project structure is correct."""
    print("\n" + "=" * 70)
    print("4. Checking Project Structure")
    print("=" * 70)
    
    required_dirs = [
        'src',
        'models',
        'data/raw',
        'data/processed',
        'data/user_uploads',
        'models/saved_models',
        'scripts',
        'app',
    ]
    
    all_exist = True
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - NOT FOUND")
            all_exist = False
    
    return all_exist


def check_training_scripts():
    """Check if training scripts exist."""
    print("\n" + "=" * 70)
    print("5. Checking Training Scripts")
    print("=" * 70)
    
    scripts = {
        'train.py': 'Main training pipeline',
        'scripts/preprocess_data.py': 'Data preprocessing',
        'models/clothing_classifier.py': 'Clothing classification model',
        'models/outfit_compatibility_model.py': 'Outfit compatibility model',
    }
    
    all_exist = True
    
    for script, description in scripts.items():
        path = Path(script)
        if path.exists():
            print(f"✅ {script} - {description}")
        else:
            print(f"❌ {script} - NOT FOUND")
            all_exist = False
    
    return all_exist


def print_next_steps(all_checks_passed):
    """Print next steps based on verification results."""
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    if all_checks_passed:
        print("\n🎉 All checks passed! You're ready to train the models.")
        print("\n📋 Next Steps:")
        print("\n1. Quick Start (Recommended):")
        print("   ./quickstart.sh")
        print("   Then select option 3 (Train models)")
        
        print("\n2. Manual Training (Step by Step):")
        print("   a) Preprocess data:")
        print("      conda run -n AI python scripts/preprocess_data.py")
        
        print("\n   b) Train clothing classifier:")
        print("      conda run -n AI python models/clothing_classifier.py")
        
        print("\n   c) Train outfit compatibility model:")
        print("      conda run -n AI python models/outfit_compatibility_model.py")
        
        print("\n3. Full Pipeline (One Command):")
        print("   conda run -n AI python train.py")
        
        print("\n4. After Training, Run Demo App:")
        print("   conda run -n AI streamlit run app/streamlit_app.py")
        
        print("\n💡 Training Tips:")
        print("   - First training may take 2-4 hours depending on hardware")
        print("   - GPU is highly recommended (10-20x faster)")
        print("   - Start with smaller batch size if you get OOM errors")
        print("   - Monitor with: watch -n 1 nvidia-smi (for GPU)")
        
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\n📋 Common Fixes:")
        print("\n1. Activate environment:")
        print("   conda activate AI")
        
        print("\n2. Install dependencies:")
        print("   pip install -r requirements.txt")
        
        print("\n3. Download dataset:")
        print("   python scripts/download_dataset.py")
        
        print("\n4. Create missing directories:")
        print("   mkdir -p data/raw data/processed models/saved_models")


def main():
    """Main verification script."""
    print("=" * 70)
    print("🔍 AI Fashion Recommendation System - Setup Verification")
    print("=" * 70)
    
    checks = {
        'conda_env': check_conda_environment(),
        'packages': check_required_packages(),
        'dataset': check_dataset(),
        'structure': check_project_structure(),
        'scripts': check_training_scripts(),
    }
    
    all_passed = all(checks.values())
    
    print_next_steps(all_passed)
    
    print("\n" + "=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
