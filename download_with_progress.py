#!/usr/bin/env python3
"""
Enhanced Dataset Downloader with Progress Bar
Downloads the Fashion Product Images dataset from Kaggle with visual progress.
"""

import os
import sys
import zipfile
from pathlib import Path
import time

def check_kaggle_credentials():
    """Check if Kaggle credentials are set up."""
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    return kaggle_json.exists()

def download_with_progress():
    """Download dataset with progress bar."""
    
    print("=" * 80)
    print("📦 FASHION PRODUCT IMAGES DATASET DOWNLOADER")
    print("=" * 80)
    print("")
    
    # Check credentials
    if not check_kaggle_credentials():
        print("❌ Kaggle credentials not found!")
        print("\nPlease run: ./setup_and_download.sh")
        print("Or manually set up credentials at ~/.kaggle/kaggle.json")
        return False
    
    # Import kaggle
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("📦 Installing kaggle library...")
        os.system(f"{sys.executable} -m pip install kaggle -q")
        from kaggle.api.kaggle_api_extended import KaggleApi
    
    # Try to import tqdm for progress bar
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        print("📦 Installing tqdm for progress bars...")
        os.system(f"{sys.executable} -m pip install tqdm -q")
        try:
            from tqdm import tqdm
            has_tqdm = True
        except:
            has_tqdm = False
    
    # Setup paths
    project_root = Path(__file__).parent
    data_dir = project_root / 'data' / 'raw'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_name = 'paramaggarwal/fashion-product-images-dataset'
    
    print(f"📥 Dataset: {dataset_name}")
    print(f"📁 Destination: {data_dir}")
    print(f"📊 Size: ~25 GB")
    print("")
    print("⏳ Starting download...")
    print("   This will take 15-30 minutes depending on your internet speed")
    print("")
    
    # Initialize API
    api = KaggleApi()
    api.authenticate()
    
    try:
        # Download as zip first (faster, shows progress)
        zip_path = data_dir / 'dataset.zip'
        
        print("📥 Step 1/2: Downloading zip file...")
        print("")
        
        # Download with progress
        api.dataset_download_files(
            dataset_name,
            path=str(data_dir),
            unzip=False,
            quiet=False
        )
        
        # Find the downloaded zip
        zip_files = list(data_dir.glob('*.zip'))
        if not zip_files:
            print("❌ Downloaded file not found!")
            return False
        
        zip_path = zip_files[0]
        file_size = zip_path.stat().st_size / (1024**3)
        
        print(f"\n✅ Download complete! ({file_size:.2f} GB)")
        print("")
        print("📦 Step 2/2: Extracting files...")
        print("")
        
        # Extract with progress
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            members = zip_ref.namelist()
            total_files = len(members)
            
            if has_tqdm:
                # Extract with progress bar
                for member in tqdm(members, desc="   Extracting", unit="files"):
                    zip_ref.extract(member, data_dir)
            else:
                # Extract with simple counter
                print(f"   Extracting {total_files:,} files...")
                for i, member in enumerate(members, 1):
                    zip_ref.extract(member, data_dir)
                    if i % 5000 == 0:
                        print(f"   Progress: {i:,}/{total_files:,} files ({i*100//total_files}%)")
        
        # Remove zip file
        print("")
        print("🗑️  Cleaning up...")
        zip_path.unlink()
        
        print("")
        print("=" * 80)
        print("✅ EXTRACTION COMPLETE!")
        print("=" * 80)
        print("")
        
        # Verify
        print("🔍 Verifying files:")
        if (data_dir / 'styles.csv').exists():
            rows = sum(1 for _ in open(data_dir / 'styles.csv')) - 1
            print(f"   ✅ styles.csv ({rows:,} products)")
        
        if (data_dir / 'images').exists():
            img_count = len(list((data_dir / 'images').glob('*.jpg')))
            print(f"   ✅ images/ directory ({img_count:,} images)")
        
        print("")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Try manual download:")
        print(f"   https://www.kaggle.com/datasets/{dataset_name}")
        return False

if __name__ == "__main__":
    print("")
    success = download_with_progress()
    
    if success:
        print("=" * 80)
        print("🎉 DATASET IS READY!")
        print("=" * 80)
        print("")
        print("📋 Next Steps:")
        print("")
        print("   1. Verify setup:")
        print("      conda activate AI")
        print("      python verify_setup.py")
        print("")
        print("   2. Preprocess data:")
        print("      python scripts/preprocess_data.py")
        print("")
        print("   3. Train models:")
        print("      python train.py")
        print("")
        print("=" * 80)
    
    sys.exit(0 if success else 1)
