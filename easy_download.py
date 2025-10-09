
#!/usr/bin/env python3
"""
Simple Dataset Downloader using opendatasets library.
This handles Kaggle authentication automatically via browser.
"""

import sys
import subprocess
from pathlib import Path

def install_opendatasets():
    """Install opendatasets if not present."""
    try:
        import opendatasets
        return True
    except ImportError:
        print("📦 Installing opendatasets library...")
        subprocess.run([sys.executable, "-m", "pip", "install", "opendatasets", "-q"], check=True)
        return True

def download_dataset():
    """Download the dataset using opendatasets (handles auth via browser)."""
    
    print("=" * 80)
    print("📦 FASHION DATASET DOWNLOADER")
    print("=" * 80)
    print("\n🔐 Authentication Required")
    print("\nYou'll be prompted to enter your Kaggle credentials:")
    print("  - Username: Your Kaggle username")
    print("  - API Key: Get it from https://www.kaggle.com/settings")
    print("           (Click 'Create New API Token' to download kaggle.json)")
    print("           (Open the file and copy the 'key' value)")
    print("\n" + "=" * 80)
    
    # Install if needed
    if not install_opendatasets():
        return False
    
    import opendatasets as od
    
    # Dataset URL
    dataset_url = 'https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset'
    
    # Download location
    data_dir = Path(__file__).parent / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"\n📥 Downloading dataset to: {data_dir}")
        print("⏳ This will take 15-30 minutes (25GB)...\n")
        
        # This will prompt for Kaggle username and key if not cached
        od.download(dataset_url, str(data_dir))
        
        print("\n✅ Download complete!")
        
        # Move files to raw directory
        downloaded_dir = data_dir / 'fashion-product-images-dataset'
        raw_dir = data_dir / 'raw'
        
        if downloaded_dir.exists():
            print(f"\n📂 Moving files to: {raw_dir}")
            import shutil
            if raw_dir.exists():
                shutil.rmtree(raw_dir)
            shutil.move(str(downloaded_dir), str(raw_dir))
            print("✅ Files organized!")
        
        # Verify
        if (raw_dir / 'styles.csv').exists():
            print("\n🔍 Verification:")
            print("   ✅ styles.csv found")
        if (raw_dir / 'images').exists():
            img_count = len(list((raw_dir / 'images').glob('*.jpg')))
            print(f"   ✅ images/ directory ({img_count:,} images)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\n💡 Alternative: Manual download")
        print("1. Go to: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset")
        print("2. Click 'Download'")
        print("3. Extract to: ./data/raw/")
        return False

if __name__ == "__main__":
    success = download_dataset()
    
    if success:
        print("\n" + "=" * 80)
        print("🎉 DATASET IS READY!")
        print("=" * 80)
        print("\n📋 Next Steps:")
        print("   1. python verify_setup.py")
        print("   2. python scripts/preprocess_data.py")
        print("   3. python train.py")
        print("=" * 80)
    
    sys.exit(0 if success else 1)
