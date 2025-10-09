"""
Script to download the Fashion Product Images Dataset from Kaggle.
You'll need to set up your Kaggle API credentials first.

Instructions:
1. Create a Kaggle account at https://www.kaggle.com
2. Go to https://www.kaggle.com/settings/account
3. Scroll to "API" section and click "Create New API Token"
4. This will download kaggle.json
5. Place it at ~/.kaggle/kaggle.json
6. Run: chmod 600 ~/.kaggle/kaggle.json
"""

import os
import zipfile
from pathlib import Path
import sys

def download_kaggle_dataset():
    """Download the Fashion Product Images dataset from Kaggle."""
    
    # Check if kaggle credentials exist
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if not kaggle_json.exists():
        print("❌ Kaggle API credentials not found!")
        print("\nPlease follow these steps:")
        print("1. Go to https://www.kaggle.com/settings/account")
        print("2. Click 'Create New API Token' under API section")
        print("3. Place the downloaded kaggle.json at:", kaggle_json)
        print("4. Run: chmod 600", kaggle_json)
        return False
    
    # Import kaggle after checking credentials
    try:
        import kaggle
    except ImportError:
        print("❌ Kaggle library not installed. Installing...")
        os.system("pip install kaggle")
        import kaggle
    
    # Define paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data' / 'raw'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_name = 'paramaggarwal/fashion-product-images-dataset'
    
    print(f"📥 Downloading dataset: {dataset_name}")
    print(f"📁 Destination: {data_dir}")
    
    try:
        # Download the dataset
        kaggle.api.dataset_download_files(
            dataset_name,
            path=str(data_dir),
            unzip=True
        )
        
        print("✅ Dataset downloaded successfully!")
        
        # List downloaded files
        print("\n📂 Downloaded files:")
        for item in data_dir.iterdir():
            print(f"  - {item.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\nAlternative: Download manually from:")
        print(f"https://www.kaggle.com/datasets/{dataset_name}")
        return False

if __name__ == "__main__":
    success = download_kaggle_dataset()
    sys.exit(0 if success else 1)
