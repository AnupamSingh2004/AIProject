"""
Script to download the Fashion Product Images Dataset from Kaggle.

Two options available:
1. Direct Download (Recommended - No API setup needed)
2. Kaggle API (Requires API credentials)

Usage:
    python scripts/download_dataset.py --direct      # Show manual download instructions
    python scripts/download_dataset.py --extract FILE  # Extract downloaded zip
    python scripts/download_dataset.py              # Try Kaggle API (requires setup)
"""

import os
import zipfile
from pathlib import Path
import sys
import argparse

def show_direct_download_instructions():
    """
    Shows instructions for downloading the dataset manually (no Kaggle API needed).
    This is the recommended method for most users.
    """
    print("=" * 80)
    print("📦 DIRECT DATASET DOWNLOAD (Recommended)")
    print("=" * 80)
    print("\n✨ No Kaggle API setup required! Just download directly from the website.\n")
    
    data_dir = Path(__file__).parent.parent / 'data' / 'raw'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("📋 Step-by-Step Instructions:")
    print("\n1️⃣  Open this URL in your browser:")
    print("   👉 https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset")
    
    print("\n2️⃣  Click the 'Download' button (you'll need to sign in/create account)")
    print("   - The file is ~25 GB")
    print("   - It will save as: fashion-product-images-dataset.zip")
    
    print("\n3️⃣  Once downloaded, extract it using this command:")
    print(f"\n   cd {data_dir}")
    print("   unzip ~/Downloads/fashion-product-images-dataset.zip")
    
    print("\n   Or use this script to extract:")
    print(f"   python scripts/download_dataset.py --extract ~/Downloads/fashion-product-images-dataset.zip")
    
    print("\n" + "=" * 80)
    print("Expected directory structure after extraction:")
    print("=" * 80)
    print(f"  {data_dir}/")
    print("  ├── styles.csv         (44k product metadata)")
    print("  └── images/            (44k product images)")
    print("      ├── 1163.jpg")
    print("      ├── 1164.jpg")
    print("      └── ...")
    print("=" * 80)
    
    # Check if already exists
    if (data_dir / 'styles.csv').exists():
        print("\n✅ Dataset already exists at:", data_dir)
        print("   You can skip the download and proceed to preprocessing!")
        return True
    else:
        print(f"\n📁 Target directory: {data_dir}")
        print("   (This directory has been created for you)")
    
    return False

def extract_downloaded_file(zip_path):
    """Extract a manually downloaded zip file."""
    print("=" * 80)
    print("📦 EXTRACTING DATASET")
    print("=" * 80)
    
    zip_file = Path(zip_path).expanduser()
    if not zip_file.exists():
        print(f"\n❌ Error: File not found: {zip_file}")
        print("\nPlease check:")
        print("  1. The file path is correct")
        print("  2. The file exists in that location")
        print(f"\nYou provided: {zip_path}")
        return False
    
    data_dir = Path(__file__).parent.parent / 'data' / 'raw'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Source: {zip_file}")
    print(f"📁 Destination: {data_dir}")
    print(f"📊 File size: {zip_file.stat().st_size / (1024**3):.2f} GB")
    print("\n⏳ Extracting... This will take a few minutes (25 GB of data)")
    print("   Please be patient...\n")
    
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # Get total file count for progress
            total_files = len(zip_ref.namelist())
            print(f"   Extracting {total_files:,} files...")
            
            zip_ref.extractall(data_dir)
        
        print("\n✅ Extraction complete!")
        
        # Verify extracted files
        print("\n🔍 Verifying extracted files...")
        if (data_dir / 'styles.csv').exists():
            print("   ✅ styles.csv found")
        else:
            print("   ⚠️  styles.csv not found")
            
        if (data_dir / 'images').exists():
            img_count = len(list((data_dir / 'images').glob('*.jpg')))
            print(f"   ✅ images/ directory found ({img_count:,} images)")
        else:
            print("   ⚠️  images/ directory not found")
        
        return True
        
    except zipfile.BadZipFile:
        print("\n❌ Error: Invalid or corrupted zip file")
        print("   Please re-download the dataset from Kaggle")
        return False
    except Exception as e:
        print(f"\n❌ Error during extraction: {e}")
        return False

def download_kaggle_dataset():
    """
    Download the Fashion Product Images dataset using Kaggle API.
    Requires Kaggle API credentials to be set up.
    """
    print("=" * 80)
    print("📦 KAGGLE API DOWNLOAD")
    print("=" * 80)
    
    # Check if kaggle credentials exist
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if not kaggle_json.exists():
        print("\n❌ Kaggle API credentials not found!")
        print("\n" + "=" * 80)
        print("💡 RECOMMENDED: Use direct download instead (no setup needed)")
        print("=" * 80)
        print("\nRun: python scripts/download_dataset.py --direct")
        print("\n" + "=" * 80)
        print("Or set up Kaggle API:")
        print("=" * 80)
        print("1. Go to https://www.kaggle.com/settings/account")
        print("2. Click 'Create New API Token' under API section")
        print("3. Place the downloaded kaggle.json at:", kaggle_json)
        print("4. Run: chmod 600", kaggle_json)
        print("5. Run this script again")
        print("=" * 80)
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
    print("")
    print("⏳ Download Progress:")
    print("   This may take 15-30 minutes depending on your internet speed")
    print("   File size: ~25 GB")
    print("")
    
    try:
        import time
        import threading
        
        # Progress indicator
        download_complete = False
        
        def show_progress():
            """Show a simple progress spinner while downloading."""
            spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
            idx = 0
            start_time = time.time()
            while not download_complete:
                elapsed = time.time() - start_time
                mins = int(elapsed // 60)
                secs = int(elapsed % 60)
                print(f"\r   {spinner[idx % len(spinner)]} Downloading... [{mins:02d}:{secs:02d}]", end='', flush=True)
                idx += 1
                time.sleep(0.1)
            print(f"\r   ✅ Download complete! [{mins:02d}:{secs:02d}]")
        
        # Start progress thread
        progress_thread = threading.Thread(target=show_progress, daemon=True)
        progress_thread.start()
        
        # Download the dataset
        kaggle.api.dataset_download_files(
            dataset_name,
            path=str(data_dir),
            unzip=True,
            quiet=False
        )
        
        download_complete = True
        time.sleep(0.2)  # Let progress thread finish
        
        print("\n✅ Dataset downloaded and extracted successfully!")
        
        # List downloaded files
        print("\n📂 Verifying downloaded files:")
        if (data_dir / 'styles.csv').exists():
            print("   ✅ styles.csv")
        if (data_dir / 'images').exists():
            img_count = len(list((data_dir / 'images').glob('*.jpg')))
            print(f"   ✅ images/ ({img_count:,} image files)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\nAlternative: Download manually from:")
        print(f"https://www.kaggle.com/datasets/{dataset_name}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Download Fashion Product Images Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show manual download instructions (recommended, no API setup)
  python scripts/download_dataset.py --direct
  
  # Extract a downloaded zip file
  python scripts/download_dataset.py --extract ~/Downloads/fashion-product-images-dataset.zip
  
  # Use Kaggle API (requires setup)
  python scripts/download_dataset.py
        """
    )
    parser.add_argument(
        '--direct', 
        action='store_true',
        help='Show instructions for direct download from Kaggle website (no API needed)'
    )
    parser.add_argument(
        '--extract', 
        type=str, 
        metavar='ZIP_FILE',
        help='Extract a manually downloaded zip file to data/raw/'
    )
    
    args = parser.parse_args()
    
    # Route to appropriate method
    if args.extract:
        success = extract_downloaded_file(args.extract)
    elif args.direct:
        success = show_direct_download_instructions()
    else:
        success = download_kaggle_dataset()
    
    # Show next steps
    if success:
        print("\n" + "=" * 80)
        print("🎉 DATASET IS READY!")
        print("=" * 80)
        print("\n📋 Next Steps:")
        print("\n1️⃣  Verify your setup:")
        print("   python verify_setup.py")
        print("\n2️⃣  Preprocess the data:")
        print("   python scripts/preprocess_data.py")
        print("\n3️⃣  Train the models:")
        print("   python train.py")
        print("\n" + "=" * 80)
    else:
        if not args.direct:
            print("\n" + "=" * 80)
            print("💡 TIP: Try direct download (easier, no API setup)")
            print("=" * 80)
            print("\nRun: python scripts/download_dataset.py --direct")
            print("=" * 80)
    
    sys.exit(0 if success else 1)
