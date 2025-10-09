#!/bin/bash

# Automated Kaggle Setup and Download Script
# For user: anupamsingh19

echo "======================================================================"
echo "🔑 KAGGLE API SETUP & DATASET DOWNLOAD"
echo "======================================================================"
echo ""
echo "Username: anupamsingh19"
echo ""
echo "======================================================================"
echo "STEP 1: Get Your API Token"
echo "======================================================================"
echo ""
echo "1. Open this URL in your browser:"
echo "   👉 https://www.kaggle.com/settings"
echo ""
echo "2. Log in to Kaggle if needed"
echo ""
echo "3. Scroll to the 'API' section"
echo ""
echo "4. Click 'Create New API Token'"
echo "   (A file 'kaggle.json' will download to ~/Downloads/)"
echo ""
echo "======================================================================"
echo ""

# Check if kaggle.json already exists in Downloads
if [ -f ~/Downloads/kaggle.json ]; then
    echo "✅ Found kaggle.json in Downloads!"
    echo ""
    echo "Setting up Kaggle credentials..."
    
    # Create .kaggle directory
    mkdir -p ~/.kaggle
    
    # Copy the file
    cp ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
    chmod 600 ~/.kaggle/kaggle.json
    
    echo "✅ Credentials installed!"
    echo ""
    echo "======================================================================"
    echo "🚀 DOWNLOADING DATASET"
    echo "======================================================================"
    echo ""
    echo "📊 Download size: ~25 GB"
    echo "⏱️  Estimated time: 15-30 minutes (depending on internet speed)"
    echo ""
    echo "Starting download with progress tracking..."
    echo ""
    
    # Activate conda and download with progress
    conda run -n AI python -u scripts/download_dataset.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "======================================================================"
        echo "✅ SUCCESS! Dataset downloaded"
        echo "======================================================================"
        echo ""
        echo "Next steps:"
        echo "  1. python verify_setup.py"
        echo "  2. python scripts/preprocess_data.py"
        echo "  3. python train.py"
        echo ""
    else
        echo ""
        echo "======================================================================"
        echo "⚠️  Download needs manual intervention"
        echo "======================================================================"
        echo ""
        echo "Please check the error above and try again."
        echo ""
    fi
    
else
    echo "⚠️  kaggle.json not found in ~/Downloads/"
    echo ""
    echo "Please:"
    echo "  1. Go to https://www.kaggle.com/settings"
    echo "  2. Click 'Create New API Token' in the API section"
    echo "  3. The file will download to ~/Downloads/kaggle.json"
    echo "  4. Then run this script again"
    echo ""
    echo "======================================================================"
    echo "ALTERNATIVE: Manual Setup"
    echo "======================================================================"
    echo ""
    echo "If you have the API key string, you can run:"
    echo ""
    echo "  mkdir -p ~/.kaggle"
    echo "  echo '{\"username\":\"anupamsingh19\",\"key\":\"YOUR_KEY_HERE\"}' > ~/.kaggle/kaggle.json"
    echo "  chmod 600 ~/.kaggle/kaggle.json"
    echo "  python scripts/download_dataset.py"
    echo ""
    echo "(Replace YOUR_KEY_HERE with your actual Kaggle API key)"
    echo ""
fi

echo "======================================================================"
