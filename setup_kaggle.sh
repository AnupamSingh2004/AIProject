#!/bin/bash

# Kaggle API Setup Helper Script
# This script helps you set up Kaggle API credentials

echo "======================================================================="
echo "🔑 Kaggle API Setup Helper"
echo "======================================================================="
echo ""

# Check if kaggle is installed
if ! conda run -n AI python -c "import kaggle" 2>/dev/null; then
    echo "❌ Kaggle package not found. Installing..."
    conda run -n AI pip install kaggle
    echo "✅ Kaggle package installed"
    echo ""
fi

# Check if kaggle.json already exists
if [ -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "✅ Kaggle credentials already configured at ~/.kaggle/kaggle.json"
    echo ""
    echo "Testing credentials..."
    if conda run -n AI kaggle datasets list --max-size 100 > /dev/null 2>&1; then
        echo "✅ Credentials are valid!"
        echo ""
        echo "You can now download the dataset:"
        echo "  python scripts/download_dataset.py"
    else
        echo "❌ Credentials appear to be invalid"
        echo ""
        echo "Please follow these steps:"
        echo "1. Go to https://www.kaggle.com/settings"
        echo "2. Scroll to API section"
        echo "3. Click 'Create New API Token'"
        echo "4. Replace ~/.kaggle/kaggle.json with the downloaded file"
    fi
    exit 0
fi

# Kaggle credentials don't exist, guide user through setup
echo "ℹ️  Kaggle credentials not found. Let's set them up!"
echo ""
echo "======================================================================="
echo "Step 1: Get Your Kaggle API Token"
echo "======================================================================="
echo ""
echo "1. Open this URL in your browser:"
echo "   👉 https://www.kaggle.com/settings"
echo ""
echo "2. Sign in to your Kaggle account (or create one)"
echo ""
echo "3. Scroll down to the 'API' section"
echo ""
echo "4. Click 'Create New API Token'"
echo "   (This will download a file called kaggle.json)"
echo ""
echo "5. Note where the file was downloaded (usually ~/Downloads/)"
echo ""
echo "======================================================================="
echo "Step 2: Install Your Token"
echo "======================================================================="
echo ""

# Create .kaggle directory
mkdir -p ~/.kaggle
echo "✅ Created ~/.kaggle directory"
echo ""

# Ask user for the path to kaggle.json
echo "Now, please provide the path to your downloaded kaggle.json file"
echo "(Press Enter to use default: ~/Downloads/kaggle.json)"
echo -n "Path to kaggle.json: "
read KAGGLE_JSON_PATH

# Use default if user pressed Enter
if [ -z "$KAGGLE_JSON_PATH" ]; then
    KAGGLE_JSON_PATH="$HOME/Downloads/kaggle.json"
fi

# Expand ~ to home directory
KAGGLE_JSON_PATH="${KAGGLE_JSON_PATH/#\~/$HOME}"

# Check if file exists
if [ ! -f "$KAGGLE_JSON_PATH" ]; then
    echo ""
    echo "❌ File not found: $KAGGLE_JSON_PATH"
    echo ""
    echo "Please:"
    echo "1. Make sure you've downloaded kaggle.json from Kaggle"
    echo "2. Note the exact path where it was saved"
    echo "3. Run this script again and provide the correct path"
    echo ""
    echo "Or manually copy the file:"
    echo "  cp /path/to/kaggle.json ~/.kaggle/"
    echo "  chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

# Copy the file
cp "$KAGGLE_JSON_PATH" ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

echo ""
echo "✅ Copied kaggle.json to ~/.kaggle/"
echo "✅ Set proper permissions (600)"
echo ""

# Verify the credentials
echo "======================================================================="
echo "Step 3: Verify Setup"
echo "======================================================================="
echo ""
echo "Testing your Kaggle credentials..."

if conda run -n AI kaggle datasets list --max-size 100 > /dev/null 2>&1; then
    echo "✅ SUCCESS! Your Kaggle API is configured correctly!"
    echo ""
    echo "======================================================================="
    echo "🎉 All Set! Next Steps:"
    echo "======================================================================="
    echo ""
    echo "1. Download the dataset:"
    echo "   python scripts/download_dataset.py"
    echo ""
    echo "2. Or use the quickstart menu:"
    echo "   ./quickstart.sh"
    echo ""
    echo "The dataset is ~25GB and will take 10-30 minutes to download."
else
    echo "❌ Credentials test failed"
    echo ""
    echo "This might be because:"
    echo "1. The kaggle.json file is invalid"
    echo "2. You haven't accepted dataset terms on Kaggle website"
    echo ""
    echo "Please:"
    echo "1. Visit: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset"
    echo "2. Click 'Download' once to accept terms"
    echo "3. Then run: python scripts/download_dataset.py"
fi

echo ""
echo "======================================================================="
