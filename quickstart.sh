#!/bin/bash

# Quick Start Script for AI Fashion Recommendation System
# This script helps you get started quickly

echo "=================================="
echo "🎨 AI Fashion Recommendation System"
echo "=================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first"
    exit 1
fi

echo "✅ Conda found"
echo ""

# Check if AI environment exists
if conda env list | grep -q "^AI "; then
    echo "✅ AI environment exists"
else
    echo "⚠️  AI environment not found. Creating it now..."
    conda create -n AI python=3.10 -y
    echo "✅ AI environment created"
fi

echo ""
echo "What would you like to do?"
echo ""
echo "1) Download dataset from Kaggle"
echo "2) Preprocess data"
echo "3) Train models (full pipeline)"
echo "4) Train only clothing classifier"
echo "5) Train only compatibility model"
echo "6) Run Streamlit demo app"
echo "7) Run example demo"
echo "8) Install/Update dependencies"
echo "9) Exit"
echo ""

read -p "Enter your choice (1-9): " choice

case $choice in
    1)
        echo ""
        echo "📥 Downloading dataset from Kaggle..."
        echo ""
        echo "Make sure you have:"
        echo "  1. Kaggle account"
        echo "  2. API token at ~/.kaggle/kaggle.json"
        echo ""
        read -p "Press Enter to continue or Ctrl+C to cancel..."
        conda run -n AI python scripts/download_dataset.py
        ;;
    2)
        echo ""
        echo "🔄 Preprocessing data..."
        conda run -n AI python scripts/preprocess_data.py \
            --data_dir data/raw \
            --output_dir data/processed \
            --img_size 224
        ;;
    3)
        echo ""
        echo "🚀 Running full training pipeline..."
        echo "⏰ This may take several hours depending on your hardware"
        read -p "Press Enter to continue or Ctrl+C to cancel..."
        conda run -n AI python train.py
        ;;
    4)
        echo ""
        echo "👕 Training clothing classifier only..."
        conda run -n AI python train.py --skip_preprocessing --skip_compatibility
        ;;
    5)
        echo ""
        echo "👔 Training outfit compatibility model only..."
        conda run -n AI python train.py --skip_preprocessing --skip_classifier
        ;;
    6)
        echo ""
        echo "🌐 Starting Streamlit demo app..."
        echo "The app will open in your browser at http://localhost:8501"
        echo ""
        conda run -n AI streamlit run app/streamlit_app.py
        ;;
    7)
        echo ""
        echo "🎬 Running demo..."
        conda run -n AI python demo_example.py
        ;;
    8)
        echo ""
        echo "📦 Installing/Updating dependencies..."
        conda run -n AI pip install -r requirements.txt
        ;;
    9)
        echo ""
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo ""
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "=================================="
echo "✅ Done!"
echo "=================================="
