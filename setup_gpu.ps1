# GPU Setup Script for RTX GPU
# This script installs TensorFlow and PyTorch with CUDA support

Write-Host "🚀 Setting up GPU environment for RTX GPU..." -ForegroundColor Green

# Check NVIDIA GPU
Write-Host "`n📊 Checking NVIDIA GPU..." -ForegroundColor Cyan
nvidia-smi

# Install TensorFlow with GPU support
Write-Host "`n📦 Installing TensorFlow with GPU support..." -ForegroundColor Cyan
pip install tensorflow[and-cuda]

# Install PyTorch with CUDA support
Write-Host "`n📦 Installing PyTorch with CUDA support..." -ForegroundColor Cyan
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other ML dependencies
Write-Host "`n📦 Installing additional dependencies..." -ForegroundColor Cyan
pip install opencv-python opencv-contrib-python mediapipe scikit-learn scikit-image
pip install colorthief matplotlib seaborn pandas numpy
pip install fastapi uvicorn python-multipart pillow

Write-Host "`n✅ Installation complete!" -ForegroundColor Green

# Test GPU availability
Write-Host "`n🧪 Testing GPU availability..." -ForegroundColor Cyan
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); gpus = tf.config.list_physical_devices('GPU'); print(f'GPUs available: {len(gpus)}'); [print(f'  - {gpu.name}') for gpu in gpus]"

Write-Host "`n✅ GPU setup complete!" -ForegroundColor Green
