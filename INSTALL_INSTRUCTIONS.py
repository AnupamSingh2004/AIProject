# Simplified GPU-Optimized Backend
# This version works with Python 3.13 and RTX GPU

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Install required packages (run once)
# pip install tensorflow opencv-python pillow numpy scikit-learn fast API uvicorn python-multipart

print("📦 Installing required packages...")
print("Run: pip install tensorflow opencv-python pillow numpy scikit-learn matplotlib pandas fastapi uvicorn python-multipart")
print("")
print("For GPU support, TensorFlow 2.20+ automatically detects CUDA if available.")
print("Your RTX 3050 GPU will be used automatically once TensorFlow is installed.")
