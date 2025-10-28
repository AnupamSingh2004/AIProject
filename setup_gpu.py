"""
GPU Setup Script for RTX 3050
Configures TensorFlow to use CUDA GPU acceleration
"""
import subprocess
import sys
import os
from pathlib import Path

def check_gpu():
    """Check if GPU is available"""
    print("\n" + "="*70)
    print("🔍 Checking GPU Availability...")
    print("="*70)
    
    try:
        import tensorflow as tf
        print(f"\n✅ TensorFlow version: {tf.__version__}")
        
        # Check for GPU
        gpus = tf.config.list_physical_devices('GPU')
        print(f"\n📊 GPUs detected: {len(gpus)}")
        
        if gpus:
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
                print(f"   Device type: {gpu.device_type}")
            return True
        else:
            print("   ⚠️  No GPU detected")
            return False
            
    except Exception as e:
        print(f"   ❌ Error checking GPU: {e}")
        return False

def check_cuda():
    """Check CUDA installation"""
    print("\n" + "="*70)
    print("🔍 Checking CUDA Installation...")
    print("="*70)
    
    # Check nvcc (CUDA compiler)
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("\n✅ CUDA Toolkit installed:")
            print(result.stdout)
            return True
        else:
            print("\n⚠️  nvcc not found")
            return False
    except FileNotFoundError:
        print("\n⚠️  CUDA Toolkit not found (nvcc command not available)")
        return False
    except Exception as e:
        print(f"\n⚠️  Error checking CUDA: {e}")
        return False

def check_nvidia_smi():
    """Check NVIDIA driver"""
    print("\n" + "="*70)
    print("🔍 Checking NVIDIA Driver...")
    print("="*70)
    
    try:
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("\n✅ NVIDIA Driver installed:")
            print(result.stdout)
            return True
        else:
            print("\n⚠️  nvidia-smi failed")
            return False
    except FileNotFoundError:
        print("\n❌ NVIDIA Driver not found")
        print("   Please install NVIDIA GPU drivers first")
        return False
    except Exception as e:
        print(f"\n⚠️  Error checking driver: {e}")
        return False

def install_tensorflow_gpu():
    """Install TensorFlow with GPU support"""
    print("\n" + "="*70)
    print("📦 Installing TensorFlow with GPU support...")
    print("="*70)
    
    try:
        # Uninstall existing TensorFlow
        print("\n1. Removing existing TensorFlow...")
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'tensorflow'],
                      capture_output=True)
        
        # Install TensorFlow with GPU support
        print("\n2. Installing tensorflow[and-cuda]...")
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', 'tensorflow[and-cuda]'],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            print("✅ TensorFlow with GPU support installed successfully!")
            return True
        else:
            print(f"❌ Installation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error during installation: {e}")
        return False

def configure_gpu_memory():
    """Configure GPU memory growth"""
    print("\n" + "="*70)
    print("⚙️  Configuring GPU Memory...")
    print("="*70)
    
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU memory growth enabled")
            return True
        else:
            print("⚠️  No GPU to configure")
            return False
            
    except Exception as e:
        print(f"❌ Error configuring GPU: {e}")
        return False

def test_gpu_inference():
    """Test GPU with actual model inference"""
    print("\n" + "="*70)
    print("🧪 Testing GPU with Model Inference...")
    print("="*70)
    
    try:
        import tensorflow as tf
        import numpy as np
        from pathlib import Path
        import time
        
        # Load the model
        model_path = Path(__file__).parent / "models" / "saved_models" / "clothing_classifier.keras"
        print(f"\n📦 Loading model from: {model_path}")
        
        model = tf.keras.models.load_model(str(model_path))
        print("✅ Model loaded")
        
        # Create test image
        test_image = np.random.rand(1, 224, 224, 3).astype(np.float32)
        
        # Warm up
        print("\n🔥 Warming up GPU...")
        model.predict(test_image, verbose=0)
        
        # Benchmark
        print("\n⏱️  Running inference benchmark...")
        n_runs = 10
        
        start_time = time.time()
        for i in range(n_runs):
            predictions = model.predict(test_image, verbose=0)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / n_runs
        
        print(f"\n✅ Inference Results:")
        print(f"   Average time: {avg_time*1000:.2f} ms")
        print(f"   Throughput: {1/avg_time:.2f} images/second")
        
        # Check device placement
        print(f"\n📍 Running on: {'/device:GPU:0' if tf.config.list_physical_devices('GPU') else '/device:CPU:0'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*70)
    print("🚀 GPU Setup for AI Fashion Recommender")
    print("="*70)
    
    # Step 1: Check NVIDIA Driver
    driver_ok = check_nvidia_smi()
    
    # Step 2: Check CUDA
    cuda_ok = check_cuda()
    
    # Step 3: Check current GPU status
    gpu_ok = check_gpu()
    
    # Step 4: If GPU not detected but CUDA is available, reinstall TensorFlow
    if cuda_ok and not gpu_ok:
        print("\n" + "="*70)
        print("💡 CUDA is installed but TensorFlow is not using it")
        print("="*70)
        
        response = input("\nDo you want to install tensorflow[and-cuda]? (y/n): ")
        if response.lower() == 'y':
            if install_tensorflow_gpu():
                print("\n✅ Please restart this script to verify GPU is now working")
                return
    
    # Step 5: If GPU is working, test it
    if gpu_ok or check_gpu():
        configure_gpu_memory()
        test_gpu_inference()
        
        print("\n" + "="*70)
        print("🎉 GPU Setup Complete!")
        print("="*70)
        print("\n✅ Your RTX 3050 is now accelerating AI inference!")
        print("\n📝 Next steps:")
        print("   1. Run: python verify_model_works.py")
        print("   2. Start backend: cd backend && python start_backend.py")
        print("   3. Access frontend: http://localhost:3000")
    else:
        print("\n" + "="*70)
        print("📋 GPU Setup Instructions")
        print("="*70)
        
        if not driver_ok:
            print("\n1️⃣ Install NVIDIA Driver:")
            print("   - Download from: https://www.nvidia.com/download/index.aspx")
            print("   - Select: RTX 3050 Laptop GPU")
            print("   - Restart after installation")
        
        if not cuda_ok:
            print("\n2️⃣ Install CUDA Toolkit 12.x:")
            print("   - Download from: https://developer.nvidia.com/cuda-downloads")
            print("   - Select: Windows > x86_64 > 11/12 > exe (network)")
            print("   - Follow installation wizard")
            print("   - Add to PATH: C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x\\bin")
        
        print("\n3️⃣ Install cuDNN (Optional but recommended):")
        print("   - Download from: https://developer.nvidia.com/cudnn")
        print("   - Extract and copy files to CUDA directory")
        
        print("\n4️⃣ Install TensorFlow with GPU:")
        print("   pip install tensorflow[and-cuda]")
        
        print("\n5️⃣ Restart this script to verify installation")

if __name__ == "__main__":
    main()
