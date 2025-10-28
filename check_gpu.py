#!/usr/bin/env python3
"""
Quick GPU check script
Run this to verify GPU is accessible
"""

def check_tensorflow_gpu():
    """Check TensorFlow GPU availability."""
    try:
        import tensorflow as tf
        print("=" * 70)
        print("TensorFlow GPU Check")
        print("=" * 70)
        print(f"TensorFlow version: {tf.__version__}")
        print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
        
        gpus = tf.config.list_physical_devices('GPU')
        print(f"\nGPUs detected: {len(gpus)}")
        
        if gpus:
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
                try:
                    details = tf.config.experimental.get_device_details(gpu)
                    print(f"    Compute Capability: {details.get('compute_capability', 'Unknown')}")
                    print(f"    Device Name: {details.get('device_name', 'Unknown')}")
                except Exception as e:
                    print(f"    (Could not get details: {e})")
            
            # Quick computation test
            print("\nTesting GPU computation...")
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
                print(f"✅ GPU computation successful: {c.numpy()}")
            
            return True
        else:
            print("❌ No GPU detected")
            return False
            
    except ImportError:
        print("❌ TensorFlow not installed")
        print("Install with: pip install tensorflow[and-cuda]")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_pytorch_gpu():
    """Check PyTorch GPU availability."""
    try:
        import torch
        print("\n" + "=" * 70)
        print("PyTorch GPU Check")
        print("=" * 70)
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"\nGPU {i}:")
                print(f"  Name: {torch.cuda.get_device_name(i)}")
                print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
                print(f"  Capability: {torch.cuda.get_device_capability(i)}")
            
            # Quick computation test
            print("\nTesting GPU computation...")
            device = torch.device("cuda:0")
            x = torch.randn(3, 3).to(device)
            y = torch.randn(3, 3).to(device)
            z = torch.matmul(x, y)
            print(f"✅ GPU computation successful on device: {z.device}")
            
            return True
        else:
            print("❌ CUDA not available for PyTorch")
            return False
            
    except ImportError:
        print("\n⚠️ PyTorch not installed (optional)")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_nvidia_driver():
    """Check NVIDIA driver using nvidia-smi."""
    import subprocess
    try:
        print("\n" + "=" * 70)
        print("NVIDIA Driver Check")
        print("=" * 70)
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print("❌ nvidia-smi failed")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found")
        print("Make sure NVIDIA drivers are installed")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all GPU checks."""
    print("\n🎮 GPU Detection and Configuration Check\n")
    
    # Check NVIDIA driver
    driver_ok = check_nvidia_driver()
    
    # Check TensorFlow
    tf_ok = check_tensorflow_gpu()
    
    # Check PyTorch (optional)
    pt_ok = check_pytorch_gpu()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"NVIDIA Driver: {'✅ OK' if driver_ok else '❌ Not Found'}")
    print(f"TensorFlow GPU: {'✅ OK' if tf_ok else '❌ Not Available'}")
    if pt_ok is not None:
        print(f"PyTorch GPU: {'✅ OK' if pt_ok else '❌ Not Available'}")
    
    if tf_ok or pt_ok:
        print("\n✅ GPU is ready for training!")
        print("\nRecommendations for RTX GPUs:")
        print("  • Use mixed precision training (FP16) for faster training")
        print("  • Enable XLA compilation for additional speedup")
        print("  • Monitor GPU memory with nvidia-smi")
    else:
        print("\n⚠️ GPU not available - will use CPU")
        print("\nTo enable GPU:")
        print("  1. Install NVIDIA drivers from: https://www.nvidia.com/drivers")
        print("  2. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
        print("  3. Install cuDNN: https://developer.nvidia.com/cudnn")
        print("  4. Install TensorFlow with GPU: pip install tensorflow[and-cuda]")
    
    print("=" * 70)

if __name__ == "__main__":
    main()
