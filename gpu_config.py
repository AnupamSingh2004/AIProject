"""
GPU Configuration and Model Loader
Optimizes TensorFlow and PyTorch to use RTX GPU
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

def setup_tensorflow_gpu():
    """Configure TensorFlow to use GPU efficiently."""
    try:
        import tensorflow as tf
        
        # Check for GPU
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"✅ Found {len(gpus)} GPU(s):")
            for gpu in gpus:
                print(f"   - {gpu.name}")
            
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"✅ Memory growth enabled for {gpu.name}")
                except RuntimeError as e:
                    print(f"⚠️  Could not set memory growth: {e}")
            
            # Set GPU as visible device
            tf.config.set_visible_devices(gpus[0], 'GPU')
            
            # Enable mixed precision for faster training on RTX GPUs
            try:
                from tensorflow.keras import mixed_precision
                policy = mixed_precision.Policy('mixed_float16')
                mixed_precision.set_global_policy(policy)
                print("✅ Mixed precision enabled (float16)")
            except Exception as e:
                print(f"⚠️  Could not enable mixed precision: {e}")
            
            return True
        else:
            print("⚠️  No GPU found, using CPU")
            return False
            
    except ImportError:
        print("❌ TensorFlow not installed")
        return False
    except Exception as e:
        print(f"❌ Error setting up GPU: {e}")
        return False

def setup_pytorch_gpu():
    """Configure PyTorch to use GPU efficiently."""
    try:
        import torch
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"✅ Found {device_count} CUDA device(s):")
            for i in range(device_count):
                print(f"   - {torch.cuda.get_device_name(i)}")
                print(f"     Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
            
            # Set default device
            torch.cuda.set_device(0)
            print(f"✅ Default CUDA device set to: {torch.cuda.get_device_name(0)}")
            
            # Enable TF32 for better performance on RTX 30 series
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("✅ TF32 enabled for faster computation")
            
            # Enable cuDNN auto-tuner
            torch.backends.cudnn.benchmark = True
            print("✅ cuDNN auto-tuner enabled")
            
            return True
        else:
            print("⚠️  CUDA not available, using CPU")
            return False
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    except Exception as e:
        print(f"❌ Error setting up GPU: {e}")
        return False

def get_device_info():
    """Get detailed GPU information."""
    info = {
        'tensorflow_gpu': False,
        'pytorch_gpu': False,
        'gpu_name': None,
        'gpu_memory': None
    }
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            info['tensorflow_gpu'] = True
            info['gpu_name'] = gpus[0].name
    except:
        pass
    
    try:
        import torch
        if torch.cuda.is_available():
            info['pytorch_gpu'] = True
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    except:
        pass
    
    return info

if __name__ == "__main__":
    print("=" * 70)
    print("🎮 GPU Configuration Check")
    print("=" * 70)
    
    print("\n📊 TensorFlow GPU Setup:")
    print("-" * 70)
    tf_gpu = setup_tensorflow_gpu()
    
    print("\n📊 PyTorch GPU Setup:")
    print("-" * 70)
    pt_gpu = setup_pytorch_gpu()
    
    print("\n" + "=" * 70)
    if tf_gpu or pt_gpu:
        print("✅ GPU is ready for deep learning!")
    else:
        print("⚠️  No GPU detected. Models will run on CPU (slower).")
    print("=" * 70)
