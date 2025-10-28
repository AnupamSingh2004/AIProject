"""
GPU Configuration for TensorFlow
Ensures optimal GPU utilization for RTX GPUs
"""

import tensorflow as tf
import os

def setup_gpu():
    """Configure TensorFlow for optimal GPU performance."""
    
    print("=" * 70)
    print("🎮 GPU Configuration")
    print("=" * 70)
    
    # Check TensorFlow version
    print(f"\nTensorFlow version: {tf.__version__}")
    
    # Check if TensorFlow was built with CUDA support
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    
    # List available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\nPhysical GPUs detected: {len(gpus)}")
    
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
            
        try:
            # Enable memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ Memory growth enabled")
            
            # Set visible devices
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"Logical GPUs: {len(logical_gpus)}")
            
            # Configure mixed precision for faster training on RTX GPUs
            # RTX 20xx/30xx/40xx series have Tensor Cores that benefit from mixed precision
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print(f"✅ Mixed precision policy set: {policy.name}")
            
            # Print compute capability
            for gpu in gpus:
                details = tf.config.experimental.get_device_details(gpu)
                compute_capability = details.get('compute_capability', 'Unknown')
                print(f"✅ GPU Compute Capability: {compute_capability}")
            
            return True
            
        except RuntimeError as e:
            print(f"❌ GPU configuration error: {e}")
            return False
    else:
        print("⚠️ No GPU detected. Using CPU.")
        print("\nTo use GPU:")
        print("1. Install CUDA 11.8 or 12.x from NVIDIA")
        print("2. Install cuDNN 8.6 or later")
        print("3. Install TensorFlow with GPU support:")
        print("   pip install tensorflow[and-cuda]")
        return False

def test_gpu():
    """Test GPU with a simple operation."""
    print("\n" + "=" * 70)
    print("🧪 GPU Performance Test")
    print("=" * 70)
    
    try:
        # Test GPU computation
        with tf.device('/GPU:0'):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
            
        print("✅ GPU computation successful!")
        print(f"Result shape: {c.shape}")
        
        # Benchmark
        import time
        
        # GPU benchmark
        with tf.device('/GPU:0'):
            start = time.time()
            for _ in range(100):
                c = tf.matmul(a, b)
            gpu_time = time.time() - start
            
        print(f"GPU Time (100 matrix multiplications): {gpu_time:.4f}s")
        
        # CPU benchmark
        with tf.device('/CPU:0'):
            start = time.time()
            for _ in range(100):
                c = tf.matmul(a, b)
            cpu_time = time.time() - start
            
        print(f"CPU Time (100 matrix multiplications): {cpu_time:.4f}s")
        print(f"GPU Speedup: {cpu_time/gpu_time:.2f}x faster")
        
    except RuntimeError as e:
        print(f"❌ GPU test failed: {e}")
        return False
    
    return True

def get_gpu_info():
    """Get detailed GPU information."""
    info = {
        "tensorflow_version": tf.__version__,
        "cuda_available": tf.test.is_built_with_cuda(),
        "gpu_available": tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None),
        "num_gpus": len(tf.config.list_physical_devices('GPU')),
        "gpu_devices": []
    }
    
    gpus = tf.config.list_physical_devices('GPU')
    for i, gpu in enumerate(gpus):
        details = tf.config.experimental.get_device_details(gpu)
        info["gpu_devices"].append({
            "id": i,
            "name": gpu.name,
            "compute_capability": details.get('compute_capability', 'Unknown'),
            "device_name": details.get('device_name', 'Unknown')
        })
    
    return info

if __name__ == "__main__":
    # Setup GPU
    success = setup_gpu()
    
    if success:
        # Test GPU
        test_gpu()
        
        # Print GPU info
        print("\n" + "=" * 70)
        print("📊 GPU Information Summary")
        print("=" * 70)
        info = get_gpu_info()
        import json
        print(json.dumps(info, indent=2))
    
    print("\n" + "=" * 70)
    print("✅ GPU Configuration Complete!")
    print("=" * 70)
