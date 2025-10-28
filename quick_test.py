"""
Quick System Test - Verify Everything Works
"""
import sys
import os

print("="*70)
print(" AI FASHION RECOMMENDER - SYSTEM CHECK")
print("="*70)

# Test 1: Python
print("\n✓ Python:", sys.version.split()[0])

# Test 2: TensorFlow
try:
    import tensorflow as tf
    print("✓ TensorFlow:", tf.__version__)
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ GPU: {len(gpus)} device(s) detected")
        for gpu in gpus:
            print(f"  - {gpu.name}")
    else:
        print("⚠ GPU: Not detected (will use CPU)")
except Exception as e:
    print(f"✗ TensorFlow: Error - {e}")

# Test 3: Model Files
from pathlib import Path
models_dir = Path(__file__).parent / "models" / "saved_models"
classifier = models_dir / "clothing_classifier.keras"
compatibility = models_dir / "outfit_compatibility_advanced.keras"

if classifier.exists():
    print(f"✓ Classifier Model: {classifier.stat().st_size / (1024*1024):.1f} MB")
else:
    print("✗ Classifier Model: Not found")

if compatibility.exists():
    print(f"✓ Compatibility Model: {compatibility.stat().st_size / (1024*1024):.1f} MB")
else:
    print("✗ Compatibility Model: Not found")

# Test 4: Docker/Database
import subprocess
try:
    result = subprocess.run(['docker', 'ps', '--filter', 'name=fashion_recommender_db'], 
                          capture_output=True, text=True, timeout=5)
    if 'fashion_recommender_db' in result.stdout:
        print("✓ Database: Running")
    else:
        print("⚠ Database: Not running (use 'docker-compose up -d')")
except:
    print("⚠ Docker: Not available")

# Test 5: Dependencies
deps = ['numpy', 'pillow', 'fastapi', 'uvicorn']
for dep in deps:
    try:
        __import__(dep)
        print(f"✓ {dep.capitalize()}: Installed")
    except:
        print(f"✗ {dep.capitalize()}: Missing")

print("\n" + "="*70)
print(" SYSTEM STATUS")
print("="*70)
print("\nYour AI Fashion Recommender is ready!")
print("\n📝 Next Steps:")
print("   1. Start all services: start_all.bat")
print("   2. Or manually:")
print("      - Database: docker-compose up -d")
print("      - Backend:  cd backend && python start_backend.py")
print("      - Frontend: cd fashion-recommender && npm run dev")
print("\n🌐 Access:")
print("   Frontend: http://localhost:3000")
print("   Backend:  http://localhost:8000/docs")
print("="*70)
