"""
Start Backend API with Loaded Models
Run this to start the FastAPI backend with AI models
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
from pathlib import Path
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import numpy as np
from PIL import Image
import io

# Configure TensorFlow
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"🎮 GPU detected: {len(gpus)} device(s)")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("💻 No GPU detected, using CPU")

app = FastAPI(title="Fashion AI Backend", version="2.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models
clothing_classifier = None
outfit_compatibility_model = None

class ClothingAnalysisResponse(BaseModel):
    clothing_type: str
    dominant_color: Dict[str, int]
    secondary_colors: List[Dict[str, int]]
    pattern: str
    style: str
    confidence: float

@app.on_event("startup")
async def load_models():
    """Load models on startup"""
    global clothing_classifier, outfit_compatibility_model
    
    print("\n" + "="*70)
    print("🚀 Loading AI Models...")
    print("="*70)
    
    models_dir = Path(__file__).parent.parent / 'models' / 'saved_models'
    
    # Load clothing classifier
    classifier_path = models_dir / 'clothing_classifier.keras'
    if classifier_path.exists():
        print(f"\n📦 Loading Clothing Classifier...")
        clothing_classifier = tf.keras.models.load_model(str(classifier_path))
        print(f"✅ Loaded successfully!")
        print(f"   Layers: {len(clothing_classifier.layers)}")
        print(f"   Input: {clothing_classifier.input_shape}")
        print(f"   Output: {clothing_classifier.output_shape}")
    else:
        print(f"⚠️  Classifier not found at {classifier_path}")
    
    # Load compatibility model
    compatibility_path = models_dir / 'outfit_compatibility_advanced.keras'
    if compatibility_path.exists():
        print(f"\n📦 Loading Outfit Compatibility Model...")
        try:
            outfit_compatibility_model = tf.keras.models.load_model(
                str(compatibility_path),
                safe_mode=False  # Allow loading Lambda layers
            )
            print(f"✅ Loaded successfully!")
            print(f"   Layers: {len(outfit_compatibility_model.layers)}")
        except Exception as e:
            print(f"⚠️  Could not load compatibility model: {e}")
    
    print("\n" + "="*70)
    print("✅ Backend Ready!")
    print("="*70 + "\n")

@app.get("/")
async def root():
    return {
        "message": "Fashion AI Backend",
        "version": "2.0.0",
        "status": "running",
        "models": {
            "clothing_classifier": clothing_classifier is not None,
            "outfit_compatibility": outfit_compatibility_model is not None
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "gpu_available": len(tf.config.list_physical_devices('GPU')) > 0,
        "models_loaded": {
            "classifier": clothing_classifier is not None,
            "compatibility": outfit_compatibility_model is not None
        }
    }

@app.post("/api/analyze-clothing")
async def analyze_clothing(file: UploadFile = File(...)):
    """Analyze clothing image with trained model"""
    
    if clothing_classifier is None:
        raise HTTPException(status_code=503, detail="Clothing classifier not loaded")
    
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Resize and normalize
        image = image.convert('RGB')
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Run inference
        predictions = clothing_classifier.predict(img_array, verbose=0)
        
        # Get top prediction
        top_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][top_idx])
        
        # Category mapping (from your training data)
        categories = ['Topwear', 'Bottomwear', 'Footwear', 'Dress', 'Accessories', 'Outerwear']
        clothing_type = categories[top_idx] if top_idx < len(categories) else 'Unknown'
        
        # Extract dominant colors (simple method)
        img_rgb = np.array(image)
        dominant = np.mean(img_rgb.reshape(-1, 3), axis=0).astype(int)
        
        return ClothingAnalysisResponse(
            clothing_type=clothing_type,
            dominant_color={
                "r": int(dominant[0]),
                "g": int(dominant[1]),
                "b": int(dominant[2])
            },
            secondary_colors=[],
            pattern="solid",
            style="casual",
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/test-model")
async def test_model():
    """Test model with random input"""
    
    if clothing_classifier is None:
        return {"error": "Model not loaded"}
    
    # Create random test image
    test_image = np.random.rand(1, 224, 224, 3).astype(np.float32)
    
    import time
    start = time.time()
    predictions = clothing_classifier.predict(test_image, verbose=0)
    inference_time = (time.time() - start) * 1000
    
    return {
        "model_loaded": True,
        "inference_time_ms": inference_time,
        "output_shape": predictions.shape,
        "prediction_sample": predictions[0].tolist()[:3]
    }

if __name__ == "__main__":
    print("\n" + "="*70)
    print("👕 Fashion AI Backend Server")
    print("="*70)
    print(f"Starting server at http://localhost:8000")
    print(f"API docs at http://localhost:8000/docs")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
