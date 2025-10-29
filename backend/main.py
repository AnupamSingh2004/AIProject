"""
FastAPI Backend for AI Fashion Recommendation System
Provides endpoints for:
- Skin tone analysis
- Clothing classification
- Outfit recommendations
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
import io
from PIL import Image
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys
import cv2
import json
import os

# Configure TensorFlow for GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU

# Set memory growth to avoid OOM errors
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
    except RuntimeError as e:
        print(f"⚠️ GPU configuration error: {e}")
else:
    print("⚠️ No GPU found, using CPU")

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))
sys.path.append(str(Path(__file__).parent.parent))

from src.skin_tone_analyzer import SkinToneAnalyzer
from src.clothing_detector import ClothingDetector, ClothingType, Pattern, ClothingStyle
from src.color_analyzer import ColorAnalyzer
from src.recommendation_engine import RecommendationEngine, Occasion, Season

app = FastAPI(title="Fashion AI Backend", version="1.0.0")

# CORS middleware for NextJS frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://nextjs_app:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
clothing_classifier = None
outfit_compatibility_model = None
skin_tone_analyzer = None
clothing_detector = None
color_analyzer = None
recommendation_engine = None
label_mapping = None

def load_models():
    """Load all AI models on startup."""
    global clothing_classifier, outfit_compatibility_model, skin_tone_analyzer
    global clothing_detector, color_analyzer, recommendation_engine, label_mapping
    
    try:
        # Check and configure GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ Found {len(gpus)} GPU(s) - enabling for use")
            for gpu in gpus:
                print(f"   - {gpu.name}")
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            print("⚠️  No GPU found - using CPU")
        
        models_dir = Path(__file__).parent.parent / 'models' / 'saved_models'
        
        # Load clothing classifier
        classifier_path = models_dir / 'clothing_classifier.keras'
        if classifier_path.exists():
            print("📦 Loading clothing classifier...")
            clothing_classifier = tf.keras.models.load_model(str(classifier_path))
            print("✅ Clothing classifier loaded!")
        else:
            print("⚠️ Clothing classifier not found")
        
        # Load outfit compatibility model
        compatibility_path = models_dir / 'outfit_compatibility_advanced.keras'
        if compatibility_path.exists():
            print("📦 Loading outfit compatibility model...")
            outfit_compatibility_model = tf.keras.models.load_model(str(compatibility_path))
            print("✅ Outfit compatibility model loaded!")
        else:
            print("⚠️ Outfit compatibility model not found")
        
        # Load label mapping
        data_dir = Path(__file__).parent.parent / 'data' / 'processed'
        label_mapping_path = data_dir / 'label_mapping.json'
        if label_mapping_path.exists():
            with open(label_mapping_path, 'r') as f:
                label_mapping = json.load(f)
            print(f"✅ Label mapping loaded: {len(label_mapping)} categories")
        
        # Initialize analyzers (if dependencies are available)
        print("📦 Initializing analyzers...")
        try:
            skin_tone_analyzer = SkinToneAnalyzer()
            clothing_detector = ClothingDetector()
            color_analyzer = ColorAnalyzer()
            recommendation_engine = RecommendationEngine()
            print("✅ All analyzers initialized!")
        except Exception as e:
            print(f"⚠️  Analyzers not fully available: {e}")
            print("   Models will still work for basic operations")
        
        return True
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return False

# Request/Response Models
class SkinToneResponse(BaseModel):
    fitzpatrick_type: str
    undertone: str
    dominant_color: Dict[str, int]
    dominant_color_hex: str
    confidence: float
    recommendations: List[str]

class ClothingAnalysisResponse(BaseModel):
    clothing_type: str
    dominant_color: Dict[str, int]
    secondary_colors: List[Dict[str, int]]
    pattern: str
    style: str
    confidence: float

class OutfitRecommendationRequest(BaseModel):
    skin_tone_hex: Optional[str] = None
    occasion: str
    season: Optional[str] = "spring"
    wardrobe_items: List[str]  # List of item IDs from database

class OutfitRecommendationResponse(BaseModel):
    outfits: List[Dict]
    scores: List[float]

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    print("🚀 Starting Fashion AI Backend...")
    load_models()
    print("✅ Backend ready!")

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Fashion AI Backend API",
        "status": "running",
        "models_loaded": {
            "clothing_classifier": clothing_classifier is not None,
            "outfit_compatibility": outfit_compatibility_model is not None,
            "analyzers": skin_tone_analyzer is not None
        }
    }

@app.post("/api/analyze-skin-tone", response_model=SkinToneResponse)
async def analyze_skin_tone(photo: UploadFile = File(...)):
    """Analyze skin tone from uploaded photo."""
    try:
        # Read and save image temporarily
        image_bytes = await photo.read()
        temp_path = "/tmp/temp_skin_analysis.jpg"
        
        # Save image
        with open(temp_path, "wb") as f:
            f.write(image_bytes)
        
        # Analyze skin tone
        if skin_tone_analyzer is None:
            raise HTTPException(status_code=503, detail="Skin tone analyzer not available")
        
        result = skin_tone_analyzer.analyze(temp_path)
        
        if result is None:
            raise HTTPException(status_code=400, detail="Could not detect face in image")
        
        # Generate recommendations based on skin tone
        recommendations = []
        if result.undertone.value == "warm":
            recommendations = [
                "Warm earth tones work beautifully with your skin",
                "Try oranges, yellows, and warm browns",
                "Avoid cool blues and stark whites"
            ]
        elif result.undertone.value == "cool":
            recommendations = [
                "Cool tones complement your skin perfectly",
                "Blues, purples, and pinks are your best friends",
                "Avoid warm oranges and yellows"
            ]
        else:
            recommendations = [
                "You can wear both warm and cool colors",
                "Experiment with various color combinations",
                "Most colors will look great on you"
            ]
        
        return SkinToneResponse(
            fitzpatrick_type=result.fitzpatrick_type.name,
            undertone=result.undertone.value,
            dominant_color={
                "r": result.dominant_color_rgb[0],
                "g": result.dominant_color_rgb[1],
                "b": result.dominant_color_rgb[2]
            },
            dominant_color_hex=result.dominant_color_hex,
            confidence=result.confidence,
            recommendations=recommendations
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in skin tone analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/analyze-clothing", response_model=ClothingAnalysisResponse)
async def analyze_clothing(file: UploadFile = File(...)):
    """Analyze clothing item from uploaded image."""
    try:
        # Read and convert image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert('RGB')
        
        # Resize for classifier
        img_array = np.array(image.resize((224, 224)))
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Classify using trained model
        if clothing_classifier is None:
            raise HTTPException(status_code=503, detail="Clothing classifier not available")
        
        predictions = clothing_classifier.predict(img_array, verbose=0)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class])
        
        # Map prediction to category
        category_names = ['Accessories', 'Bottomwear', 'Dress', 'Footwear', 'Other', 'Topwear']
        if label_mapping:
            # Use label mapping if available
            category_name = next((k for k, v in label_mapping.items() if v == predicted_class), 'Other')
        else:
            category_name = category_names[predicted_class] if predicted_class < len(category_names) else 'Other'
        
        # Extract colors using K-means
        img_resized = image.resize((100, 100))  # Smaller for faster processing
        img_np = np.array(img_resized)
        pixels = img_np.reshape(-1, 3)
        
        # K-means clustering for dominant colors
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=min(4, len(pixels)), random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        colors = kmeans.cluster_centers_.astype(int)
        dominant_color = colors[0]
        secondary_colors = [{"r": int(c[0]), "g": int(c[1]), "b": int(c[2])} for c in colors[1:4]]
        
        # Detect pattern using edge detection
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Simple pattern classification
        if edge_density < 0.05:
            pattern = "solid"
        elif edge_density > 0.20:
            pattern = "textured"
        else:
            # Check for lines
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=5)
            if lines is not None and len(lines) > 10:
                # Analyze line directions
                angles = []
                for line in lines[:20]:
                    x1, y1, x2, y2 = line[0]
                    angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    angles.append(angle)
                
                horizontal = sum(1 for a in angles if a < 15 or a > 165)
                vertical = sum(1 for a in angles if 75 < a < 105)
                
                if horizontal > len(angles) * 0.6:
                    pattern = "striped_horizontal"
                elif vertical > len(angles) * 0.6:
                    pattern = "striped_vertical"
                else:
                    pattern = "checkered"
            else:
                pattern = "solid"
        
        # Determine style based on category and colors
        brightness = np.mean(img_np)
        if brightness > 200:
            style = "casual"
        elif brightness < 80:
            style = "formal"
        else:
            style = "casual"
        
        return ClothingAnalysisResponse(
            clothing_type=category_name,
            dominant_color={
                "r": int(dominant_color[0]),
                "g": int(dominant_color[1]),
                "b": int(dominant_color[2])
            },
            secondary_colors=secondary_colors,
            pattern=pattern,
            style=style,
            confidence=confidence
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in clothing analysis: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/recommend-outfits", response_model=OutfitRecommendationResponse)
async def recommend_outfits(request: OutfitRecommendationRequest):
    """Generate outfit recommendations based on wardrobe items."""
    try:
        # This endpoint would integrate with the database to fetch actual wardrobe items
        # For now, return a basic response
        
        if recommendation_engine is None:
            raise HTTPException(status_code=503, detail="Recommendation engine not available")
        
        # TODO: Integrate with database to fetch actual wardrobe items
        # For now, return placeholder
        
        return OutfitRecommendationResponse(
            outfits=[],
            scores=[]
        )
    except Exception as e:
        print(f"Error in outfit recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": {
            "clothing_classifier": clothing_classifier is not None,
            "outfit_compatibility": outfit_compatibility_model is not None,
            "skin_tone_analyzer": skin_tone_analyzer is not None,
            "clothing_detector": clothing_detector is not None,
            "color_analyzer": color_analyzer is not None,
            "recommendation_engine": recommendation_engine is not None
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
