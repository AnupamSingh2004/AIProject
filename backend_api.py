"""
FastAPI Backend for AI Fashion Recommender
Serves the trained AI models via REST API
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import cv2
from pathlib import Path
import sys
import json

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'models'))

# Import AI modules
from src.skin_tone_analyzer import SkinToneAnalyzer
from src.color_analyzer import ColorAnalyzer
from src.recommendation_engine import RecommendationEngine, Occasion, Season
from src.clothing_detector import ClothingDetector, ClothingItem

app = FastAPI(title="Fashion Recommender AI Backend", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI components
skin_analyzer = SkinToneAnalyzer()
color_analyzer = ColorAnalyzer()
clothing_detector = ClothingDetector()
recommendation_engine = RecommendationEngine()

# Pydantic models
class SkinToneResult(BaseModel):
    fitzpatrick_type: str
    undertone: str
    dominant_color: Dict[str, int]
    confidence: float

class ClothingAnalysisResult(BaseModel):
    clothing_type: str
    dominant_color: Dict[str, int]
    secondary_colors: List[Dict[str, int]]
    style: str
    pattern: str
    confidence: float

class OutfitRecommendationRequest(BaseModel):
    skinTone: Optional[Dict[str, Any]]
    wardrobe: List[Dict[str, Any]]
    occasion: str
    season: Optional[str]
    count: int = 10

class OutfitRecommendation(BaseModel):
    top_id: Optional[str]
    bottom_id: Optional[str]
    shoes_id: Optional[str]
    accessories_id: Optional[str]
    compatibility_score: float
    skin_tone_match_score: Optional[float]
    color_harmony_type: Optional[str]
    reason: str

@app.get("/")
async def root():
    return {
        "message": "Fashion Recommender AI Backend",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/analyze-skin-tone", response_model=SkinToneResult)
async def analyze_skin_tone(file: UploadFile = File(...)):
    """
    Analyze skin tone from uploaded photo
    """
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Analyze skin tone
        result = skin_analyzer.analyze(image)
        
        return SkinToneResult(
            fitzpatrick_type=result.fitzpatrick_type.name,
            undertone=result.undertone.name,
            dominant_color={
                "r": int(result.dominant_color[0]),
                "g": int(result.dominant_color[1]),
                "b": int(result.dominant_color[2])
            },
            confidence=float(result.confidence)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/analyze-clothing", response_model=ClothingAnalysisResult)
async def analyze_clothing(file: UploadFile = File(...)):
    """
    Analyze clothing item from uploaded photo
    """
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Analyze clothing
        result = clothing_detector.detect(image)
        
        return ClothingAnalysisResult(
            clothing_type=result.type.name,
            dominant_color={
                "r": int(result.dominant_color[0]),
                "g": int(result.dominant_color[1]),
                "b": int(result.dominant_color[2])
            },
            secondary_colors=[
                {"r": int(c[0]), "g": int(c[1]), "b": int(c[2])}
                for c in result.color_palette[:3]
            ],
            style=result.style.name,
            pattern=result.pattern.name,
            confidence=float(result.confidence)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/recommend-outfits")
async def recommend_outfits(request: OutfitRecommendationRequest):
    """
    Generate outfit recommendations
    """
    try:
        # Convert skin tone data
        skin_tone = None
        if request.skinTone:
            from src.skin_tone_analyzer import SkinToneResult as STResult, FitzpatrickType, Undertone
            
            fitz_type = FitzpatrickType[request.skinTone['fitzpatrick_type']]
            undertone = Undertone[request.skinTone['undertone']]
            
            skin_tone = STResult(
                fitzpatrick_type=fitz_type,
                undertone=undertone,
                dominant_color=(
                    request.skinTone['dominant_color']['r'],
                    request.skinTone['dominant_color']['g'],
                    request.skinTone['dominant_color']['b']
                ),
                color_clusters=[],
                confidence=1.0
            )
        
        # Convert wardrobe items
        wardrobe_items = []
        for item in request.wardrobe:
            from src.clothing_detector import ClothingType, ClothingStyle, Pattern
            
            clothing_item = ClothingItem(
                type=ClothingType[item['category'].upper()] if item.get('category') else ClothingType.TOPWEAR,
                dominant_color=(
                    item['dominant_color']['r'],
                    item['dominant_color']['g'],
                    item['dominant_color']['b']
                ) if item.get('dominant_color') else (128, 128, 128),
                color_palette=[],
                style=ClothingStyle[item['style'].upper()] if item.get('style') else ClothingStyle.CASUAL,
                pattern=Pattern[item['pattern'].upper()] if item.get('pattern') else Pattern.SOLID,
                confidence=1.0
            )
            wardrobe_items.append(clothing_item)
        
        # Generate recommendations
        occasion = Occasion[request.occasion.upper()]
        season = Season[request.season.upper()] if request.season else Season.SPRING
        
        recommendations = recommendation_engine.recommend_outfits(
            wardrobe=wardrobe_items,
            skin_tone=skin_tone,
            occasion=occasion,
            season=season,
            count=request.count
        )
        
        # Convert to response format
        outfits = []
        for rec in recommendations:
            outfit = OutfitRecommendation(
                top_id=None,  # These would map to actual IDs from the wardrobe
                bottom_id=None,
                shoes_id=None,
                accessories_id=None,
                compatibility_score=float(rec.compatibility_score),
                skin_tone_match_score=float(rec.skin_tone_match_score) if rec.skin_tone_match_score else None,
                color_harmony_type=rec.color_harmony_type.value if rec.color_harmony_type else None,
                reason=rec.reason
            )
            outfits.append(outfit)
        
        return {"outfits": [outfit.dict() for outfit in outfits]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
