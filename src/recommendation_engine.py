"""
Recommendation Engine Module
Combines skin tone analysis, color theory, and clothing database to recommend outfits.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import random

from skin_tone_analyzer import SkinToneResult, FitzpatrickType, Undertone
from clothing_detector import ClothingItem, ClothingType, ClothingStyle
from color_analyzer import ColorAnalyzer, ColorHarmony, ColorMatch

class Occasion(Enum):
    """Types of occasions."""
    CASUAL = "casual"
    FORMAL = "formal"
    BUSINESS = "business"
    PARTY = "party"
    DATE = "date"
    ATHLETIC = "athletic"
    BEACH = "beach"

class Season(Enum):
    """Seasons for outfit recommendations."""
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"

@dataclass
class OutfitRecommendation:
    """Represents a complete outfit recommendation."""
    top: ClothingItem
    bottom: ClothingItem
    shoes: Optional[ClothingItem]
    accessories: Optional[List[ClothingItem]]
    compatibility_score: float
    reason: str
    skin_tone_match_score: float
    color_harmony_type: ColorHarmony
    
    def __str__(self):
        outfit_str = f"Outfit Recommendation (Score: {self.compatibility_score:.2%})\n"
        outfit_str += "=" * 60 + "\n"
        outfit_str += f"\nTop:\n{self.top}\n"
        outfit_str += f"\nBottom:\n{self.bottom}\n"
        if self.shoes:
            outfit_str += f"\nShoes:\n{self.shoes}\n"
        outfit_str += f"\nColor Harmony: {self.color_harmony_type.value}\n"
        outfit_str += f"Skin Tone Match: {self.skin_tone_match_score:.2%}\n"
        outfit_str += f"\nReason: {self.reason}\n"
        return outfit_str

class RecommendationEngine:
    """
    Main recommendation engine that combines all components to suggest outfits.
    """
    
    def __init__(self):
        """Initialize the recommendation engine."""
        self.color_analyzer = ColorAnalyzer()
        
        # Define complementary colors for each Fitzpatrick type
        self.fitzpatrick_color_guidelines = {
            FitzpatrickType.TYPE_I: {
                'best_colors': [(0, 0, 139), (70, 130, 180), (255, 182, 193), (221, 160, 221)],  # Deep blues, pinks, purples
                'avoid_colors': [(255, 255, 0), (255, 165, 0)],  # Bright yellows, oranges
            },
            FitzpatrickType.TYPE_II: {
                'best_colors': [(0, 128, 128), (100, 149, 237), (255, 192, 203), (230, 230, 250)],  # Teals, corals, lavenders
                'avoid_colors': [(255, 255, 0), (255, 69, 0)],  # Neon colors
            },
            FitzpatrickType.TYPE_III: {
                'best_colors': [(0, 100, 0), (165, 42, 42), (218, 165, 32), (106, 90, 205)],  # Greens, browns, golds
                'avoid_colors': [(255, 255, 255)],  # Stark white
            },
            FitzpatrickType.TYPE_IV: {
                'best_colors': [(255, 69, 0), (218, 112, 214), (0, 191, 255), (34, 139, 34)],  # Bright oranges, orchids, vibrant blues
                'avoid_colors': [(128, 128, 128)],  # Dull grays
            },
            FitzpatrickType.TYPE_V: {
                'best_colors': [(255, 215, 0), (255, 20, 147), (0, 255, 255), (255, 255, 255)],  # Gold, hot pink, cyan, white
                'avoid_colors': [(0, 0, 0)],  # Black can be too harsh
            },
            FitzpatrickType.TYPE_VI: {
                'best_colors': [(255, 255, 0), (255, 0, 255), (0, 255, 0), (255, 255, 255)],  # Vibrant yellows, magentas, whites
                'avoid_colors': [(139, 69, 19)],  # Muddy browns
            },
        }
    
    def recommend_outfits(self,
                         skin_tone: SkinToneResult,
                         wardrobe: List[ClothingItem],
                         occasion: Occasion = Occasion.CASUAL,
                         season: Season = Season.SPRING,
                         count: int = 5) -> List[OutfitRecommendation]:
        """
        Generate outfit recommendations based on skin tone and wardrobe.
        
        Args:
            skin_tone: Skin tone analysis result
            wardrobe: List of available clothing items
            occasion: Occasion for the outfit
            season: Current season
            count: Number of recommendations to generate
            
        Returns:
            List of outfit recommendations sorted by compatibility score
        """
        if not wardrobe or len(wardrobe) < 2:
            print("Warning: Need at least 2 items in wardrobe to create outfits")
            return []
        
        # Separate items by type
        tops = [item for item in wardrobe if item.item_type in [
            ClothingType.SHIRT, ClothingType.T_SHIRT, ClothingType.BLOUSE,
            ClothingType.SWEATER, ClothingType.HOODIE
        ]]
        
        bottoms = [item for item in wardrobe if item.item_type in [
            ClothingType.PANTS, ClothingType.JEANS, ClothingType.SHORTS, ClothingType.SKIRT
        ]]
        
        dresses = [item for item in wardrobe if item.item_type == ClothingType.DRESS]
        
        shoes = [item for item in wardrobe if item.item_type == ClothingType.SHOES]
        
        outfits = []
        
        # Generate outfit combinations
        # Case 1: Dress outfits
        for dress in dresses:
            shoe = random.choice(shoes) if shoes else None
            
            score, harmony, reason, skin_match = self._score_outfit(
                top=dress,
                bottom=None,
                skin_tone=skin_tone,
                occasion=occasion,
                season=season
            )
            
            outfits.append(OutfitRecommendation(
                top=dress,
                bottom=dress,  # Use dress for both
                shoes=shoe,
                accessories=None,
                compatibility_score=score,
                reason=reason,
                skin_tone_match_score=skin_match,
                color_harmony_type=harmony
            ))
        
        # Case 2: Top + Bottom combinations
        for top in tops:
            for bottom in bottoms:
                shoe = random.choice(shoes) if shoes else None
                
                score, harmony, reason, skin_match = self._score_outfit(
                    top=top,
                    bottom=bottom,
                    skin_tone=skin_tone,
                    occasion=occasion,
                    season=season
                )
                
                outfits.append(OutfitRecommendation(
                    top=top,
                    bottom=bottom,
                    shoes=shoe,
                    accessories=None,
                    compatibility_score=score,
                    reason=reason,
                    skin_tone_match_score=skin_match,
                    color_harmony_type=harmony
                ))
        
        # Sort by compatibility score
        outfits.sort(key=lambda x: x.compatibility_score, reverse=True)
        
        return outfits[:count]
    
    def _score_outfit(self,
                     top: ClothingItem,
                     bottom: Optional[ClothingItem],
                     skin_tone: SkinToneResult,
                     occasion: Occasion,
                     season: Season) -> Tuple[float, ColorHarmony, str, float]:
        """
        Score an outfit combination.
        
        Args:
            top: Top clothing item (or dress)
            bottom: Bottom clothing item (or None for dress)
            skin_tone: Skin tone analysis
            occasion: Occasion
            season: Season
            
        Returns:
            Tuple of (compatibility_score, harmony_type, reason, skin_match_score)
        """
        scores = []
        reasons = []
        
        # 1. Skin tone compatibility (30% weight)
        skin_match = self._score_skin_tone_match(top.dominant_color, skin_tone)
        scores.append(skin_match * 0.3)
        
        if skin_match > 0.7:
            reasons.append("Excellent color match for your skin tone")
        
        # 2. Color harmony (30% weight)
        if bottom:
            color_harmony_score, harmony_type = self._score_color_harmony(
                top.dominant_color, bottom.dominant_color
            )
            scores.append(color_harmony_score * 0.3)
            
            if color_harmony_score > 0.8:
                reasons.append(f"Colors create {harmony_type.value} harmony")
        else:
            harmony_type = ColorHarmony.MONOCHROMATIC
            scores.append(0.25)
        
        # 3. Occasion appropriateness (20% weight)
        occasion_score = self._score_occasion_match(top, bottom, occasion)
        scores.append(occasion_score * 0.2)
        
        if occasion_score > 0.7:
            reasons.append(f"Perfect for {occasion.value} occasions")
        
        # 4. Seasonal appropriateness (20% weight)
        seasonal_score = self._score_seasonal_match(top, bottom, season, skin_tone)
        scores.append(seasonal_score * 0.2)
        
        # Calculate total score
        total_score = sum(scores)
        
        # Generate overall reason
        if not reasons:
            reasons.append("A good basic combination")
        
        reason = ". ".join(reasons) + "."
        
        return total_score, harmony_type, reason, skin_match
    
    def _score_skin_tone_match(self, color: Tuple[int, int, int], 
                               skin_tone: SkinToneResult) -> float:
        """Score how well a color matches the skin tone."""
        fitzpatrick = skin_tone.fitzpatrick_type
        
        if fitzpatrick not in self.fitzpatrick_color_guidelines:
            return 0.5  # Neutral score
        
        guidelines = self.fitzpatrick_color_guidelines[fitzpatrick]
        
        # Check if color is in best colors
        min_distance_best = min(
            self.color_analyzer.calculate_color_distance(color, best_color)
            for best_color in guidelines['best_colors']
        )
        
        # Check if color is in avoid colors
        min_distance_avoid = min(
            [self.color_analyzer.calculate_color_distance(color, avoid_color)
             for avoid_color in guidelines['avoid_colors']]
        ) if guidelines['avoid_colors'] else 1.0
        
        # Score based on distances
        if min_distance_best < 0.3:
            return 0.9  # Very close to recommended
        elif min_distance_avoid < 0.2:
            return 0.3  # Close to avoided color
        else:
            return 0.6  # Neutral
    
    def _score_color_harmony(self, color1: Tuple[int, int, int],
                            color2: Tuple[int, int, int]) -> Tuple[float, ColorHarmony]:
        """Score color harmony between two colors."""
        # Get complementary
        comp = self.color_analyzer.get_complementary_color(color1)
        comp_distance = self.color_analyzer.calculate_color_distance(color2, comp)
        
        if comp_distance < 0.2:
            return 0.95, ColorHarmony.COMPLEMENTARY
        
        # Get analogous
        analogous = self.color_analyzer.get_analogous_colors(color1, count=3)
        min_analogous_dist = min(
            self.color_analyzer.calculate_color_distance(color2, analog)
            for analog in analogous
        )
        
        if min_analogous_dist < 0.15:
            return 0.90, ColorHarmony.ANALOGOUS
        
        # Check if monochromatic (similar colors)
        direct_distance = self.color_analyzer.calculate_color_distance(color1, color2)
        
        if direct_distance < 0.15:
            return 0.85, ColorHarmony.MONOCHROMATIC
        
        # Check for neutral colors
        if self.color_analyzer.is_neutral_color(color1) or \
           self.color_analyzer.is_neutral_color(color2):
            return 0.80, ColorHarmony.MONOCHROMATIC
        
        # Default moderate harmony
        return 0.60, ColorHarmony.ANALOGOUS
    
    def _score_occasion_match(self, top: ClothingItem,
                             bottom: Optional[ClothingItem],
                             occasion: Occasion) -> float:
        """Score how well the outfit matches the occasion."""
        # Map occasions to appropriate styles
        occasion_style_map = {
            Occasion.CASUAL: [ClothingStyle.CASUAL, ClothingStyle.STREETWEAR],
            Occasion.FORMAL: [ClothingStyle.FORMAL, ClothingStyle.BUSINESS],
            Occasion.BUSINESS: [ClothingStyle.BUSINESS, ClothingStyle.FORMAL, ClothingStyle.MINIMALIST],
            Occasion.PARTY: [ClothingStyle.BOHEMIAN, ClothingStyle.VINTAGE, ClothingStyle.STREETWEAR],
            Occasion.DATE: [ClothingStyle.CASUAL, ClothingStyle.BOHEMIAN, ClothingStyle.VINTAGE],
            Occasion.ATHLETIC: [ClothingStyle.ATHLETIC],
            Occasion.BEACH: [ClothingStyle.CASUAL, ClothingStyle.BOHEMIAN],
        }
        
        appropriate_styles = occasion_style_map.get(occasion, [ClothingStyle.CASUAL])
        
        if top.style in appropriate_styles:
            if not bottom or bottom.style in appropriate_styles:
                return 0.9
            return 0.7
        
        return 0.5
    
    def _score_seasonal_match(self, top: ClothingItem,
                             bottom: Optional[ClothingItem],
                             season: Season,
                             skin_tone: SkinToneResult) -> float:
        """Score seasonal appropriateness."""
        # Get seasonal palette
        season_name = season.value
        undertone = skin_tone.undertone.value
        
        seasonal_colors = self.color_analyzer.get_seasonal_palette(season_name, undertone)
        
        # Check if colors are in seasonal palette
        top_seasonal = min(
            self.color_analyzer.calculate_color_distance(top.dominant_color, seasonal_color)
            for seasonal_color in seasonal_colors
        )
        
        if top_seasonal < 0.3:
            return 0.85
        
        return 0.6


def test_recommendation_engine():
    """Test the recommendation engine."""
    print("Recommendation Engine initialized successfully!")
    print("\nThis module combines:")
    print("  - Skin tone analysis")
    print("  - Color theory")
    print("  - Clothing database")
    print("\nTo provide personalized outfit recommendations!")


if __name__ == "__main__":
    test_recommendation_engine()
