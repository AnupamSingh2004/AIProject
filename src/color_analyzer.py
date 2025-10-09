"""
Color Analyzer Module
Implements color theory principles for fashion recommendations.
"""

import numpy as np
from typing import Dict, List, Tuple, Set
from enum import Enum
from dataclasses import dataclass
import colorsys

class ColorHarmony(Enum):
    """Types of color harmony."""
    COMPLEMENTARY = "complementary"
    ANALOGOUS = "analogous"
    TRIADIC = "triadic"
    SPLIT_COMPLEMENTARY = "split_complementary"
    TETRADIC = "tetradic"
    MONOCHROMATIC = "monochromatic"

@dataclass
class ColorMatch:
    """Represents a color matching recommendation."""
    recommended_color: Tuple[int, int, int]
    harmony_type: ColorHarmony
    compatibility_score: float
    reason: str
    
    def color_hex(self) -> str:
        """Get color as hex string."""
        r, g, b = self.recommended_color
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def __str__(self):
        return (f"Color: {self.color_hex()}\n"
                f"Harmony: {self.harmony_type.value}\n"
                f"Score: {self.compatibility_score:.2%}\n"
                f"Reason: {self.reason}")

class ColorAnalyzer:
    """Analyzes colors and provides matching recommendations based on color theory."""
    
    # Define seasonal color palettes
    SEASONAL_PALETTES = {
        'spring': {
            'warm': True,
            'colors': [
                (255, 229, 180),  # Peach
                (255, 218, 185),  # Warm beige
                (255, 179, 186),  # Light coral
                (173, 216, 230),  # Light blue
                (152, 251, 152),  # Pale green
                (255, 228, 225),  # Warm pink
            ]
        },
        'summer': {
            'warm': False,
            'colors': [
                (176, 224, 230),  # Powder blue
                (221, 160, 221),  # Plum
                (230, 230, 250),  # Lavender
                (240, 248, 255),  # Cool white
                (192, 192, 192),  # Cool gray
                (255, 182, 193),  # Light pink
            ]
        },
        'autumn': {
            'warm': True,
            'colors': [
                (139, 69, 19),    # Saddle brown
                (160, 82, 45),    # Sienna
                (218, 165, 32),   # Goldenrod
                (188, 143, 143),  # Rosy brown
                (128, 128, 0),    # Olive
                (205, 133, 63),   # Peru
            ]
        },
        'winter': {
            'warm': False,
            'colors': [
                (0, 0, 0),        # Black
                (255, 255, 255),  # Pure white
                (220, 20, 60),    # Crimson
                (25, 25, 112),    # Midnight blue
                (128, 0, 128),    # Purple
                (192, 192, 192),  # Silver
            ]
        }
    }
    
    def __init__(self):
        """Initialize the color analyzer."""
        pass
    
    def get_complementary_color(self, rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Get complementary color (opposite on color wheel).
        
        Args:
            rgb: Input color in RGB
            
        Returns:
            Complementary color in RGB
        """
        # Convert to HSV
        r, g, b = [x / 255.0 for x in rgb]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        
        # Shift hue by 180 degrees (0.5 in normalized space)
        h_comp = (h + 0.5) % 1.0
        
        # Convert back to RGB
        r_comp, g_comp, b_comp = colorsys.hsv_to_rgb(h_comp, s, v)
        
        return tuple(int(x * 255) for x in [r_comp, g_comp, b_comp])
    
    def get_analogous_colors(self, rgb: Tuple[int, int, int], count: int = 2) -> List[Tuple[int, int, int]]:
        """
        Get analogous colors (adjacent on color wheel).
        
        Args:
            rgb: Input color in RGB
            count: Number of analogous colors to generate
            
        Returns:
            List of analogous colors in RGB
        """
        r, g, b = [x / 255.0 for x in rgb]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        
        analogous = []
        shift = 30 / 360  # 30 degrees shift
        
        for i in range(1, count + 1):
            # Create colors on both sides
            h1 = (h + i * shift) % 1.0
            h2 = (h - i * shift) % 1.0
            
            for h_new in [h1, h2]:
                r_new, g_new, b_new = colorsys.hsv_to_rgb(h_new, s, v)
                analogous.append(tuple(int(x * 255) for x in [r_new, g_new, b_new]))
        
        return analogous[:count]
    
    def get_triadic_colors(self, rgb: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """
        Get triadic colors (120 degrees apart on color wheel).
        
        Args:
            rgb: Input color in RGB
            
        Returns:
            List of two triadic colors in RGB
        """
        r, g, b = [x / 255.0 for x in rgb]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        
        triadic = []
        
        for shift in [120/360, 240/360]:
            h_new = (h + shift) % 1.0
            r_new, g_new, b_new = colorsys.hsv_to_rgb(h_new, s, v)
            triadic.append(tuple(int(x * 255) for x in [r_new, g_new, b_new]))
        
        return triadic
    
    def get_monochromatic_colors(self, rgb: Tuple[int, int, int], count: int = 3) -> List[Tuple[int, int, int]]:
        """
        Get monochromatic colors (same hue, different saturation/value).
        
        Args:
            rgb: Input color in RGB
            count: Number of variations to generate
            
        Returns:
            List of monochromatic colors in RGB
        """
        r, g, b = [x / 255.0 for x in rgb]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        
        monochromatic = []
        
        for i in range(count):
            # Vary saturation and value
            s_new = max(0.2, min(1.0, s + (i - count//2) * 0.2))
            v_new = max(0.3, min(1.0, v + (i - count//2) * 0.15))
            
            r_new, g_new, b_new = colorsys.hsv_to_rgb(h, s_new, v_new)
            monochromatic.append(tuple(int(x * 255) for x in [r_new, g_new, b_new]))
        
        return monochromatic
    
    def get_matching_colors(self, base_color: Tuple[int, int, int],
                           harmony_types: List[ColorHarmony] = None) -> List[ColorMatch]:
        """
        Get all matching colors based on color harmony principles.
        
        Args:
            base_color: Base color in RGB
            harmony_types: List of harmony types to use (if None, use all)
            
        Returns:
            List of ColorMatch objects
        """
        if harmony_types is None:
            harmony_types = list(ColorHarmony)
        
        matches = []
        
        for harmony in harmony_types:
            if harmony == ColorHarmony.COMPLEMENTARY:
                color = self.get_complementary_color(base_color)
                matches.append(ColorMatch(
                    recommended_color=color,
                    harmony_type=harmony,
                    compatibility_score=0.95,
                    reason="Complementary colors create high contrast and visual interest"
                ))
            
            elif harmony == ColorHarmony.ANALOGOUS:
                colors = self.get_analogous_colors(base_color, count=2)
                for color in colors:
                    matches.append(ColorMatch(
                        recommended_color=color,
                        harmony_type=harmony,
                        compatibility_score=0.90,
                        reason="Analogous colors create harmonious, pleasing combinations"
                    ))
            
            elif harmony == ColorHarmony.TRIADIC:
                colors = self.get_triadic_colors(base_color)
                for color in colors:
                    matches.append(ColorMatch(
                        recommended_color=color,
                        harmony_type=harmony,
                        compatibility_score=0.85,
                        reason="Triadic colors create vibrant, balanced combinations"
                    ))
            
            elif harmony == ColorHarmony.MONOCHROMATIC:
                colors = self.get_monochromatic_colors(base_color, count=3)
                for color in colors:
                    matches.append(ColorMatch(
                        recommended_color=color,
                        harmony_type=harmony,
                        compatibility_score=0.88,
                        reason="Monochromatic scheme creates elegant, cohesive looks"
                    ))
        
        return matches
    
    def get_seasonal_palette(self, season: str, undertone: str) -> List[Tuple[int, int, int]]:
        """
        Get seasonal color palette based on season and skin undertone.
        
        Args:
            season: Season name ('spring', 'summer', 'autumn', 'winter')
            undertone: Skin undertone ('warm', 'cool', 'neutral')
            
        Returns:
            List of recommended colors in RGB
        """
        season = season.lower()
        
        if season not in self.SEASONAL_PALETTES:
            # Default to all seasons
            all_colors = []
            for palette in self.SEASONAL_PALETTES.values():
                all_colors.extend(palette['colors'])
            return all_colors
        
        palette = self.SEASONAL_PALETTES[season]
        
        # If undertone matches season's warmth, return full palette
        # Otherwise, filter to neutral colors
        if undertone == 'neutral' or \
           (undertone == 'warm' and palette['warm']) or \
           (undertone == 'cool' and not palette['warm']):
            return palette['colors']
        else:
            # Return a subset of more neutral colors from the palette
            return palette['colors'][:3]
    
    def calculate_color_distance(self, color1: Tuple[int, int, int],
                                 color2: Tuple[int, int, int]) -> float:
        """
        Calculate perceptual distance between two colors using LAB color space.
        
        Args:
            color1: First color in RGB
            color2: Second color in RGB
            
        Returns:
            Distance value (lower = more similar)
        """
        # Convert RGB to LAB (simplified - proper conversion requires more steps)
        # For now, use simple Euclidean distance in RGB space
        r1, g1, b1 = color1
        r2, g2, b2 = color2
        
        distance = np.sqrt((r1 - r2)**2 + (g1 - g2)**2 + (b1 - b2)**2)
        
        # Normalize to 0-1 range
        return distance / (255 * np.sqrt(3))
    
    def is_neutral_color(self, rgb: Tuple[int, int, int], threshold: int = 30) -> bool:
        """
        Check if a color is neutral (low saturation).
        
        Args:
            rgb: Color in RGB
            threshold: Maximum difference between RGB channels
            
        Returns:
            True if neutral, False otherwise
        """
        r, g, b = rgb
        return abs(r - g) < threshold and abs(g - b) < threshold and abs(r - b) < threshold
    
    def get_color_temperature(self, rgb: Tuple[int, int, int]) -> str:
        """
        Determine if a color is warm or cool.
        
        Args:
            rgb: Color in RGB
            
        Returns:
            'warm', 'cool', or 'neutral'
        """
        r, g, b = rgb
        
        # Convert to HSV to check hue
        r_norm, g_norm, b_norm = [x / 255.0 for x in rgb]
        h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
        
        # Hue ranges (in degrees, normalized to 0-1)
        # Warm: Red (0-60°), Yellow-Orange (30-90°)
        # Cool: Blue (180-240°), Green-Blue (120-240°)
        
        hue_degrees = h * 360
        
        if s < 0.2:  # Low saturation = neutral
            return 'neutral'
        elif (hue_degrees < 60) or (hue_degrees > 300):
            return 'warm'  # Red-Orange range
        elif 60 <= hue_degrees < 150:
            return 'warm'  # Yellow-Green range
        elif 150 <= hue_degrees < 300:
            return 'cool'  # Blue-Purple range
        else:
            return 'neutral'


def test_color_analyzer():
    """Test the color analyzer."""
    analyzer = ColorAnalyzer()
    
    # Test with a sample color (coral)
    test_color = (255, 127, 80)
    
    print("Color Analyzer Test")
    print("=" * 50)
    print(f"\nBase Color (RGB): {test_color}")
    print(f"Base Color (HEX): #{test_color[0]:02x}{test_color[1]:02x}{test_color[2]:02x}")
    
    # Get complementary
    comp = analyzer.get_complementary_color(test_color)
    print(f"\nComplementary: RGB{comp}, HEX#{comp[0]:02x}{comp[1]:02x}{comp[2]:02x}")
    
    # Get analogous
    analogous = analyzer.get_analogous_colors(test_color, count=2)
    print(f"\nAnalogous colors:")
    for i, color in enumerate(analogous, 1):
        print(f"  {i}. RGB{color}, HEX#{color[0]:02x}{color[1]:02x}{color[2]:02x}")
    
    # Get color temperature
    temp = analyzer.get_color_temperature(test_color)
    print(f"\nColor Temperature: {temp}")


if __name__ == "__main__":
    test_color_analyzer()
