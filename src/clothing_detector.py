"""
Clothing Detector and Classifier Module
Detects clothing items, extracts colors, identifies patterns, and classifies types.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from colorthief import ColorThief
from PIL import Image
import io

class ClothingType(Enum):
    """Types of clothing items."""
    SHIRT = "shirt"
    T_SHIRT = "t-shirt"
    BLOUSE = "blouse"
    PANTS = "pants"
    JEANS = "jeans"
    SHORTS = "shorts"
    DRESS = "dress"
    SKIRT = "skirt"
    JACKET = "jacket"
    COAT = "coat"
    SWEATER = "sweater"
    HOODIE = "hoodie"
    SHOES = "shoes"
    ACCESSORIES = "accessories"
    UNKNOWN = "unknown"

class Pattern(Enum):
    """Clothing pattern types."""
    SOLID = "solid"
    STRIPED = "striped"
    CHECKERED = "checkered"
    PLAID = "plaid"
    FLORAL = "floral"
    GEOMETRIC = "geometric"
    POLKA_DOT = "polka_dot"
    ABSTRACT = "abstract"
    PRINTED = "printed"

class ClothingStyle(Enum):
    """Clothing style categories."""
    CASUAL = "casual"
    FORMAL = "formal"
    BUSINESS = "business"
    ATHLETIC = "athletic"
    BOHEMIAN = "bohemian"
    VINTAGE = "vintage"
    STREETWEAR = "streetwear"
    MINIMALIST = "minimalist"

@dataclass
class ClothingItem:
    """Represents a detected clothing item."""
    item_type: ClothingType
    dominant_color: Tuple[int, int, int]
    color_palette: List[Tuple[int, int, int]]
    pattern: Pattern
    style: ClothingStyle
    confidence: float
    image_path: Optional[str] = None
    
    def dominant_color_hex(self) -> str:
        """Get dominant color as hex."""
        return f"#{self.dominant_color[0]:02x}{self.dominant_color[1]:02x}{self.dominant_color[2]:02x}"
    
    def __str__(self):
        colors_hex = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in self.color_palette]
        return (f"Type: {self.item_type.value}\n"
                f"Dominant Color: {self.dominant_color_hex()}\n"
                f"Color Palette: {', '.join(colors_hex)}\n"
                f"Pattern: {self.pattern.value}\n"
                f"Style: {self.style.value}\n"
                f"Confidence: {self.confidence:.2%}")

class ClothingDetector:
    """Detects and classifies clothing items from images."""
    
    def __init__(self):
        """Initialize the clothing detector."""
        pass
    
    def detect(self, image_path: str, clothing_type: Optional[ClothingType] = None) -> ClothingItem:
        """
        Detect and analyze a clothing item from an image.
        
        Args:
            image_path: Path to the clothing image
            clothing_type: Optional manual specification of clothing type
            
        Returns:
            ClothingItem object with analysis results
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Extract color information
        dominant_color, color_palette = self._extract_colors(image_path)
        
        # Detect pattern
        pattern = self._detect_pattern(image)
        
        # Classify clothing type (if not provided)
        if clothing_type is None:
            clothing_type = self._classify_type(image, dominant_color, pattern)
        
        # Determine style
        style = self._determine_style(clothing_type, pattern, dominant_color)
        
        # Calculate confidence
        confidence = 0.85  # Placeholder for now
        
        return ClothingItem(
            item_type=clothing_type,
            dominant_color=dominant_color,
            color_palette=color_palette,
            pattern=pattern,
            style=style,
            confidence=confidence,
            image_path=image_path
        )
    
    def _extract_colors(self, image_path: str, num_colors: int = 5) -> Tuple[Tuple[int, int, int], List[Tuple[int, int, int]]]:
        """
        Extract dominant color and color palette from image.
        
        Args:
            image_path: Path to image
            num_colors: Number of colors in palette
            
        Returns:
            Tuple of (dominant_color, color_palette)
        """
        try:
            color_thief = ColorThief(image_path)
            
            # Get dominant color
            dominant_color = color_thief.get_color(quality=1)
            
            # Get color palette
            palette = color_thief.get_palette(color_count=num_colors, quality=1)
            
            return dominant_color, palette
            
        except Exception as e:
            print(f"Warning: Could not extract colors with ColorThief: {e}")
            # Fallback: use simple color extraction
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Calculate dominant color using k-means
            pixels = image_rgb.reshape(-1, 3)
            
            # Simple method: use median color
            dominant = tuple(np.median(pixels, axis=0).astype(int))
            
            # Create simple palette
            palette = [dominant]
            
            return dominant, palette
    
    def _detect_pattern(self, image: np.ndarray) -> Pattern:
        """
        Detect clothing pattern using edge detection and texture analysis.
        
        Args:
            image: OpenCV image array
            
        Returns:
            Pattern enum
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate variance to detect patterns
        variance = np.var(gray)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / edges.size
        
        # FFT for frequency analysis (detects repetitive patterns)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Simple heuristics for pattern detection
        if edge_density < 0.05 and variance < 500:
            return Pattern.SOLID
        elif edge_density > 0.15:
            # High edge density could indicate stripes, checks, etc.
            if variance > 1000:
                return Pattern.CHECKERED
            else:
                return Pattern.STRIPED
        elif variance > 1500:
            # High variance suggests complex pattern
            return Pattern.FLORAL
        else:
            return Pattern.SOLID
    
    def _classify_type(self, image: np.ndarray, 
                       dominant_color: Tuple[int, int, int],
                       pattern: Pattern) -> ClothingType:
        """
        Classify the type of clothing item.
        This is a simplified version - in production, you'd use a trained CNN.
        
        Args:
            image: OpenCV image array
            dominant_color: Dominant color RGB
            pattern: Detected pattern
            
        Returns:
            ClothingType enum
        """
        # For now, return unknown - in production, this would use:
        # - Pre-trained model (ResNet, EfficientNet)
        # - Fashion-specific models trained on DeepFashion dataset
        # - YOLO or other object detection for item localization
        
        # Placeholder logic based on image aspect ratio
        h, w = image.shape[:2]
        aspect_ratio = h / w
        
        if aspect_ratio > 1.5:
            return ClothingType.DRESS
        elif aspect_ratio < 0.8:
            return ClothingType.SHIRT
        else:
            return ClothingType.UNKNOWN
    
    def _determine_style(self, clothing_type: ClothingType,
                        pattern: Pattern,
                        dominant_color: Tuple[int, int, int]) -> ClothingStyle:
        """
        Determine the style of the clothing based on type, pattern, and color.
        
        Args:
            clothing_type: Type of clothing
            pattern: Pattern of clothing
            dominant_color: Dominant color RGB
            
        Returns:
            ClothingStyle enum
        """
        # Simple heuristics for style classification
        r, g, b = dominant_color
        
        # Check if it's a neutral/dark color (business/formal)
        is_neutral = abs(r - g) < 30 and abs(g - b) < 30 and abs(r - b) < 30
        is_dark = max(r, g, b) < 100
        
        if clothing_type in [ClothingType.SHIRT, ClothingType.BLOUSE] and pattern == Pattern.SOLID:
            if is_neutral or is_dark:
                return ClothingStyle.FORMAL
            else:
                return ClothingStyle.CASUAL
        
        elif clothing_type in [ClothingType.T_SHIRT, ClothingType.JEANS, ClothingType.SHORTS]:
            return ClothingStyle.CASUAL
        
        elif clothing_type in [ClothingType.JACKET, ClothingType.COAT]:
            if is_dark:
                return ClothingStyle.BUSINESS
            else:
                return ClothingStyle.CASUAL
        
        elif pattern in [Pattern.FLORAL, Pattern.ABSTRACT]:
            return ClothingStyle.BOHEMIAN
        
        elif clothing_type == ClothingType.HOODIE:
            return ClothingStyle.STREETWEAR
        
        else:
            return ClothingStyle.CASUAL


def load_kaggle_dataset(data_dir: str) -> List[Dict]:
    """
    Load the Kaggle Fashion Product Images dataset.
    
    Args:
        data_dir: Directory containing the dataset
        
    Returns:
        List of clothing items with metadata
    """
    import pandas as pd
    from pathlib import Path
    
    data_path = Path(data_dir)
    styles_csv = data_path / 'styles.csv'
    
    if not styles_csv.exists():
        print(f"Warning: styles.csv not found at {styles_csv}")
        return []
    
    # Load the CSV
    df = pd.read_csv(styles_csv, on_bad_lines='skip')
    
    print(f"Loaded {len(df)} items from Kaggle dataset")
    print(f"Columns: {df.columns.tolist()}")
    
    return df.to_dict('records')


def test_clothing_detector():
    """Test the clothing detector."""
    detector = ClothingDetector()
    print("Clothing Detector initialized successfully!")
    print("\nUsage example:")
    print("  detector = ClothingDetector()")
    print("  item = detector.detect('path/to/clothing_image.jpg')")
    print("  print(item)")


if __name__ == "__main__":
    test_clothing_detector()
