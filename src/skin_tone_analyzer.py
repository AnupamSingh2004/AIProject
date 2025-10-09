"""
Skin Tone Analyzer Module
Detects faces, extracts skin tone, classifies Fitzpatrick type, and determines undertones.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum

class FitzpatrickType(Enum):
    """Fitzpatrick skin type classification."""
    TYPE_I = 1    # Very fair, always burns, never tans
    TYPE_II = 2   # Fair, usually burns, tans minimally
    TYPE_III = 3  # Medium, sometimes burns, tans uniformly
    TYPE_IV = 4   # Olive, rarely burns, tans easily
    TYPE_V = 5    # Brown, very rarely burns, tans very easily
    TYPE_VI = 6   # Dark brown to black, never burns

class Undertone(Enum):
    """Skin undertone classification."""
    COOL = "cool"     # Pink, red, or bluish undertones
    WARM = "warm"     # Yellow, peachy, or golden undertones
    NEUTRAL = "neutral"  # Mix of cool and warm

@dataclass
class SkinToneResult:
    """Result of skin tone analysis."""
    fitzpatrick_type: FitzpatrickType
    undertone: Undertone
    dominant_color_rgb: Tuple[int, int, int]
    dominant_color_hex: str
    hsv_values: Tuple[float, float, float]
    lab_values: Tuple[float, float, float]
    confidence: float
    
    def __str__(self):
        return (f"Fitzpatrick Type: {self.fitzpatrick_type.name}\n"
                f"Undertone: {self.undertone.value}\n"
                f"Dominant Color (RGB): {self.dominant_color_rgb}\n"
                f"Dominant Color (HEX): {self.dominant_color_hex}\n"
                f"Confidence: {self.confidence:.2%}")

class SkinToneAnalyzer:
    """Analyzes skin tone from facial images."""
    
    def __init__(self):
        """Initialize the skin tone analyzer with MediaPipe face detection."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
    def analyze(self, image_path: str) -> Optional[SkinToneResult]:
        """
        Analyze skin tone from an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            SkinToneResult object or None if face not detected
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect face landmarks
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            print("⚠️ No face detected in the image")
            return None
        
        # Get face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract skin pixels from facial regions
        skin_pixels = self._extract_skin_pixels(image_rgb, face_landmarks)
        
        if len(skin_pixels) == 0:
            print("⚠️ Could not extract skin pixels")
            return None
        
        # Calculate dominant skin color
        dominant_color_rgb = self._calculate_dominant_color(skin_pixels)
        
        # Convert to other color spaces
        hsv_values = self._rgb_to_hsv(dominant_color_rgb)
        lab_values = self._rgb_to_lab(dominant_color_rgb)
        
        # Classify Fitzpatrick type
        fitzpatrick_type = self._classify_fitzpatrick(dominant_color_rgb, lab_values)
        
        # Determine undertone
        undertone = self._determine_undertone(dominant_color_rgb, hsv_values, lab_values)
        
        # Calculate confidence (based on consistency of skin pixels)
        confidence = self._calculate_confidence(skin_pixels, dominant_color_rgb)
        
        # Convert to hex
        dominant_color_hex = self._rgb_to_hex(dominant_color_rgb)
        
        return SkinToneResult(
            fitzpatrick_type=fitzpatrick_type,
            undertone=undertone,
            dominant_color_rgb=dominant_color_rgb,
            dominant_color_hex=dominant_color_hex,
            hsv_values=hsv_values,
            lab_values=lab_values,
            confidence=confidence
        )
    
    def _extract_skin_pixels(self, image: np.ndarray, face_landmarks) -> np.ndarray:
        """Extract skin pixels from facial regions avoiding eyes, mouth, etc."""
        h, w, _ = image.shape
        
        # Define facial regions to sample (cheeks, forehead, nose)
        # These landmark indices are for MediaPipe face mesh
        forehead_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        left_cheek_indices = [425, 436, 434, 432, 430, 431, 262, 428, 199, 428, 262]
        right_cheek_indices = [205, 214, 212, 210, 208, 207, 32, 204, 202, 201]
        
        skin_pixels = []
        
        # Sample pixels from defined regions
        for indices in [forehead_indices, left_cheek_indices, right_cheek_indices]:
            for idx in indices:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                
                # Sample a small region around the landmark
                if 5 < x < w-5 and 5 < y < h-5:
                    region = image[y-2:y+3, x-2:x+3]
                    skin_pixels.extend(region.reshape(-1, 3))
        
        return np.array(skin_pixels)
    
    def _calculate_dominant_color(self, pixels: np.ndarray) -> Tuple[int, int, int]:
        """Calculate the dominant color from skin pixels using median."""
        # Use median to be robust against outliers
        median_color = np.median(pixels, axis=0).astype(int)
        return tuple(median_color)
    
    def _rgb_to_hsv(self, rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Convert RGB to HSV color space."""
        rgb_normalized = np.array([[rgb]], dtype=np.uint8)
        hsv = cv2.cvtColor(rgb_normalized, cv2.COLOR_RGB2HSV)[0][0]
        return tuple(map(float, hsv))
    
    def _rgb_to_lab(self, rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Convert RGB to LAB color space."""
        rgb_normalized = np.array([[rgb]], dtype=np.uint8)
        lab = cv2.cvtColor(rgb_normalized, cv2.COLOR_RGB2LAB)[0][0]
        return tuple(map(float, lab))
    
    def _rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB to hexadecimal color code."""
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    
    def _classify_fitzpatrick(self, rgb: Tuple[int, int, int], 
                             lab: Tuple[float, float, float]) -> FitzpatrickType:
        """
        Classify Fitzpatrick skin type based on RGB and LAB values.
        L* value in LAB is most indicative of skin lightness.
        """
        L = lab[0]  # Lightness value (0-255)
        
        # Classification based on L* value
        # These thresholds are approximations
        if L >= 85:
            return FitzpatrickType.TYPE_I
        elif L >= 75:
            return FitzpatrickType.TYPE_II
        elif L >= 65:
            return FitzpatrickType.TYPE_III
        elif L >= 55:
            return FitzpatrickType.TYPE_IV
        elif L >= 40:
            return FitzpatrickType.TYPE_V
        else:
            return FitzpatrickType.TYPE_VI
    
    def _determine_undertone(self, rgb: Tuple[int, int, int], 
                            hsv: Tuple[float, float, float],
                            lab: Tuple[float, float, float]) -> Undertone:
        """
        Determine skin undertone (cool/warm/neutral) based on color values.
        
        Cool undertones: Pink, red, or bluish (higher b* in LAB, lower red in RGB)
        Warm undertones: Yellow, peachy, or golden (lower b*, higher yellow)
        Neutral: Balanced mix
        """
        r, g, b = rgb
        L, a, b_lab = lab
        
        # a* represents green-red axis
        # b* represents blue-yellow axis
        
        # Calculate undertone indicators
        yellow_indicator = b_lab  # Higher = more yellow (warm)
        red_vs_blue = r - b  # Higher = more red undertone
        
        # Warm undertone indicators
        if yellow_indicator > 138 and red_vs_blue > 20:
            return Undertone.WARM
        # Cool undertone indicators
        elif yellow_indicator < 130 or red_vs_blue < 10:
            return Undertone.COOL
        # Neutral
        else:
            return Undertone.NEUTRAL
    
    def _calculate_confidence(self, pixels: np.ndarray, 
                             dominant_color: Tuple[int, int, int]) -> float:
        """
        Calculate confidence score based on consistency of skin pixels.
        Higher consistency = higher confidence.
        """
        if len(pixels) == 0:
            return 0.0
        
        # Calculate standard deviation of color values
        std_dev = np.std(pixels, axis=0)
        avg_std = np.mean(std_dev)
        
        # Convert to confidence score (lower std = higher confidence)
        # Normalize to 0-1 range
        confidence = max(0.0, min(1.0, 1.0 - (avg_std / 100.0)))
        
        return confidence
    
    def __del__(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()


def test_skin_tone_analyzer():
    """Test the skin tone analyzer with a sample image."""
    analyzer = SkinToneAnalyzer()
    
    # You would replace this with an actual image path
    print("Skin Tone Analyzer initialized successfully!")
    print("Usage example:")
    print("  analyzer = SkinToneAnalyzer()")
    print("  result = analyzer.analyze('path/to/face_image.jpg')")
    print("  print(result)")


if __name__ == "__main__":
    test_skin_tone_analyzer()
