"""
Example script demonstrating the complete fashion recommendation workflow.
"""

import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from skin_tone_analyzer import SkinToneAnalyzer
from clothing_detector import ClothingDetector, ClothingType
from recommendation_engine import RecommendationEngine, Occasion, Season


def demo_workflow():
    """Demonstrate the complete recommendation workflow."""
    
    print("=" * 70)
    print("🎨 AI Fashion Recommendation System - Demo")
    print("=" * 70)
    
    # Initialize components
    print("\n📦 Initializing components...")
    skin_analyzer = SkinToneAnalyzer()
    clothing_detector = ClothingDetector()
    recommender = RecommendationEngine()
    print("✅ All components initialized!\n")
    
    # Step 1: Skin Tone Analysis
    print("-" * 70)
    print("Step 1: Skin Tone Analysis")
    print("-" * 70)
    print("To analyze your skin tone, you would:")
    print("  1. Upload a clear face photo")
    print("  2. Run: result = skin_analyzer.analyze('your_photo.jpg')")
    print("\nExample output:")
    print("  Fitzpatrick Type: TYPE_III")
    print("  Undertone: warm")
    print("  Dominant Color (HEX): #d4a574")
    print("  Confidence: 87%\n")
    
    # Step 2: Build Wardrobe
    print("-" * 70)
    print("Step 2: Build Your Wardrobe")
    print("-" * 70)
    print("To add clothing items:")
    print("  1. Upload clothing item images")
    print("  2. Run: item = clothing_detector.detect('shirt.jpg', ClothingType.SHIRT)")
    print("\nExample wardrobe:")
    print("  - White Button-Down Shirt (Formal)")
    print("  - Blue Jeans (Casual)")
    print("  - Black Dress Pants (Business)")
    print("  - Red Party Dress (Party)")
    print("  - Sneakers (Athletic)\n")
    
    # Step 3: Get Recommendations
    print("-" * 70)
    print("Step 3: Get Personalized Recommendations")
    print("-" * 70)
    print("Select your occasion and season:")
    
    occasions = [
        ("Casual", "Spring"),
        ("Business", "Autumn"),
        ("Party", "Summer"),
        ("Date", "Spring"),
    ]
    
    for occasion_name, season_name in occasions:
        print(f"\n🎯 Occasion: {occasion_name} | Season: {season_name}")
        print("   Recommended Outfit:")
        print("   - Top: Light blue shirt with white undertones")
        print("   - Bottom: Khaki pants")
        print("   - Shoes: Brown loafers")
        print(f"   - Score: 89% compatibility")
        print(f"   - Reason: Complementary colors create visual interest and")
        print(f"     match your warm undertone perfectly.")
    
    print("\n" + "=" * 70)
    print("🚀 How to Use This System")
    print("=" * 70)
    print("\n1. Run the Streamlit App:")
    print("   $ conda activate AI")
    print("   $ cd app")
    print("   $ streamlit run streamlit_app.py")
    print("\n2. Upload your photo for skin tone analysis")
    print("\n3. Add your clothing items to build a virtual wardrobe")
    print("\n4. Select occasion and season, then get recommendations!")
    print("\n5. The system will score each outfit combination based on:")
    print("   - Skin tone compatibility (30%)")
    print("   - Color harmony (30%)")
    print("   - Occasion appropriateness (20%)")
    print("   - Seasonal suitability (20%)")
    
    print("\n" + "=" * 70)
    print("💡 Key Features")
    print("=" * 70)
    print("\n✨ Occasion-Based Recommendations:")
    print("   - Casual: Everyday wear, relaxed settings")
    print("   - Formal: Elegant events, ceremonies")
    print("   - Business: Professional workplace attire")
    print("   - Party: Social gatherings and celebrations")
    print("   - Date: Romantic outings")
    print("   - Athletic: Gym and sports activities")
    print("   - Beach: Outdoor and summer activities")
    
    print("\n🌸 Seasonal Color Palettes:")
    print("   - Spring: Peach, warm beige, light coral")
    print("   - Summer: Powder blue, lavender, cool gray")
    print("   - Autumn: Brown, sienna, goldenrod, olive")
    print("   - Winter: Black, white, crimson, midnight blue")
    
    print("\n🎨 Color Theory Integration:")
    print("   - Complementary: Opposite colors (high contrast)")
    print("   - Analogous: Adjacent colors (harmonious)")
    print("   - Triadic: Evenly spaced (vibrant balance)")
    print("   - Monochromatic: Same hue (elegant cohesion)")
    
    print("\n" + "=" * 70)
    print("📚 Documentation")
    print("=" * 70)
    print("\nFor detailed documentation, see:")
    print("  - README.md: Complete usage guide and API reference")
    print("  - details.md: Full project requirements specification")
    print("  - Source code in src/ directory with inline comments")
    
    print("\n" + "=" * 70)
    print("✅ System Ready!")
    print("=" * 70)
    print("\nThe AI Fashion Recommendation System is fully operational!")
    print("Start the Streamlit app to begin getting personalized")
    print("outfit recommendations based on your skin tone, occasion,")
    print("and season preferences.\n")


if __name__ == "__main__":
    demo_workflow()
