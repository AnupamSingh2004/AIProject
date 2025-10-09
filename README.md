# 👗 AI Fashion Recommendation System

An intelligent fashion recommendation system that analyzes your skin tone and suggests personalized outfit combinations based on color theory, occasion, and season.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.12-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🌟 Features

- **🎨 Skin Tone Analysis**: Automatically detects and classifies skin tone using the Fitzpatrick scale
- **👔 Occasion-Based Recommendations**: Get outfit suggestions for different purposes:
  - Casual
  - Formal
  - Business
  - Party
  - Date
  - Athletic/Gym
  - Beach/Outdoor
- **🌸 Seasonal Recommendations**: Outfit suggestions tailored to Spring, Summer, Autumn, or Winter
- **🎯 Color Theory Integration**: Uses complementary, analogous, and triadic color harmonies
- **👕 Virtual Wardrobe**: Upload and manage your own clothing items
- **📊 Smart Scoring System**: Evaluates outfits based on:
  - Skin tone compatibility
  - Color harmony
  - Occasion appropriateness
  - Seasonal suitability

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [API Reference](#api-reference)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Prerequisites

- Python 3.10+
- Conda (Miniconda or Anaconda)
- Kaggle Account (optional, for dataset)

### Step 1: Clone the Repository

```bash
git clone https://github.com/AnupamSingh2004/AIProject.git
cd AIProject
```

### Step 2: Create Conda Environment

```bash
conda create -n AI python=3.10 -y
conda activate AI
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install packages individually:

```bash
pip install tensorflow opencv-python mediapipe Pillow scikit-image colorthief scikit-learn pandas matplotlib seaborn tqdm pyyaml requests streamlit
```

### Step 4: Download Dataset (Optional)

If you want to use the Kaggle Fashion Product Images dataset:

1. Create a Kaggle account at https://www.kaggle.com
2. Go to https://www.kaggle.com/settings/account
3. Click "Create New API Token" under API section
4. Place `kaggle.json` at `~/.kaggle/kaggle.json`
5. Run:

```bash
chmod 600 ~/.kaggle/kaggle.json
conda run -n AI python scripts/download_dataset.py
```

## 🎯 Quick Start

### Run the Streamlit App

```bash
conda activate AI
cd app
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

### Using the Application

1. **Upload Your Photo** (Tab 1)
   - Upload a clear face photo
   - Click "Analyze Skin Tone"
   - View your Fitzpatrick type, undertone, and recommended colors

2. **Build Your Wardrobe** (Tab 2)
   - Upload clothing item images
   - Select clothing type
   - Add items to your virtual wardrobe

3. **Get Recommendations** (Tab 3)
   - Select occasion (Casual, Formal, Party, etc.)
   - Choose current season
   - Click "Generate Recommendations"
   - View personalized outfit suggestions!

## 📖 Usage

### Python API

#### Analyze Skin Tone

```python
from src.skin_tone_analyzer import SkinToneAnalyzer

analyzer = SkinToneAnalyzer()
result = analyzer.analyze('path/to/your/photo.jpg')

print(f"Fitzpatrick Type: {result.fitzpatrick_type.name}")
print(f"Undertone: {result.undertone.value}")
print(f"Dominant Color: {result.dominant_color_hex}")
```

#### Detect Clothing

```python
from src.clothing_detector import ClothingDetector, ClothingType

detector = ClothingDetector()
item = detector.detect(
    'path/to/clothing.jpg',
    clothing_type=ClothingType.SHIRT
)

print(f"Dominant Color: {item.dominant_color_hex()}")
print(f"Pattern: {item.pattern.value}")
print(f"Style: {item.style.value}")
```

#### Get Outfit Recommendations

```python
from src.recommendation_engine import RecommendationEngine, Occasion, Season

recommender = RecommendationEngine()

recommendations = recommender.recommend_outfits(
    skin_tone=skin_tone_result,
    wardrobe=clothing_items,
    occasion=Occasion.PARTY,
    season=Season.SUMMER,
    count=5
)

for i, outfit in enumerate(recommendations, 1):
    print(f"\nOutfit #{i} (Score: {outfit.compatibility_score:.2%})")
    print(f"Top: {outfit.top.item_type.value}")
    print(f"Bottom: {outfit.bottom.item_type.value}")
    print(f"Reason: {outfit.reason}")
```

## 📁 Project Structure

```
AIProject/
├── app/
│   └── streamlit_app.py          # Streamlit web application
├── data/
│   ├── raw/                       # Original Kaggle dataset
│   ├── processed/                 # Processed data
│   └── user_uploads/              # User uploaded images
├── models/
│   └── saved_models/              # Trained model weights
├── notebooks/                     # Jupyter notebooks for experiments
├── scripts/
│   └── download_dataset.py        # Dataset download script
├── src/
│   ├── skin_tone_analyzer.py      # Skin tone detection module
│   ├── clothing_detector.py       # Clothing detection module
│   ├── color_analyzer.py          # Color theory engine
│   └── recommendation_engine.py   # Main recommendation system
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── details.md                     # Detailed project requirements
```

## 🔬 How It Works

### 1. Skin Tone Analysis

The system uses **MediaPipe Face Mesh** to detect facial landmarks and extract skin pixels from key facial regions (forehead, cheeks). It then:

- Calculates dominant skin color in RGB, HSV, and LAB color spaces
- Classifies Fitzpatrick skin type (I-VI) based on lightness values
- Determines undertone (warm/cool/neutral) using color theory
- Provides confidence scores based on pixel consistency

### 2. Clothing Detection

The clothing detector:

- Extracts dominant colors using ColorThief library
- Detects patterns (solid, striped, checkered, etc.) using edge detection and texture analysis
- Classifies clothing style based on type, pattern, and colors
- Creates a comprehensive profile for each clothing item

### 3. Color Matching

The color analyzer implements color theory principles:

- **Complementary**: Colors opposite on the color wheel (high contrast)
- **Analogous**: Adjacent colors (harmonious combinations)
- **Triadic**: Colors evenly spaced on the wheel (vibrant balance)
- **Monochromatic**: Same hue with varying saturation/brightness

### 4. Recommendation Engine

The engine scores outfit combinations based on:

1. **Skin Tone Compatibility (30%)**: How well colors match your skin tone
2. **Color Harmony (30%)**: How well outfit colors work together
3. **Occasion Match (20%)**: Appropriateness for the selected purpose
4. **Seasonal Match (20%)**: Suitability for the current season

## 🎨 Supported Occasions

| Occasion | Description | Suitable Styles |
|----------|-------------|-----------------|
| **Casual** | Everyday wear, relaxed settings | Casual, Streetwear |
| **Formal** | Elegant events, ceremonies | Formal, Business |
| **Business** | Professional workplace | Business, Formal, Minimalist |
| **Party** | Social gatherings, celebrations | Bohemian, Vintage, Streetwear |
| **Date** | Romantic outings | Casual, Bohemian, Vintage |
| **Athletic** | Gym, sports activities | Athletic |
| **Beach** | Outdoor, summer activities | Casual, Bohemian |

## 🌡️ Seasonal Palettes

The system provides season-specific color recommendations:

- **Spring**: Peach, warm beige, light coral, pale green
- **Summer**: Powder blue, lavender, cool gray, light pink
- **Autumn**: Brown, sienna, goldenrod, olive
- **Winter**: Black, white, crimson, midnight blue, purple

## 📊 API Reference

### SkinToneAnalyzer

```python
class SkinToneAnalyzer:
    def analyze(image_path: str) -> SkinToneResult
```

**Returns:** `SkinToneResult` with fitzpatrick_type, undertone, dominant_color_rgb, hsv_values, lab_values, confidence

### ClothingDetector

```python
class ClothingDetector:
    def detect(image_path: str, clothing_type: Optional[ClothingType]) -> ClothingItem
```

**Returns:** `ClothingItem` with item_type, dominant_color, color_palette, pattern, style, confidence

### RecommendationEngine

```python
class RecommendationEngine:
    def recommend_outfits(
        skin_tone: SkinToneResult,
        wardrobe: List[ClothingItem],
        occasion: Occasion,
        season: Season,
        count: int = 5
    ) -> List[OutfitRecommendation]
```

**Returns:** List of `OutfitRecommendation` sorted by compatibility score

## 📦 Dataset

This project can use the **Fashion Product Images Dataset** from Kaggle:

- **Dataset**: [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)
- **Size**: 44k products with category labels and high-res images
- **Format**: Images + CSV with metadata

## 🛠️ Technologies Used

- **Python 3.10**: Core programming language
- **TensorFlow 2.20**: Deep learning framework
- **OpenCV 4.12**: Computer vision operations
- **MediaPipe 0.10**: Face detection and mesh
- **Streamlit 1.50**: Web application framework
- **NumPy, Pandas**: Data processing
- **Scikit-learn**: Machine learning utilities
- **ColorThief**: Color palette extraction

## 🤝 Contributing

Contributions are welcome! Here are some ways you can contribute:

1. **Add More Clothing Types**: Extend the ClothingType enum
2. **Improve Pattern Detection**: Enhance the pattern recognition algorithm
3. **Train Custom Models**: Train CNNs for better clothing classification
4. **Add More Occasions**: Extend occasion types and scoring
5. **UI Improvements**: Enhance the Streamlit interface

## 📝 Future Enhancements

- [ ] Train custom CNN for clothing classification
- [ ] Add body type recommendations
- [ ] Implement user preference learning
- [ ] Add outfit history tracking
- [ ] Social sharing features
- [ ] Mobile app development
- [ ] Integration with e-commerce platforms

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Anupam Singh**
- GitHub: [@AnupamSingh2004](https://github.com/AnupamSingh2004)

## 🙏 Acknowledgments

- Kaggle Fashion Product Images Dataset by Param Aggarwal
- MediaPipe team for face detection
- Color theory resources from various fashion experts
- Open source community for amazing libraries

## 📞 Support

If you have any questions or issues, please open an issue on GitHub or contact the author.

---

Made with ❤️ and AI
