# 👗 AI Fashion Recommendation System

An intelligent **deep learning-powered** fashion recommendation system that uses computer vision to analyze clothing patterns, extract RGB colors from images, and suggest complete outfit combinations (Top + Bottom + Shoes) using a trained Siamese CNN model with 2.7M parameters.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.12-green)
![GPU](https://img.shields.io/badge/GPU-CUDA%2012.x-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🌟 Features

### **Core AI Models**

1. **🧠 Clothing Classifier (95.48% Accuracy)**
   - **Architecture**: MobileNetV2 transfer learning (2.6M parameters)
   - **Categories**: Topwear, Bottomwear, Footwear, Dress, Accessories, Other
   - **Training**: 2-phase (frozen base → fine-tuned), 31,093 training images
   - **GPU Accelerated**: Trains on NVIDIA RTX 3050 in ~20 minutes

2. **👔 Advanced Outfit Compatibility Model**
   - **Architecture**: 3-input Siamese CNN (2.7M parameters)
   - **Inputs**: Top image + Bottom image + Shoes image (224x224x3 each)
   - **Output**: Compatibility score (0-1, sigmoid activation)
   - **Features Considered**:
     - RGB color harmony (complementary, analogous, monochromatic)
     - Pattern matching (solid, striped, checkered, floral, dotted)
     - Gender separation (Men's/Women's outfits)
     - Style consistency (Casual, Formal, Sports, Party, Ethnic)
   - **Training Strategy**: Balanced 50/50 positive/negative outfit sets

### **Visual Feature Extraction**

- **🎨 RGB Color Extraction**: K-means clustering to extract 3 dominant colors per item
- **🔍 Pattern Detection**: Edge detection + frequency analysis for:
  - Solid, Striped (horizontal/vertical), Checkered, Floral, Dotted, Textured
- **� Color Metrics**: Brightness, color diversity, temperature (warm/cool), saturation
- **⚡ Processing**: Cached image loading for 10x faster training

### **Recommendation Engine**

- **🎯 Complete Outfit Suggestions**: Recommends full 3-item sets (Top + Bottom + Shoes)
- **🌸 Seasonal Recommendations**: Outfit suggestions tailored to Spring, Summer, Autumn, Winter
- **� Smart Scoring System**: Multi-factor scoring:
  - Color harmony: 30-40%
  - Pattern compatibility: 30-40%
  - Style matching: 10-20%
  - Brightness balance: 10-20%
- **👗 Dress Support**: Special handling for one-piece dresses with footwear pairing

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
- GPU (recommended but not required)

### Quick Start (Automated)

```bash
# 1. Clone the repository
git clone https://github.com/AnupamSingh2004/AIProject.git
cd AIProject

# 2. Run the quickstart script
./quickstart.sh
```

The quickstart script provides an interactive menu for:
- Downloading the dataset
- Installing dependencies
- Training models
- Running the demo app

### Manual Installation

#### Step 1: Clone the Repository

```bash
git clone https://github.com/AnupamSingh2004/AIProject.git
cd AIProject
```

#### Step 2: Create Conda Environment

```bash
conda create -n AI python=3.10 -y
conda activate AI
```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install packages individually:

```bash
pip install tensorflow opencv-python mediapipe Pillow scikit-image colorthief scikit-learn pandas matplotlib seaborn tqdm pyyaml requests streamlit kaggle python-dotenv
```

#### Step 4: Verify Setup

```bash
python verify_setup.py
```

This will check that everything is properly configured.

#### Step 5: Download Dataset

**Option A: Using Kaggle API (Recommended)**

1. Create a Kaggle account at https://www.kaggle.com
2. Go to https://www.kaggle.com/settings/account
3. Click "Create New API Token" under API section
4. Place `kaggle.json` at `~/.kaggle/kaggle.json`
5. Run:

```bash
chmod 600 ~/.kaggle/kaggle.json
python scripts/download_dataset.py
```

**Option B: Manual Download**

1. Download from: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset
2. Extract to `data/raw/` directory

#### Step 6: Preprocess Data

```bash
python scripts/preprocess_data.py --data_dir data/raw --output_dir data/processed --img_size 224
```

This creates train/val/test splits and prepares the data for training.

#### Step 7: Train Models

**Option A: Full Pipeline (Recommended)**

```bash
python train.py
```

This trains both models automatically.

**Option B: Individual Models**

```bash
# Train clothing classifier only
python models/clothing_classifier.py

# Train outfit compatibility model only
python models/outfit_compatibility_model.py
```

For detailed training instructions, see [TRAINING.md](TRAINING.md)

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

### **System Architecture & Pipeline**

```
User Upload Image
      ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Clothing Classification (CNN)                       │
│ Model: clothing_classifier.keras (95.48% accuracy)          │
│ Input: 224x224x3 RGB image                                  │
│ Output: Category {Topwear, Bottomwear, Footwear, ...}      │
│ Architecture: MobileNetV2 + Custom Dense Layers             │
└─────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Visual Feature Extraction (Real-Time)              │
│ • RGB Color Extraction: K-means clustering (k=3)            │
│   - Dominant color 1, 2, 3 with percentages                 │
│ • Pattern Detection: Canny edge detection + analysis        │
│   - Edge density, horizontal/vertical gradients             │
│   - Blob detection for dots/polka patterns                  │
│ • Color Metrics: brightness, diversity, temperature         │
└─────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Load User's Wardrobe                               │
│ • Retrieve all uploaded items with cached features          │
│ • Filter by gender and category                             │
└─────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: Outfit Compatibility Scoring (Siamese CNN)         │
│ Model: outfit_compatibility_advanced.keras                  │
│                                                             │
│ For each outfit combination (Top, Bottom, Shoes):          │
│   ├─ Extract 64-dim features from each item (shared CNN)   │
│   ├─ Concatenate: [f_top, f_bottom, f_shoes] = 192-dim     │
│   ├─ Dense layers with dropout (256→128→64)                │
│   └─ Sigmoid output: compatibility_score ∈ [0, 1]          │
│                                                             │
│ Multi-Factor Scoring:                                       │
│   • Color Harmony (30-40%): HSV-based color wheel analysis │
│   • Pattern Compatibility (30-40%): Clash detection        │
│   • Style Matching (10-20%): Casual/Formal/Sports          │
│   • Brightness Balance (10-20%): Visual contrast           │
└─────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: Rank & Return Top Recommendations                  │
│ • Sort by compatibility score (descending)                  │
│ • Filter incompatible categories                            │
│ • Return top 10-20 outfits with reasons                     │
└─────────────────────────────────────────────────────────────┘
```

---

### **1. Clothing Classification Model**

**Architecture**: Transfer Learning with MobileNetV2

```python
Input: (224, 224, 3)
    ↓
MobileNetV2 (ImageNet pretrained, frozen initially)
    ↓ 1280-dim features
GlobalAveragePooling2D
    ↓
Dense(512, activation='relu')
    ↓
Dropout(0.5)
    ↓
Dense(256, activation='relu')
    ↓
Dropout(0.3)
    ↓
Dense(6, activation='softmax')  # 6 categories
```

**Training Strategy**:
- **Phase 1**: Freeze base, train head (5 epochs, LR=0.001)
- **Phase 2**: Unfreeze, fine-tune (15 epochs, LR=0.0001)

**Loss Function**: Categorical Cross-Entropy
$$
L = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)
$$

**Results**:
- **Train Accuracy**: 96.31%
- **Validation Accuracy**: 95.72%
- **Test Accuracy**: 95.48%
- **Training Time**: ~20 minutes on RTX 3050

**Dataset Split**:
- Train: 31,093 images (70%)
- Validation: 6,663 images (15%)
- Test: 6,663 images (15%)

---

### **2. Visual Feature Extraction**

#### **A. RGB Color Extraction (K-Means Clustering)**

Extract the 3 most dominant colors from each clothing image:

```python
Algorithm: K-Means (k=3)
Input: Image resized to 100x100 for speed
Process:
  1. Convert BGR → RGB
  2. Reshape to pixel list: (10000, 3)
  3. Apply K-means clustering
  4. Sort clusters by frequency
Output: 
  - color1_rgb: (R, G, B), percentage
  - color2_rgb: (R, G, B), percentage  
  - color3_rgb: (R, G, B), percentage
```

**Example**: Blue shirt → `(30, 50, 120)` 65%, `(200, 200, 200)` 25%, `(50, 50, 50)` 10%

#### **B. Pattern Detection**

**Method**: Edge Density + Directional Analysis

```python
Patterns Detected:
  • Solid: edge_density < 0.05 AND std_dev < 30
  • Striped_Horizontal: horizontal_edges >> vertical_edges
  • Striped_Vertical: vertical_edges >> horizontal_edges
  • Checkered: horizontal_edges ≈ vertical_edges AND edge_density > 0.15
  • Dotted: blob_count > 10 (circular patterns)
  • Floral: std_dev > 50 AND edge_density > 0.2
  • Textured: edge_density > 0.1 (catch-all)
```

**Edge Ratio Formula**:
$$
\text{edge\_ratio} = \frac{|E_h - E_v|}{E_h + E_v + \epsilon}
$$

Where:
- $E_h$ = horizontal edge strength (Sobel X)
- $E_v$ = vertical edge strength (Sobel Y)
- $\epsilon$ = small constant (1e-6) to prevent division by zero

**Pattern Clash Detection**:
- Striped + Checkered = ❌
- Floral + Checkered = ❌
- Solid + Any = ✅

#### **C. Color Metrics**

1. **Brightness** (0-255):
$$
B = \frac{1}{3}(\text{mean}(C_1) + \text{mean}(C_2) + \text{mean}(C_3))
$$

2. **Color Diversity** (Euclidean distance):
$$
D = \frac{1}{3}(||C_1 - C_2|| + ||C_2 - C_3|| + ||C_1 - C_3||)
$$

3. **Temperature** (Warm vs Cool):
$$
T = 
\begin{cases}
\text{warm} & \text{if } \bar{R} > \bar{B} \\
\text{cool} & \text{otherwise}
\end{cases}
$$

4. **Saturation**:
$$
S = \frac{1}{3}\sum_{i=1}^{3} \text{std}(C_i - \text{mean}(C_i))
$$

---

### **3. Outfit Compatibility Model**

**Architecture**: 3-Input Siamese CNN

```python
Input 1: Top image (224, 224, 3)
Input 2: Bottom image (224, 224, 3)
Input 3: Shoes image (224, 224, 3)
    ↓ (shared feature extractor for all 3)
MobileNetV2 (frozen, ImageNet weights)
    ↓
Dense(128, activation='relu')
    ↓
Dropout(0.3)
    ↓
Dense(64, activation='relu')  # 64-dim feature vector per item
    ↓
Concatenate([f_top, f_bottom, f_shoes])  # 192-dim combined
    ↓
Dense(256, activation='relu')
    ↓
Dropout(0.4)
    ↓
Dense(128, activation='relu')
    ↓
Dropout(0.3)
    ↓
Dense(64, activation='relu')
    ↓
Dropout(0.2)
    ↓
Dense(1, activation='sigmoid')  # Compatibility score [0, 1]
```

**Training Configuration**:
- **Optimizer**: Adam (LR=0.0001)
- **Loss**: Binary Cross-Entropy
$$
L = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]
$$
- **Metrics**: Accuracy, AUC
- **Batch Size**: 32
- **Epochs**: 25 (with early stopping)
- **Callbacks**: ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

**Training Data Generation**:

For each outfit set, calculate rule-based compatibility before labeling:

```python
def outfit_score(top, bottom, shoes):
    color_score = (
        color_harmony(top, bottom) * 0.30 +
        color_harmony(top, shoes) * 0.10 +
        color_harmony(bottom, shoes) * 0.10
    )
    
    pattern_score = pattern_compatibility(top, bottom) * 0.40
    
    style_score = (1.0 if match(top.usage, bottom.usage) else 0.7) * 0.10
    
    return color_score + pattern_score + style_score

Label:
  • 1 (Good outfit) if score > 0.55
  • 0 (Bad outfit) if score < 0.50 OR has pattern clash
```

**Balanced Training**:
- **Positive samples**: 50% (well-coordinated outfits)
- **Negative samples**: 50% (clashing colors/patterns, style mismatches)

---

### **4. Color Harmony Calculation**

**RGB → HSV Conversion**:

$$
\begin{align*}
V &= \max(R, G, B) \\
S &= \begin{cases} 
0 & \text{if } V = 0 \\
\frac{V - \min(R,G,B)}{V} & \text{otherwise}
\end{cases} \\
H &= \begin{cases}
60 \times \frac{G-B}{V-\min} + 0 & \text{if } \max = R \\
60 \times \frac{B-R}{V-\min} + 120 & \text{if } \max = G \\
60 \times \frac{R-G}{V-\min} + 240 & \text{if } \max = B
\end{cases}
\end{align*}
$$

**Color Harmony Score**:

$$
\text{harmony}(C_1, C_2) = 
\begin{cases}
0.9 & \text{if } 150° \leq |\Delta H| \leq 210° \text{ (complementary)} \\
0.85 & \text{if } |\Delta H| \leq 60° \text{ (analogous)} \\
0.75 & \text{if } 100° \leq |\Delta H| \leq 140° \text{ (triadic)} \\
0.8 & \text{if } S_1 < 0.2 \text{ or } S_2 < 0.2 \text{ (neutral)} \\
0.5 & \text{otherwise (moderate)}
\end{cases}
$$

Where $\Delta H = \min(|H_1 - H_2|, 360 - |H_1 - H_2|)$

---

### **5. Pattern Compatibility Rules**

```python
pattern_compatibility(P1, P2):
    if P1 == 'solid' or P2 == 'solid':
        return 1.0  # Solid matches everything
    
    if P1 == P2:
        return 0.8  # Same patterns okay
    
    # Clash pairs (return 0.2 for bad combo)
    clashes = [
        (checkered, striped),
        (floral, checkered),
        (floral, striped),
        (dotted, striped)
    ]
    
    if (P1, P2) in clashes or (P2, P1) in clashes:
        return 0.2  # Strong penalty
    
    if 'textured' in [P1, P2]:
        return 0.7  # Textured is neutral
    
    return 0.6  # Default moderate compatibility
```

---

### **6. Gender-Specific Outfit Generation**

**Men's Outfits**:
```
- Type: Regular (Top + Bottom + Shoes)
- Top: Shirts, T-shirts, Sweatshirts, Jackets
- Bottom: Jeans, Trousers, Shorts
- Shoes: Casual shoes, Sneakers, Formal shoes
- Style Preference: Solid colors, minimal patterns
```

**Women's Outfits**:
```
- Type: Regular (60%) OR Dress (40%)
- Regular:
  - Top: Tops, Shirts, Blouses, Sweaters
  - Bottom: Jeans, Trousers, Skirts, Leggings
  - Shoes: Heels, Sandals, Sneakers
- Dress:
  - Dress: One-piece (solid/floral/textured)
  - Shoes: Heels, Sandals
- Style Preference: More colors, patterns allowed
```

---

### **7. Performance Metrics**

#### **Clothing Classifier**
| Metric | Value |
|--------|-------|
| Test Accuracy | **95.48%** |
| Val Accuracy | 95.72% |
| Train Accuracy | 96.31% |
| Parameters | 2,619,590 |
| Training Time | ~20 min (GPU) |
| Inference Time | ~50 ms/image |

#### **Compatibility Model**
| Metric | Target |
|--------|--------|
| Expected Test Accuracy | 70-85% |
| Expected AUC | 0.80-0.90 |
| Parameters | 2,791,361 |
| Training Time | ~15 min (GPU, cached images) |
| Inference Time | ~50 ms/outfit |

#### **Visual Feature Extraction**
| Operation | Time (per image) |
|-----------|------------------|
| Color Extraction (K-means) | ~300-400 ms |
| Pattern Detection | ~100-200 ms |
| **Total** | **~500 ms** |

**Batch Processing** (44,419 images):
- **With caching**: ~25 minutes total
- **On-the-fly**: Would take hours (too slow)

---

### **8. Real-World Usage Flow**

```python
# User uploads image from phone
uploaded_image = "user_photo.jpg"

# Step 1: Classify (50ms)
category = clothing_classifier.predict(uploaded_image)
# Output: "Topwear"

# Step 2: Extract features (500ms)
features = extract_visual_features(uploaded_image)
# Output: {
#   'color1_rgb': (45, 67, 123),
#   'color1_pct': 0.65,
#   'pattern': 'striped_vertical',
#   'brightness': 142,
#   'temperature': 'cool'
# }

# Step 3: Load wardrobe (100ms)
wardrobe = load_user_wardrobe(user_id, gender='male')
# Output: [47 items: 18 tops, 12 bottoms, 17 shoes]

# Step 4: Score all combinations (5s for 100 items)
outfits = []
for top in wardrobe['tops']:
    for bottom in wardrobe['bottoms']:
        for shoes in wardrobe['shoes']:
            score = compatibility_model.predict([top, bottom, shoes])
            outfits.append((top, bottom, shoes, score))

# Step 5: Rank and return (50ms)
outfits.sort(key=lambda x: x[3], reverse=True)
return outfits[:10]  # Top 10 recommendations
```

**Total Time**: ~6 seconds for 100 wardrobe items

---

### **9. Considerations & Edge Cases**

#### **Pattern Clash Examples**:
| Outfit | Verdict | Reason |
|--------|---------|--------|
| Striped shirt + Solid pants | ✅ Good | Solid balances pattern |
| Checkered shirt + Striped pants | ❌ Bad | Pattern overload |
| Floral dress + Solid sandals | ✅ Good | Solid doesn't compete |
| Striped shirt + Checkered tie | ❌ Bad | Visual chaos |

#### **Color Harmony Examples**:
| Color 1 | Color 2 | Harmony Type | Score |
|---------|---------|--------------|-------|
| Blue (240°) | Orange (30°) | Complementary | 0.9 |
| Red (0°) | Pink (340°) | Analogous | 0.85 |
| Blue (240°) | White (low sat) | Neutral pairing | 0.8 |
| Red (0°) | Green (120°) | Triadic | 0.75 |
| Red (0°) | Purple (280°) | Moderate | 0.5 |

#### **Gender Separation**:
- Prevents suggesting women's dresses for men
- Allows unisex items (t-shirts, sneakers) for both
- Different style preferences

#### **Dress Handling**:
```python
# Dresses use same image for both top & bottom slots
outfit = {
    'top': dress_image,      # Same
    'bottom': dress_image,   # Same
    'shoes': shoes_image
}
```

#### **Missing Images**:
- Fallback to zero tensor: `np.zeros((224, 224, 3))`
- Model trained to handle this gracefully

---

### **10. Training Loss & Accuracy Progression**

**Expected Training Curve** (Compatibility Model):

```
Epoch 1:  Loss: 0.65  |  Acc: 62%  |  Val_Loss: 0.61  |  Val_Acc: 65%
Epoch 5:  Loss: 0.48  |  Acc: 74%  |  Val_Loss: 0.52  |  Val_Acc: 73%
Epoch 10: Loss: 0.38  |  Acc: 81%  |  Val_Loss: 0.45  |  Val_Acc: 78%
Epoch 15: Loss: 0.32  |  Acc: 85%  |  Val_Loss: 0.42  |  Val_Acc: 80%
Epoch 20: Loss: 0.28  |  Acc: 87%  |  Val_Loss: 0.41  |  Val_Acc: 81%
```

**Preventing Overfitting**:
- Dropout layers (0.2-0.4)
- Early stopping (patience=7)
- Balanced positive/negative samples
- Data augmentation through shuffling

---

### **11. Key Formulas Summary**

| Component | Formula |
|-----------|---------|
| **Classification Loss** | $L = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$ |
| **Compatibility Loss** | $L = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$ |
| **Color Harmony** | $\Delta H = \min(\|H_1 - H_2\|, 360 - \|H_1 - H_2\|)$ |
| **Edge Ratio** | $r = \frac{\|E_h - E_v\|}{E_h + E_v + \epsilon}$ |
| **Brightness** | $B = \frac{1}{3}(\text{mean}(C_1) + \text{mean}(C_2) + \text{mean}(C_3))$ |
| **Outfit Score** | $S = 0.35 C_{\text{color}} + 0.35 C_{\text{pattern}} + 0.2 C_{\text{style}} + 0.1 C_{\text{brightness}}$ |

---

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
