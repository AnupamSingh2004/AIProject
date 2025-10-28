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

2. **👔 Advanced Outfit Compatibility Model (78.28% Accuracy, 88.09% AUC)**
   - **Architecture**: 3-input Siamese CNN (3.16M parameters)
   - **Inputs**: Top image + Bottom image + Shoes image (224x224x3 each)
   - **Output**: Compatibility score (0-1, sigmoid activation)
   - **Performance Metrics** (Test Set):
     - Accuracy: **78.28%**
     - AUC-ROC: **88.09%**
     - Precision: **83.20%**
     - Recall: **70.87%**
     - F1-Score: **76.54%**
   - **Features Considered**:
     - RGB color harmony (complementary, analogous, monochromatic)
     - Pattern clash detection (striped+checkered penalty: -0.4)
     - Gender separation (Men's/Women's outfits)
     - Occasion matching (Casual, Formal, Sports, Party, Ethnic)
   - **Training Strategy**: Balanced 50/50 positive/negative outfits, strict 70% threshold
   - **Training Time**: ~13 minutes on RTX 3050 (35 epochs, early stopping)

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
python train_classifier_simple.py

# Train advanced outfit compatibility model (3-input Siamese CNN)
python train_compatibility_advanced.py
```

**Training Output** (Compatibility Model):
- Creates: `models/saved_models/outfit_compatibility_advanced.keras` (21 MB)
- History: `models/saved_models/compatibility_advanced_history.json`
- Training time: ~13 minutes on RTX 3050 GPU
- Expected metrics: 78% accuracy, 88% AUC

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
| Metric | Value |
|--------|-------|
| **Test Accuracy** | **78.28%** ✅ |
| **Test AUC-ROC** | **88.09%** ✅ |
| **Precision** | **83.20%** |
| **Recall** | **70.87%** |
| **F1-Score** | **76.54%** |
| Val Accuracy (Best) | 81.19% |
| Val AUC (Best) | 91.16% |
| Parameters | 3,165,697 |
| Training Time | ~13 min (GPU, 35 epochs) |
| Training Pairs | 8,000 (50/50 balanced) |
| Inference Time | ~50 ms/outfit |
| Early Stopping | Epoch 25 (patience 10) |

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

**Actual Training Results** (Compatibility Model):

```
Epoch 1:  Loss: 17.73 |  Acc: 51.41% |  Val_Loss: 17.43 |  Val_Acc: 60.50% |  Val_AUC: 65.79%
Epoch 5:  Loss: 16.23 |  Acc: 66.30% |  Val_Loss: 15.55 |  Val_Acc: 75.38% |  Val_AUC: 81.48%
Epoch 10: Loss: 13.97 |  Acc: 73.59% |  Val_Loss: 13.37 |  Val_Acc: 76.88% |  Val_AUC: 87.41%
Epoch 15: Loss: 10.84 |  Acc: 77.80% |  Val_Loss: 10.33 |  Val_Acc: 80.44% |  Val_AUC: 89.95%
Epoch 20: Loss: 8.97  |  Acc: 79.94% |  Val_Loss: 8.53  |  Val_Acc: 80.87% |  Val_AUC: 90.49%
Epoch 25: Loss: 6.60  |  Acc: 81.41% |  Val_Loss: 6.28  |  Val_Acc: 81.19% |  Val_AUC: 91.16% ⭐
Epoch 35: Early stopping triggered (best model from epoch 25)
```

**Final Test Performance**:
- **Accuracy**: 78.28%
- **AUC-ROC**: 88.09%
- **Precision**: 83.20%
- **Recall**: 70.87%
- **F1-Score**: 76.54%

**Preventing Overfitting**:
- Dropout layers (0.2-0.4)
- Early stopping (patience=10, triggered at epoch 35)
- Balanced positive/negative samples (50/50)
- Strict compatibility threshold (70% for positive outfits)
- L2 regularization on MobileNetV2 features

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

### **12. Complete Image Processing Flow (Detailed)**

This section explains exactly what happens to an image from upload to final outfit recommendation, covering every transformation, model layer, and operation.

#### **Step 1: Raw Image Upload**
- **Input Format**: User uploads an image (JPEG/PNG) from phone camera or gallery
- **Data Structure**: Image loaded as numpy array with shape `(H, W, 3)` where:
  - `H` = height in pixels (variable, e.g., 3024 for phone photos)
  - `W` = width in pixels (variable, e.g., 4032 for phone photos)
  - `3` = RGB color channels (Red, Green, Blue)
  - Data type: `uint8` (values 0-255)
- **File Size**: Typically 2-8 MB for phone photos
- **Color Space**: sRGB (standard RGB)

#### **Step 2: Image Preprocessing**

**2.1 Resize Operation**
```python
# OpenCV resize with bilinear interpolation
img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
# Shape: (3024, 4032, 3) → (224, 224, 3)
```
- **Method**: Bilinear interpolation (weighted average of 4 nearest pixels)
- **Output**: Fixed 224×224 size required by MobileNetV2 architecture
- **Computation**: ~0.5ms on CPU

**2.2 Normalization**
```python
# Convert pixel values from [0, 255] to [0.0, 1.0]
img_normalized = img_resized.astype('float32') / 255.0
# uint8 [0, 255] → float32 [0.0, 1.0]
```
- **Purpose**: Neural networks perform better with normalized inputs (0-1 range)
- **Effect**: Black pixel (0, 0, 0) → (0.0, 0.0, 0.0), White pixel (255, 255, 255) → (1.0, 1.0, 1.0)

**2.3 Color Space Handling**
- **OpenCV Quirk**: cv2.imread() loads images in BGR order (Blue, Green, Red) instead of RGB
- **Conversion**: `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` swaps channels
- **Result**: Standard RGB format for model input

#### **Step 3: Clothing Classification (MobileNetV2)**

**3.1 Model Architecture Breakdown**

The image passes through MobileNetV2, a highly efficient CNN designed for mobile devices:

**Layer 1: Initial Convolution**
```
Input: (224, 224, 3) [RGB image]
↓ Conv2D(32 filters, 3×3 kernel, stride=2, ReLU6 activation)
Output: (112, 112, 32) [32 feature maps]
```
- **Operation**: Each 3×3 kernel slides over image, computing weighted sum of pixels
- **Effect**: Detects basic edges and color gradients
- **Reduction**: Spatial size halved (224→112) due to stride=2

**Layer 2-17: Inverted Residual Blocks (Bottleneck Architecture)**

Each block follows this pattern:
```
Input: (H, W, C) [C channels]
↓ 1×1 Conv (Expand): Increase channels from C to 6C
↓ Activation: ReLU6 (clips values at 6.0)
↓ 3×3 Depthwise Conv: Apply separate filter per channel
↓ Activation: ReLU6
↓ 1×1 Conv (Project): Reduce channels back to C'
↓ Skip Connection: Add input if C = C' (residual learning)
Output: (H', W', C') [Compressed features]
```

**Key Blocks:**
- **Block 1-3**: Extract low-level features (edges, textures) → (112, 112) → (56, 56)
- **Block 4-7**: Mid-level features (patterns, shapes) → (56, 56) → (28, 28)
- **Block 8-14**: High-level features (clothing types, styles) → (28, 28) → (14, 14)
- **Block 15-17**: Abstract semantic features → (14, 14) → (7, 7)

**Final Feature Extraction:**
```
Input: (7, 7, 1280) [1280 feature maps]
↓ GlobalAveragePooling2D: Average each 7×7 feature map to single value
Output: (1280,) [1D feature vector]
↓ Dense(128, activation='relu', kernel_regularizer=L2(0.01))
Output: (128,) [Compressed embedding]
↓ BatchNormalization: Normalize activations (mean=0, std=1)
↓ Dropout(0.5): Randomly set 50% of values to 0 (training only)
```

**3.2 Classification Head**
```
Input: (128,) [Image embedding]
↓ Dense(6, activation='softmax')
Output: (6,) [Probability distribution]
```

**Softmax Operation:**
$$
P(\text{class}_i) = \frac{e^{z_i}}{\sum_{j=1}^{6} e^{z_j}}
$$

**Output Example:**
```python
[0.78, 0.12, 0.05, 0.03, 0.01, 0.01]  # Probabilities sum to 1.0
# Topwear: 78%, Bottomwear: 12%, Footwear: 5%, ...
```

**3.3 Predicted Category**
- **Decision**: `argmax(probabilities)` → Index of highest probability
- **Mapping**: `{0: 'Topwear', 1: 'Bottomwear', 2: 'Footwear', 3: 'Dress', 4: 'Accessories', 5: 'Other'}`
- **Confidence Threshold**: Only accept if max probability > 0.6 (60%)
- **Processing Time**: ~50ms on GPU, ~200ms on CPU

#### **Step 4: Visual Feature Extraction**

**4.1 Dominant Color Extraction (K-Means Clustering)**

**Algorithm:**
```python
# Step 1: Reshape image from (224, 224, 3) to (50176, 3)
pixels = img.reshape(-1, 3)  # Flatten spatial dimensions
# Each row = one pixel's [R, G, B] values

# Step 2: K-Means clustering (k=3)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(pixels)
# Finds 3 cluster centers that minimize within-cluster variance
```

**Mathematics:**
$$
\min_{C_1, C_2, C_3} \sum_{i=1}^{3} \sum_{p \in \text{cluster}_i} \|p - C_i\|^2
$$
where $C_i$ = cluster center (dominant color RGB), $p$ = pixel RGB value

**Output:**
```python
colors = [
    [180, 120, 85],   # Color 1 (dominant): Brown
    [220, 200, 190],  # Color 2: Light beige
    [50, 30, 20]      # Color 3 (least): Dark brown
]
percentages = [0.65, 0.25, 0.10]  # How much of image each color covers
```

**4.2 Pattern Detection**

**Grayscale Conversion:**
```python
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# Formula: Gray = 0.299*R + 0.587*G + 0.114*B (weighted by human perception)
```

**Edge Detection (Canny Algorithm):**
```python
edges = cv2.Canny(gray, threshold1=50, threshold2=150)
# Step 1: Gaussian blur to reduce noise
# Step 2: Compute gradients (Sobel operators)
# Step 3: Non-maximum suppression (thin edges)
# Step 4: Double thresholding (weak/strong edges)
# Step 5: Edge tracking by hysteresis
```

**Hough Line Transform (Detect Straight Lines):**
```python
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=30)
# Detects lines in polar coordinates (ρ, θ)
# Returns list of line segments: [[x1, y1, x2, y2], ...]
```

**Directional Analysis:**
```python
for line in lines:
    x1, y1, x2, y2 = line[0]
    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
    if -15 < angle < 15:          # Horizontal
        horizontal_count += 1
    elif 75 < angle < 105:        # Vertical
        vertical_count += 1

# Decision logic:
if horizontal_count > 10 and horizontal_count / total_lines > 0.6:
    pattern = "striped_horizontal"
elif vertical_count > 10 and vertical_count / total_lines > 0.6:
    pattern = "striped_vertical"
```

**Blob Detection (Dotted Patterns):**
```python
params = cv2.SimpleBlobDetector_Params()
params.filterByCircularity = True
params.minCircularity = 0.7  # Dots are circular
detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(edges)

if len(keypoints) > 20:  # Many circular regions
    pattern = "dotted"
```

**Texture Analysis (Floral/Textured):**
```python
# Variance of Laplacian (measures texture richness)
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
texture_variance = laplacian.var()

if texture_variance > 500:  # High variance = complex texture
    pattern = "textured" or "floral"
else:
    pattern = "solid"  # Low variance = plain/solid color
```

**4.3 Color Metrics**

**Brightness:**
```python
brightness = np.mean(gray)  # Average pixel intensity [0-255]
# Bright image: ~200, Dark image: ~50
```

**Color Diversity (Standard Deviation):**
```python
color_diversity = np.std(img)  # How varied the colors are
# High diversity (~200): Multi-colored, Low diversity (~20): Monochrome
```

**Temperature (Warm vs Cool):**
```python
avg_red = np.mean(img[:, :, 0])
avg_blue = np.mean(img[:, :, 2])

if avg_red > avg_blue:
    temperature = "warm"  # Red, orange, yellow dominant
else:
    temperature = "cool"  # Blue, green, purple dominant
```

**Saturation (Color Intensity):**
```python
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
saturation = np.mean(hsv[:, :, 1])  # Average S channel [0-255]
# High saturation (~200): Vibrant colors, Low saturation (~20): Dull/grayish
```

#### **Step 5: Outfit Compatibility Scoring (3-Input Siamese CNN)**

**5.1 Input Preparation**

User's wardrobe contains classified items:
```python
wardrobe = {
    'Topwear': [img_top1, img_top2, ...],      # e.g., 10 tops
    'Bottomwear': [img_bottom1, img_bottom2, ...],  # e.g., 8 bottoms
    'Footwear': [img_shoe1, img_shoe2, ...]    # e.g., 6 shoes
}
```

Generate all valid outfit combinations:
```python
# For complete outfits (top + bottom + shoes):
outfits = []
for top in wardrobe['Topwear']:
    for bottom in wardrobe['Bottomwear']:
        for shoes in wardrobe['Footwear']:
            outfits.append((top, bottom, shoes))
# Total combinations: 10 × 8 × 6 = 480 outfits
```

**5.2 Siamese CNN Architecture**

**Feature Extraction (Shared MobileNetV2 × 3):**
```
Input 1 (Top):    (224, 224, 3) → MobileNetV2 → GlobalAvgPool → Dense(128) → L2Norm → feat1 (128,)
Input 2 (Bottom): (224, 224, 3) → MobileNetV2 → GlobalAvgPool → Dense(128) → L2Norm → feat2 (128,)
Input 3 (Shoes):  (224, 224, 3) → MobileNetV2 → GlobalAvgPool → Dense(128) → L2Norm → feat3 (128,)
```

**Note**: All 3 inputs share the **same MobileNetV2 weights** (Siamese architecture)

**L2 Normalization:**
$$
\text{feat}_{\text{normalized}} = \frac{\text{feat}}{\|\text{feat}\|_2} = \frac{\text{feat}}{\sqrt{\sum_{i=1}^{128} \text{feat}_i^2}}
$$
- **Purpose**: Makes features unit-length, focuses on direction (not magnitude)

**5.3 Feature Fusion**

Three complementary operations capture different interaction patterns:

**Concatenation (Direct Features):**
```python
concat = tf.concat([feat1, feat2, feat3], axis=-1)
# Shape: (128,) + (128,) + (128,) = (384,)
# Preserves individual item characteristics
```

**Difference (Contrast Between Items):**
```python
diff_12 = tf.abs(feat1 - feat2)  # Top vs Bottom
diff_13 = tf.abs(feat1 - feat3)  # Top vs Shoes
diff_23 = tf.abs(feat2 - feat3)  # Bottom vs Shoes
diff = tf.concat([diff_12, diff_13, diff_23], axis=-1)
# Shape: (128,) + (128,) + (128,) = (384,)
# Captures dissimilarity between pairs
```

**Product (Interaction Between Items):**
```python
prod_12 = feat1 * feat2  # Element-wise multiplication
prod_13 = feat1 * feat3
prod_23 = feat2 * feat3
prod = tf.concat([prod_12, prod_13, prod_23], axis=-1)
# Shape: (128,) + (128,) + (128,) = (384,)
# Captures co-occurrence patterns
```

**Combined Feature Vector:**
```python
combined = tf.concat([concat, diff, prod], axis=-1)
# Shape: (384,) + (384,) + (384,) = (1152,) → Actually (640,) after optimization
```

**5.4 Compatibility Scoring Head**
```
Input: (640,) [Fused features]
↓ Dense(256, activation='relu')
↓ Dropout(0.4)  # Prevent overfitting
↓ Dense(128, activation='relu')
↓ Dropout(0.3)
↓ Dense(64, activation='relu')
↓ Dense(1, activation='sigmoid')
Output: (1,) [Compatibility score 0.0-1.0]
```

**Sigmoid Activation:**
$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$
- **Output Range**: (0, 1)
- **Interpretation**: Probability that outfit is compatible
  - Score > 0.7: High compatibility (good outfit)
  - Score 0.4-0.7: Medium compatibility (acceptable)
  - Score < 0.4: Low compatibility (poor outfit)

**5.5 Batch Processing**
```python
# Score all 480 outfits efficiently in batches
scores = model.predict([all_tops, all_bottoms, all_shoes], batch_size=32)
# Processing time: ~480 outfits × 50ms / 32 batch = ~750ms total
```

#### **Step 6: Color Harmony Validation**

**Extract Colors from Predictions:**
```python
top_colors = kmeans_top.cluster_centers_      # 3 dominant RGB colors
bottom_colors = kmeans_bottom.cluster_centers_
shoe_colors = kmeans_shoe.cluster_centers_
```

**Convert RGB to HSV (Hue-Saturation-Value):**
```python
top_hsv = rgb_to_hsv(top_colors[0])     # Use most dominant color
bottom_hsv = rgb_to_hsv(bottom_colors[0])
shoe_hsv = rgb_to_hsv(shoe_colors[0])
```

**Hue Difference (Color Wheel Distance):**
```python
hue_diff = abs(top_hsv[0] - bottom_hsv[0])
if hue_diff > 180:
    hue_diff = 360 - hue_diff  # Wrap around color wheel
```

**Harmony Rules:**

**Complementary Colors (Opposite on Color Wheel):**
$$
\text{complementary} = \begin{cases}
1.0 & \text{if } 160° \leq |\Delta \text{hue}| \leq 200° \\
0.0 & \text{otherwise}
\end{cases}
$$
Example: Blue (240°) + Orange (60°) → Δ180° → Complementary

**Analogous Colors (Adjacent on Color Wheel):**
$$
\text{analogous} = \begin{cases}
1.0 & \text{if } |\Delta \text{hue}| \leq 30° \\
0.0 & \text{otherwise}
\end{cases}
$$
Example: Blue (240°) + Blue-green (210°) → Δ30° → Analogous

**Warm/Cool Balance:**
```python
warm_count = sum([1 for item in [top, bottom, shoe] if item['temperature'] == 'warm'])
cool_count = 3 - warm_count

balance_score = (warm_count + cool_count) / 3.0
# Perfect: 2 warm + 1 cool = 1.0
# Unbalanced: 3 warm + 0 cool = 0.67
```

**Final Color Compatibility:**
$$
C_{\text{color}} = 0.4 \times \text{complementary} + 0.4 \times \text{analogous} + 0.2 \times \text{balance}
$$

#### **Step 7: Pattern Clash Detection**

**Pattern Clash Matrix:**
```python
CLASH_PENALTIES = {
    ('striped_horizontal', 'checkered'): -0.4,
    ('striped_vertical', 'checkered'): -0.4,
    ('striped_horizontal', 'dotted'): -0.2,
    ('striped_vertical', 'dotted'): -0.2,
}

# Check all pairs
pattern_score = 0.0
if (top_pattern, bottom_pattern) in CLASH_PENALTIES:
    pattern_score += CLASH_PENALTIES[(top_pattern, bottom_pattern)]
if (top_pattern, shoe_pattern) in CLASH_PENALTIES:
    pattern_score += CLASH_PENALTIES[(top_pattern, shoe_pattern)]
if (bottom_pattern, shoe_pattern) in CLASH_PENALTIES:
    pattern_score += CLASH_PENALTIES[(bottom_pattern, shoe_pattern)]

# Multiple patterns penalty
pattern_count = len(set([top_pattern, bottom_pattern, shoe_pattern]) - {'solid'})
if pattern_count > 2:
    pattern_score -= 0.3
```

#### **Step 8: Occasion & Gender Filtering**

**Gender Validation:**
```python
# Extract gender from metadata (Men/Women/Unisex)
if not (top_gender == bottom_gender == shoe_gender):
    score = 0.0  # Reject cross-gender outfits
```

**Occasion Matching:**
```python
COMPATIBLE_OCCASIONS = {
    'Casual': ['Casual', 'Smart Casual'],
    'Formal': ['Formal', 'Party'],
    'Sports': ['Sports', 'Casual'],
}

if item3_occasion not in COMPATIBLE_OCCASIONS.get(item1_occasion, []):
    score *= 0.5  # Penalize occasion mismatch
```

#### **Step 9: Final Ranking & Output**

**Combine All Scores:**
$$
\text{Final Score} = 0.5 \times \text{CNN Score} + 0.25 \times C_{\text{color}} + 0.15 \times C_{\text{pattern}} + 0.1 \times C_{\text{occasion}}
$$

**Sort Outfits:**
```python
outfits_ranked = sorted(outfits, key=lambda x: x['final_score'], reverse=True)
top_10_recommendations = outfits_ranked[:10]
```

**Output to User:**
```json
{
  "outfit_id": 1,
  "items": {
    "top": "Blue Oxford Shirt",
    "bottom": "Khaki Chinos",
    "shoes": "Brown Leather Loafers"
  },
  "compatibility_score": 0.87,
  "color_harmony": "complementary",
  "pattern_check": "no_clash",
  "occasion": "Smart Casual",
  "reasoning": "Complementary blue/brown pairing, solid patterns avoid clash, appropriate for business casual."
}
```

**Total Processing Time (End-to-End):**
- Image upload & preprocessing: ~10ms
- Clothing classification: ~50ms (GPU) per image
- Feature extraction: ~30ms per image
- Compatibility scoring: ~750ms for 480 outfits (batch)
- Ranking & filtering: ~20ms
- **Total**: ~1.5 seconds for complete wardrobe analysis

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
