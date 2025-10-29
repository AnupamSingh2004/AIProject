# 📓 Jupyter Notebooks - AI Fashion Recommender

This folder contains comprehensive Jupyter notebooks for training, testing, and analyzing the AI Fashion Recommender models.

## 📚 Notebooks Overview

### 1️⃣ `1_clothing_classifier_training.ipynb`
**Clothing Classification Model Training**

**What it does:**
- Trains a MobileNetV2-based clothing classifier
- Classifies clothing into 6 categories (Topwear, Bottomwear, Footwear, etc.)
- Uses 2-phase training: frozen base → fine-tuning
- Expected accuracy: **95%+**

**Key Features:**
- ✅ Complete data loading and preprocessing
- ✅ Data augmentation visualization
- ✅ Model architecture diagram
- ✅ Training progress graphs (loss, accuracy)
- ✅ Confusion matrix
- ✅ Per-class metrics (precision, recall, F1-score)
- ✅ Model saving and history export

**Visualizations:**
- Class distribution across train/val/test sets
- Sample training images with augmentation
- Phase 1 & Phase 2 training curves
- Combined training history
- Confusion matrix heatmap
- Per-class performance metrics

**Runtime:** ~30-40 minutes (with GPU), ~2-3 hours (CPU only)

---

### 2️⃣ `2_outfit_compatibility_training.ipynb`
**Advanced Outfit Compatibility Model Training**

**What it does:**
- Trains a 3-input Siamese CNN for outfit compatibility
- Takes Top + Bottom + Shoes images as input
- Predicts compatibility score (0-1)
- Uses color harmony, pattern detection, and occasion matching
- Expected performance: **78%+ accuracy, 88%+ AUC**

**Key Features:**
- ✅ Visual feature extraction (RGB colors, patterns)
- ✅ Color harmony calculation (HSV-based)
- ✅ Pattern compatibility rules
- ✅ Occasion matching logic
- ✅ Gender-specific outfit generation
- ✅ Balanced 50/50 positive/negative samples
- ✅ Multi-factor scoring visualization

**Visualizations:**
- Category, gender, pattern, usage distributions
- Color harmony examples with scores
- Training history (loss, accuracy, AUC, precision, recall)
- Compatibility scoring weight distribution
- Model interpretation diagrams

**Runtime:** ~40-60 minutes (with GPU), ~4-6 hours (CPU only)

---

### 3️⃣ `3_model_testing_and_inference.ipynb`
**Model Testing & Performance Analysis**

**What it does:**
- Loads trained models
- Tests with sample images
- Visualizes predictions
- Benchmarks inference speed
- Analyzes compatibility score distributions

**Key Features:**
- ✅ Model loading and verification
- ✅ Single image classification
- ✅ Outfit compatibility prediction
- ✅ Batch performance analysis
- ✅ Inference speed benchmarking
- ✅ Custom image testing

**Visualizations:**
- Classification results with probability bars
- Outfit compatibility scores with images
- Score distribution histograms
- Inference speed comparisons
- Real-time prediction displays

**Runtime:** ~5-10 minutes

---

## 🚀 How to Use

### Prerequisites

```powershell
# Install Jupyter
pip install jupyter notebook

# Install required packages
pip install tensorflow keras matplotlib seaborn pandas numpy pillow scikit-learn
```

### Starting Jupyter

```powershell
# Navigate to project root
cd C:\Users\Prachi\Desktop\qq\AIProject

# Start Jupyter
jupyter notebook
```

Your browser will open at `http://localhost:8888`

### Running Notebooks

1. **Open any notebook** from the `notebooks/` folder
2. **Run cells sequentially** using `Shift + Enter`
3. **Wait for each cell** to complete before running the next
4. **View visualizations** inline as they are generated

---

## 📊 What Graphs You'll See

### Clothing Classifier Notebook:

1. **Class Distribution Charts** (Bar charts)
   - Train/Val/Test set distributions
   - Category counts with labels

2. **Sample Images Grid** (2×4 grid)
   - Training images with augmentation
   - Category labels

3. **Training Curves** (Line plots)
   - Phase 1: Loss & Accuracy
   - Phase 2: Loss & Accuracy
   - Combined view across both phases

4. **Confusion Matrix** (Heatmap)
   - True vs Predicted labels
   - Color-coded counts

5. **Per-Class Metrics** (Bar charts)
   - Precision, Recall, F1-Score
   - Horizontal bars for easy comparison

---

### Outfit Compatibility Notebook:

1. **Data Distribution** (Multiple charts)
   - Category distribution (bar)
   - Gender distribution (pie)
   - Pattern distribution (horizontal bar)
   - Usage/Occasion distribution (horizontal bar)

2. **Color Harmony Examples** (Image + Score grid)
   - Color pair swatches
   - Harmony scores with bar charts
   - Examples: Complementary, Analogous, Neutral, Clashing

3. **Training Curves** (4-panel grid)
   - Loss (train vs val)
   - Accuracy (train vs val)
   - AUC (train vs val)
   - Precision & Recall (4 lines)

4. **Model Interpretation** (2-panel)
   - Compatibility weight distribution (pie chart)
   - Training data balance (bar chart)

5. **Test Results** (Bar chart)
   - All metrics: Accuracy, AUC, Precision, Recall, F1-Score
   - Color-coded bars with value labels

---

### Model Testing Notebook:

1. **Classification Results** (Grid layout)
   - Image + Probability distribution bars
   - True vs Predicted labels
   - Color-coded correctness

2. **Outfit Compatibility** (4-column grid)
   - Top, Bottom, Shoes images
   - Compatibility score bar
   - Recommendation with color coding

3. **Score Distribution** (2-panel)
   - Histogram with threshold lines
   - Pie chart (Excellent/Good/Poor)

4. **Inference Speed** (Bar chart)
   - Classifier vs Compatibility model
   - Time in milliseconds

---

## 📁 Output Files

Notebooks generate these files:

```
models/saved_models/
├── clothing_classifier.keras           # Final classifier
├── classifier_phase1_best.keras        # Phase 1 checkpoint
├── classifier_phase2_best.keras        # Phase 2 checkpoint
├── outfit_compatibility_advanced.keras # Final compatibility model
├── history.json                        # Classifier training history
└── compatibility_advanced_history.json # Compatibility training history

notebooks/
├── clothing_classifier_architecture.png    # Model diagram
└── outfit_compatibility_architecture.png   # Model diagram
```

---

## 🎨 Customization

### Modify Training Parameters:

```python
# In clothing classifier notebook
EPOCHS1 = 15  # Change to train longer
EPOCHS2 = 15
BATCH_SIZE = 32  # Adjust based on GPU memory

# In compatibility notebook
N_TRAIN_PAIRS = 8000   # More pairs = better training
N_VAL_PAIRS = 1600
MAX_EPOCHS = 100
```

### Change Visualizations:

```python
# Change plot style
plt.style.use('seaborn-v0_8-darkgrid')  # Options: 'ggplot', 'dark_background'

# Modify colors
sns.set_palette("husl")  # Options: "Set2", "Paired", "coolwarm"

# Adjust figure sizes
fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # Change (16, 12)
```

---

## 🔧 Troubleshooting

### Out of Memory Error:

```python
# Reduce batch size
BATCH_SIZE = 16  # Instead of 32

# Or enable memory growth
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

### Notebook Kernel Dies:

```powershell
# Restart kernel in Jupyter: Kernel → Restart
# Or restart Jupyter server:
jupyter notebook stop
jupyter notebook
```

### Plots Not Showing:

```python
# Add this at top of notebook
%matplotlib inline

# Force display
plt.show()
```

### Slow Training:

```python
# Enable GPU if available
# Check GPU status:
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU: {gpus}")

# If no GPU, reduce dataset size for testing:
train_df = train_df.sample(frac=0.1)  # Use 10% of data
```

---

## 📖 Learning Resources

### Understanding the Models:

1. **MobileNetV2**: Lightweight CNN for mobile/edge devices
   - Paper: [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)

2. **Siamese Networks**: Neural networks with shared weights
   - Use case: Similarity learning
   - Our application: 3-input outfit compatibility

3. **Transfer Learning**: Using pre-trained ImageNet weights
   - Faster training
   - Better performance with less data

4. **Color Theory**: HSV color wheel
   - Complementary: Opposite colors (180°)
   - Analogous: Adjacent colors (30-60°)
   - Monochromatic: Same hue, different values

---

## 🎯 Next Steps

After running notebooks:

1. **Verify Models**:
   ```powershell
   python verify_model_works.py
   ```

2. **Start Backend**:
   ```powershell
   cd backend
   python start_backend.py
   ```

3. **Test API**:
   - Open http://localhost:8000/docs
   - Try `/api/test-model` endpoint

4. **Deploy**:
   - Use saved `.keras` files in production
   - Load with `keras.models.load_model(path)`

---

## 🤝 Contributing

Want to improve notebooks?

1. Add new visualizations
2. Improve training performance
3. Add data analysis sections
4. Create new notebooks for:
   - Data preprocessing
   - Hyperparameter tuning
   - Model comparison
   - Error analysis

---

## 📧 Support

Having issues?

1. Check [TROUBLESHOOTING.md](../TROUBLESHOOTING.md)
2. Review cell outputs for error messages
3. Verify data files exist in `data/processed/`
4. Ensure models are saved in `models/saved_models/`

---

## ✨ Features Summary

| Notebook | Training Time | Output | Graphs |
|----------|--------------|--------|--------|
| **Clothing Classifier** | 30-40 min (GPU) | 95%+ accuracy | 6+ visualizations |
| **Outfit Compatibility** | 40-60 min (GPU) | 78%+ accuracy, 88%+ AUC | 8+ visualizations |
| **Model Testing** | 5-10 min | Inference results | 5+ visualizations |

---

**🎉 Happy Training!** 

Run notebooks sequentially for the complete training pipeline. All visualizations are generated automatically with detailed graphs and metrics.
