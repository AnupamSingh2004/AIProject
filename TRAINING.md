# 🎓 Model Training Guide

This guide explains how to train the AI Fashion Recommendation System models from scratch.

## 📋 Overview

The system consists of two main models:

1. **Clothing Classification Model**: Classifies clothing items into categories (Topwear, Bottomwear, Dress, Footwear, Accessories)
2. **Outfit Compatibility Model**: Scores how well two clothing items work together

## 🔧 Prerequisites

### 1. Environment Setup

```bash
# Activate the AI conda environment
conda activate AI

# Verify installations
python -c "import tensorflow; print('TensorFlow:', tensorflow.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
```

### 2. Dataset Download

Download the Kaggle Fashion Product Images dataset:

```bash
# Set up Kaggle API credentials first
# Place kaggle.json at ~/.kaggle/kaggle.json

python scripts/download_dataset.py
```

The dataset should be in `data/raw/` directory with structure:
```
data/raw/
├── styles.csv
└── images/
    ├── 1000.jpg
    ├── 1001.jpg
    └── ...
```

## 🚀 Training Pipeline

### Method 1: Complete Pipeline (Recommended)

Run the entire training pipeline with one command:

```bash
python train.py
```

This will:
1. Preprocess the dataset
2. Train the clothing classifier
3. Train the outfit compatibility model

### Method 2: Step-by-Step Training

#### Step 1: Data Preprocessing

```bash
python scripts/preprocess_data.py \
    --data_dir data/raw \
    --output_dir data/processed \
    --img_size 224 \
    --extract_colors \
    --color_samples 5000
```

**Options:**
- `--data_dir`: Raw dataset directory
- `--output_dir`: Where to save processed data
- `--img_size`: Target image size (224 for most models)
- `--extract_colors`: Extract color features (optional, slower)
- `--color_samples`: Number of samples for color extraction

**Output:**
- `data/processed/train.csv`: Training set
- `data/processed/val.csv`: Validation set
- `data/processed/test.csv`: Test set
- `data/processed/label_mapping.json`: Category label mappings
- `data/processed/dataset_summary.json`: Dataset statistics

#### Step 2: Train Clothing Classifier

```bash
cd models
python clothing_classifier.py
```

Or with the main training script:

```bash
python train.py --skip_preprocessing --skip_compatibility
```

**Training Process:**
- **Phase 1** (10 epochs): Train with frozen base model
- **Phase 2** (10 epochs): Fine-tune the last layers

**Model Architecture:**
- Base: EfficientNetB0 (pretrained on ImageNet)
- Custom head for clothing classification
- Dropout and batch normalization for regularization

**Expected Results:**
- Training Accuracy: ~85-90%
- Validation Accuracy: ~80-85%
- Top-3 Accuracy: ~95%

#### Step 3: Train Outfit Compatibility Model

```bash
cd models
python outfit_compatibility_model.py
```

Or:

```bash
python train.py --skip_preprocessing --skip_classifier
```

**Training Process:**
- Creates positive pairs (compatible outfits)
- Creates negative pairs (incompatible outfits)
- Trains siamese-style network to score compatibility

**Model Architecture:**
- Shared feature extractor (MobileNetV2)
- Feature comparison layers
- Binary classification (compatible vs incompatible)

**Expected Results:**
- Accuracy: ~70-80%
- AUC: ~0.75-0.85

## ⚙️ Training Configuration

### Custom Training Parameters

```bash
python train.py \
    --data_dir data/raw \
    --processed_dir data/processed \
    --models_dir models/saved_models \
    --img_size 224 \
    --batch_size 32 \
    --model_architecture efficientnet \
    --epochs_phase1 10 \
    --epochs_phase2 10 \
    --fine_tune \
    --compatibility_epochs 20 \
    --num_positive_pairs 3000 \
    --num_negative_pairs 3000
```

### Model Architecture Options

- `efficientnet`: EfficientNetB0 (Best accuracy, slower)
- `mobilenet`: MobileNetV2 (Good accuracy, faster)
- `resnet`: ResNet50 (Classic, reliable)

### Batch Size Recommendations

- **GPU 4GB**: batch_size=16
- **GPU 8GB**: batch_size=32
- **GPU 16GB+**: batch_size=64

## 📊 Model Evaluation

After training, evaluate the models:

```python
import tensorflow as tf
from models.clothing_classifier import ClothingClassificationModel

# Load model
model = ClothingClassificationModel(num_classes=5, img_size=(224, 224))
model.load_model('models/saved_models/clothing_classifier.keras')

# Evaluate on test set
test_dataset = ...  # Create test dataset
metrics = model.evaluate(test_dataset)

print(f"Test Accuracy: {metrics['accuracy']:.2%}")
print(f"Top-3 Accuracy: {metrics['top_3_accuracy']:.2%}")
```

## 🎯 Using Trained Models

### Clothing Classification

```python
from models.clothing_classifier import ClothingClassificationModel
import json

# Load model
model = ClothingClassificationModel(num_classes=5, img_size=(224, 224))
model.load_model('models/saved_models/clothing_classifier.keras')

# Load label mapping
with open('data/processed/label_mapping.json', 'r') as f:
    label_mapping = json.load(f)
    idx_to_label = {v: k for k, v in label_mapping.items()}

# Predict
result = model.predict('path/to/clothing.jpg', label_mapping=idx_to_label)
print(result)
# Output: {'top_predictions': [{'category': 'Topwear', 'confidence': 0.85}, ...]}
```

### Outfit Compatibility

```python
from models.outfit_compatibility_model import OutfitCompatibilityModel

# Load model
model = OutfitCompatibilityModel(feature_dim=128)
model.load_model('models/saved_models/outfit_compatibility.keras')

# Predict compatibility
score = model.predict_compatibility('shirt.jpg', 'pants.jpg')
print(f"Compatibility Score: {score:.2%}")
# Output: Compatibility Score: 87%
```

## 🐛 Troubleshooting

### Issue: Out of Memory (OOM)

**Solution:** Reduce batch size
```bash
python train.py --batch_size 16
```

### Issue: Training Too Slow

**Solutions:**
1. Use smaller model: `--model_architecture mobilenet`
2. Reduce image size: `--img_size 128`
3. Use fewer epochs: `--epochs_phase1 5 --epochs_phase2 5`

### Issue: Low Accuracy

**Solutions:**
1. Train for more epochs
2. Increase dataset size
3. Enable data augmentation (already enabled by default)
4. Try different model architecture

### Issue: Model Not Learning

**Solutions:**
1. Check data preprocessing
2. Verify labels are correct
3. Check for class imbalance
4. Reduce learning rate
5. Increase model capacity

## 📈 Monitoring Training

### TensorBoard (Optional)

```python
# Add TensorBoard callback to training
from tensorflow.keras.callbacks import TensorBoard

callback = TensorBoard(log_dir='logs', histogram_freq=1)

model.train(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    epochs=20,
    callbacks=[callback]
)
```

Then view in browser:
```bash
tensorboard --logdir logs
```

## 🔄 Retraining / Transfer Learning

### Update Model with New Data

```python
# Load existing model
model = ClothingClassificationModel(num_classes=5)
model.load_model('models/saved_models/clothing_classifier.keras')

# Continue training
model.train(
    train_dataset=new_train_dataset,
    val_dataset=new_val_dataset,
    epochs=5
)

# Save updated model
model.save_model('models/saved_models/clothing_classifier_v2.keras')
```

## 💾 Model Export

### Save for Production

```python
# Export to TensorFlow Lite (mobile)
converter = tf.lite.TFLiteConverter.from_keras_model(model.model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Save to ONNX (cross-platform)

```bash
pip install tf2onnx

python -m tf2onnx.convert \
    --keras models/saved_models/clothing_classifier.keras \
    --output clothing_classifier.onnx
```

## 📝 Training Logs

Training logs are automatically saved:
- Model checkpoints: `models/saved_models/`
- Training plots: `models/saved_models/training_history.png`
- Dataset info: `data/processed/dataset_summary.json`

## 🎓 Advanced Topics

### Custom Loss Functions

```python
def custom_loss(y_true, y_pred):
    # Weighted categorical crossentropy
    weights = tf.constant([1.0, 2.0, 1.5, 1.0, 1.0])
    return tf.keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=False
    ) * weights
```

### Learning Rate Scheduling

```python
from tensorflow.keras.callbacks import LearningRateScheduler

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

callback = LearningRateScheduler(scheduler)
```

## 🚀 Next Steps

After training:

1. **Test Models**: Run evaluation on test set
2. **Deploy**: Integrate models into the Streamlit app
3. **Monitor**: Track model performance over time
4. **Iterate**: Collect feedback and retrain

## 📚 Additional Resources

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Applications](https://keras.io/api/applications/)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Model Optimization](https://www.tensorflow.org/model_optimization)

---

For questions or issues, please open a GitHub issue or contact the maintainers.
