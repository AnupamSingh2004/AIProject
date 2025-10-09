#!/usr/bin/env python3
"""
Simple Training Script - Trains the clothing classifier model.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'models'))
sys.path.insert(0, str(project_root / 'scripts'))

print("=" * 80)
print("🎨 SIMPLE CLOTHING CLASSIFIER TRAINING")
print("=" * 80)

# Import after path is set
from models.clothing_classifier import ClothingClassificationModel
import pandas as pd
import json

# Paths
processed_dir = project_root / 'data' / 'processed'
models_dir = project_root / 'models' / 'saved_models'
models_dir.mkdir(parents=True, exist_ok=True)

# Load label mapping
print("\n📂 Loading label mapping...")
with open(processed_dir / 'label_mapping.json', 'r') as f:
    label_info = json.load(f)

num_classes = len(label_info['label_to_name'])
print(f"Number of classes: {num_classes}")
print(f"Classes: {list(label_info['label_to_name'].values())}")

# Create model
print("\n🏗️  Creating model...")
model = ClothingClassificationModel(
    num_classes=num_classes,
    img_size=(224, 224),
    model_name='efficientnet',
    use_pretrained=True
)

# Build model
model.build_model()

# Create data generators
print("\n📊 Creating data generators...")
train_csv = str(processed_dir / 'train.csv')
val_csv = str(processed_dir / 'val.csv')

train_dataset, val_dataset = model.create_data_generators(
    train_csv=train_csv,
    val_csv=val_csv,
    batch_size=16
)

# Train Phase 1 (frozen base)
print("\n" + "=" * 80)
print("PHASE 1: Training with frozen base model")
print("=" * 80)

history1 = model.train(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    epochs=3,
    model_save_path=str(models_dir / 'clothing_classifier_phase1.keras')
)

print("\n✅ Phase 1 complete!")

# Train Phase 2 (fine-tuning)
print("\n" * 80)
print("PHASE 2: Fine-tuning model")
print("=" * 80)

model.unfreeze_base_model(layers_to_unfreeze=-30)

history2 = model.train(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    epochs=3,
    model_save_path=str(models_dir / 'clothing_classifier.keras')
)

print("\n✅ Phase 2 complete!")

# Final evaluation
print("\n" + "=" * 80)
print("📊 FINAL EVALUATION")
print("=" * 80)

test_csv = str(processed_dir / 'test.csv')
test_df = pd.read_csv(test_csv)

print(f"\nEvaluating on {len(test_df)} test samples...")

# Create test dataset
import tensorflow as tf

def parse_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0
    return image, label

test_paths = test_df['image_path'].values
test_labels = test_df['category_label'].values

test_dataset = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
test_dataset = test_dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(16)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

results = model.evaluate(test_dataset)

print("\n" + "=" * 80)
print("🎉 TRAINING COMPLETE!")
print("=" * 80)
print(f"\nModel saved to: {models_dir / 'clothing_classifier.keras'}")
print("\nNext steps:")
print("  1. Run the compatibility model training")
print("  2. Test with: streamlit run app/streamlit_app.py")
print("=" * 80)
