#!/usr/bin/env python3
"""Simple Clothing Classifier Training Script"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import json

# Paths
ROOT = Path('/home/anupam/code/AIProject')
DATA = ROOT / 'data'
PROCESSED = DATA / 'processed'
RAW = DATA / 'raw'
MODELS = ROOT / 'models' / 'saved_models'
MODELS.mkdir(parents=True, exist_ok=True)

# Config
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS1 = 15
EPOCHS2 = 15

print("=" * 80)
print("🚀 CLOTHING CLASSIFIER TRAINING")
print("=" * 80)
print(f"Started: {datetime.now()}")
print(f"Data: {PROCESSED}")
print(f"Models: {MODELS}")

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
print(f"\nGPU: {'Yes' if gpus else 'No (using CPU)'}")

# Load data
print("\n📂 Loading data...")
train_df = pd.read_csv(PROCESSED / 'train.csv')
val_df = pd.read_csv(PROCESSED / 'val.csv')
test_df = pd.read_csv(PROCESSED / 'test.csv')

with open(PROCESSED / 'label_mapping.json') as f:
    labels = json.load(f)

num_classes = len(labels)
print(f"✅ Train: {len(train_df):,}")
print(f"✅ Val: {len(val_df):,}")
print(f"✅ Test: {len(test_df):,}")
print(f"✅ Classes: {num_classes}")

# Create dataset
def make_dataset(df, training=True):
    def load_img(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32) / 255.0
        if training:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.2)
        return img, label
    
    paths = [str(RAW / 'images' / f"{r['id']}.jpg") for _, r in df.iterrows()]
    labels_list = df['category_label'].values
    
    ds = tf.data.Dataset.from_tensor_slices((paths, labels_list))
    ds = ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(1000)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

print("\n📊 Creating datasets...")
train_ds = make_dataset(train_df, True)
val_ds = make_dataset(val_df, False)
test_ds = make_dataset(test_df, False)

# Build model
print("\n🏗️  Building model...")
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

base = MobileNetV2(weights='imagenet', include_top=False, 
                   input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])

print(f"✅ Model: {model.count_params():,} params")

# Phase 1
print("\n" + "=" * 80)
print("PHASE 1: Frozen base")
print("=" * 80)

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

cb1 = [
    tf.keras.callbacks.ModelCheckpoint(
        str(MODELS / 'classifier_p1.keras'),
        save_best_only=True, monitor='val_accuracy', mode='max'
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
]

print("\n🚀 Training Phase 1...")
h1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS1, callbacks=cb1)

# Phase 2
print("\n" + "=" * 80)
print("PHASE 2: Fine-tuning")
print("=" * 80)

base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.00001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

cb2 = [
    tf.keras.callbacks.ModelCheckpoint(
        str(MODELS / 'classifier_final.keras'),
        save_best_only=True, monitor='val_accuracy', mode='max'
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
]

print("\n🚀 Training Phase 2...")
h2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS2, callbacks=cb2)

# Evaluate
print("\n" + "=" * 80)
print("EVALUATION")
print("=" * 80)

results = model.evaluate(test_ds, return_dict=True)
print(f"\nTest Loss: {results['loss']:.4f}")
print(f"Test Accuracy: {results['accuracy']*100:.2f}%")

# Save
final_path = MODELS / 'clothing_classifier.keras'
model.save(final_path)
print(f"\n💾 Saved: {final_path}")

# Save history
history = {
    'phase1': {k: [float(v) for v in h1.history[k]] for k in h1.history},
    'phase2': {k: [float(v) for v in h2.history[k]] for k in h2.history},
    'test': {k: float(v) for k, v in results.items()}
}
with open(MODELS / 'history.json', 'w') as f:
    json.dump(history, f, indent=2)

print("\n" + "=" * 80)
print("🎉 TRAINING COMPLETE!")
print("=" * 80)
print(f"Finished: {datetime.now()}")
