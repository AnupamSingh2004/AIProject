#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import json

# Config
ROOT = Path('/home/anupam/code/AIProject')
PROC = ROOT / 'data' / 'processed'
RAW = ROOT / 'data' / 'raw'
MODELS = ROOT / 'models' / 'saved_models'
MODELS.mkdir(parents=True, exist_ok=True)

SIZE = 224
BATCH = 32
EP1 = 20
EP2 = 20

print("="*80)
print("TRAINING CLOTHING CLASSIFIER")
print("="*80)
print(f"Started: {datetime.now()}")

# Load data
train_df = pd.read_csv(PROC / 'train.csv')
val_df = pd.read_csv(PROC / 'val.csv')
test_df = pd.read_csv(PROC / 'test.csv')

with open(PROC / 'label_mapping.json') as f:
    labels = json.load(f)

num_cls = len(labels)
print(f"Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")
print(f"Classes: {num_cls}")

# Data pipeline
def make_dataset(df, train=True):
    def load(path, lbl):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, 3)
        img = tf.image.resize(img, [SIZE, SIZE])
        img = tf.cast(img, tf.float32) / 255.0
        if train:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.2)
            img = tf.image.random_contrast(img, 0.8, 1.2)
        return img, lbl
    
    paths = [str(RAW / 'images' / f"{r['id']}.jpg") for _, r in df.iterrows()]
    lbls = df['category_label'].values
    
    ds = tf.data.Dataset.from_tensor_slices((paths, lbls))
    ds = ds.map(load, tf.data.AUTOTUNE)
    if train:
        ds = ds.shuffle(1000)
    ds = ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)
    return ds

print("Creating datasets...")
train_ds = make_dataset(train_df, True)
val_ds = make_dataset(val_df, False)
test_ds = make_dataset(test_df, False)
print("Done!")

# Build custom CNN model  
print("\nBuilding model...")
from tensorflow.keras import layers, models, applications

# Use MobileNetV2 which works better with Keras 3
base = applications.MobileNetV2(
    input_shape=(SIZE, SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_cls, activation='softmax')
])

print(f"Model parameters: {model.count_params():,}")

# Phase 1 - Frozen base
print("\n" + "="*80)
print("PHASE 1: Training with frozen base (20 epochs)")
print("="*80)

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

cb1 = [
    tf.keras.callbacks.ModelCheckpoint(
        str(MODELS/'p1_best.keras'),
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

print("\nStarting training...\n")
h1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EP1,
    callbacks=cb1,
    verbose=1
)

print("\n✓ Phase 1 complete!")

# Phase 2 - Fine-tuning
print("\n" + "="*80)
print("PHASE 2: Fine-tuning (20 epochs)")
print("="*80)

base.trainable = True
# Freeze first 100 layers, fine-tune rest
for layer in base.layers[:100]:
    layer.trainable = False

print(f"Unfroze {sum(1 for l in base.layers if l.trainable)} layers")

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.00001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

cb2 = [
    tf.keras.callbacks.ModelCheckpoint(
        str(MODELS/'p2_best.keras'),
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-8,
        verbose=1
    )
]

print("\nStarting fine-tuning...\n")
h2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EP2,
    callbacks=cb2,
    verbose=1
)

print("\n✓ Phase 2 complete!")

# Final evaluation
print("\n" + "="*80)
print("FINAL EVALUATION ON TEST SET")
print("="*80)

results = model.evaluate(test_ds, return_dict=True, verbose=1)
print(f"\n✓ Test Accuracy: {results['accuracy']*100:.2f}%")
print(f"✓ Test Loss: {results['loss']:.4f}")

# Save final model
final_path = MODELS / 'clothing_classifier.keras'
model.save(final_path)
print(f"\n✓ Model saved: {final_path}")

# Save training history
history = {
    'phase1': {k: [float(v) for v in h1.history[k]] for k in h1.history},
    'phase2': {k: [float(v) for v in h2.history[k]] for k in h2.history},
    'test_results': {k: float(v) for k, v in results.items()}
}

history_path = MODELS / 'training_history.json'
with open(history_path, 'w') as f:
    json.dump(history, f, indent=2)

print(f"✓ History saved: {history_path}")

print("\n" + "="*80)
print("✓ TRAINING COMPLETE!")
print("="*80)
print(f"Finished: {datetime.now()}")
print(f"\nSaved models:")
print(f"  - {final_path}")
print(f"  - {MODELS/'p1_best.keras'}")
print(f"  - {MODELS/'p2_best.keras'}")
print("\nNext: Train outfit compatibility model or test with Streamlit app")
