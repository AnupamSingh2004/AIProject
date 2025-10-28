#!/usr/bin/env python3
"""Train Outfit Compatibility Model - Learns which clothing items look good together"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import json
import random

# Paths
ROOT = Path('/home/anupam/code/AIProject')
DATA = ROOT / 'data'
PROCESSED = DATA / 'processed'
RAW = DATA / 'raw'
MODELS = ROOT / 'models' / 'saved_models'
MODELS.mkdir(parents=True, exist_ok=True)

# Config
IMG_SIZE = 224
BATCH_SIZE = 16  # Smaller batch for pair training
EPOCHS = 20
FEATURE_DIM = 128

print("=" * 80)
print("👔 OUTFIT COMPATIBILITY MODEL TRAINING")
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

print(f"✅ Train: {len(train_df):,} items")
print(f"✅ Val: {len(val_df):,} items")

# Create outfit pairs
def create_outfit_pairs(df, num_pairs=3000):
    """Create positive and negative outfit pairs."""
    print(f"\n🔗 Creating {num_pairs} outfit pairs...")
    
    pairs = []
    labels = []
    
    # Get items by category
    topwear = df[df['main_category'] == 'Topwear']['id'].values
    bottomwear = df[df['main_category'] == 'Bottomwear']['id'].values
    footwear = df[df['main_category'] == 'Footwear']['id'].values
    
    print(f"   Topwear: {len(topwear)}")
    print(f"   Bottomwear: {len(bottomwear)}")
    print(f"   Footwear: {len(footwear)}")
    
    # Positive pairs (complementary items)
    num_positive = num_pairs // 2
    for _ in range(num_positive):
        if len(topwear) > 0 and len(bottomwear) > 0:
            top = random.choice(topwear)
            bottom = random.choice(bottomwear)
            pairs.append((top, bottom))
            labels.append(1)
    
    # Negative pairs (same category or random incompatible)
    num_negative = num_pairs - len(pairs)
    for _ in range(num_negative):
        # Same category (e.g., two tops - not an outfit)
        if random.random() < 0.5 and len(topwear) > 1:
            item1, item2 = random.sample(list(topwear), 2)
        elif len(bottomwear) > 1:
            item1, item2 = random.sample(list(bottomwear), 2)
        else:
            item1 = random.choice(topwear)
            item2 = random.choice(topwear)
        
        pairs.append((item1, item2))
        labels.append(0)
    
    print(f"✅ Created {len(pairs)} pairs (positive: {sum(labels)}, negative: {len(labels) - sum(labels)})")
    return pairs, labels

train_pairs, train_labels = create_outfit_pairs(train_df, num_pairs=6000)
val_pairs, val_labels = create_outfit_pairs(val_df, num_pairs=1000)

# Create TensorFlow dataset
def create_pair_dataset(pairs, labels, training=True):
    """Create dataset of image pairs."""
    def load_pair(path1, path2, label):
        # Load images
        img1 = tf.io.read_file(path1)
        img1 = tf.image.decode_jpeg(img1, channels=3)
        img1 = tf.image.resize(img1, [IMG_SIZE, IMG_SIZE])
        img1 = tf.cast(img1, tf.float32) / 255.0
        
        img2 = tf.io.read_file(path2)
        img2 = tf.image.decode_jpeg(img2, channels=3)
        img2 = tf.image.resize(img2, [IMG_SIZE, IMG_SIZE])
        img2 = tf.cast(img2, tf.float32) / 255.0
        
        if training:
            img1 = tf.image.random_flip_left_right(img1)
            img2 = tf.image.random_flip_left_right(img2)
        
        return (img1, img2), label
    
    # Create full paths (as strings, not tensors)
    paths1 = [str(RAW / 'images' / f'{p[0]}.jpg') for p in pairs]
    paths2 = [str(RAW / 'images' / f'{p[1]}.jpg') for p in pairs]
    
    ds = tf.data.Dataset.from_tensor_slices((paths1, paths2, labels))
    ds = ds.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(1000)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return ds

print("\n📊 Creating datasets...")
train_ds = create_pair_dataset(train_pairs, train_labels, training=True)
val_ds = create_pair_dataset(val_pairs, val_labels, training=False)

# Build compatibility model
print("\n🏗️  Building compatibility model...")
from tensorflow.keras import layers, models

# Feature extractor (shared for both items)
def build_feature_extractor():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Use MobileNetV2
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling=None
    )
    base.trainable = False
    
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    features = layers.Dense(FEATURE_DIM, activation=None)(x)
    
    # L2 normalize
    features = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(features)
    
    return models.Model(inputs, features, name='feature_extractor')

# Build the full model
feature_extractor = build_feature_extractor()

input1 = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='item1')
input2 = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='item2')

# Extract features
feat1 = feature_extractor(input1)
feat2 = feature_extractor(input2)

# Combine features
concat = layers.Concatenate()([feat1, feat2])
diff = layers.Subtract()([feat1, feat2])
prod = layers.Multiply()([feat1, feat2])

combined = layers.Concatenate()([concat, diff, prod])

# Compatibility scoring
x = layers.Dense(256, activation='relu')(combined)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation='relu')(x)

output = layers.Dense(1, activation='sigmoid', name='compatibility')(x)

model = models.Model(inputs=[input1, input2], outputs=output, name='outfit_compatibility')

print(f"✅ Model: {model.count_params():,} params")

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# Callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        str(MODELS / 'outfit_compatibility_best.keras'),
        save_best_only=True,
        monitor='val_auc',
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

# Train
print("\n" + "=" * 80)
print("🚀 TRAINING")
print("=" * 80)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Save final model
final_path = MODELS / 'outfit_compatibility.keras'
model.save(final_path)
print(f"\n💾 Saved: {final_path}")

# Save history
history_dict = {k: [float(v) for v in history.history[k]] for k in history.history}
with open(MODELS / 'compatibility_history.json', 'w') as f:
    json.dump(history_dict, f, indent=2)

# Test predictions
print("\n" + "=" * 80)
print("🧪 SAMPLE PREDICTIONS")
print("=" * 80)

# Get some test pairs
test_pairs = train_pairs[:5]
test_labels = train_labels[:5]

for i, (pair, label) in enumerate(zip(test_pairs, test_labels)):
    path1 = str(RAW / 'images' / f'{pair[0]}.jpg')
    path2 = str(RAW / 'images' / f'{pair[1]}.jpg')
    
    img1 = tf.io.read_file(path1)
    img1 = tf.image.decode_jpeg(img1, channels=3)
    img1 = tf.image.resize(img1, [IMG_SIZE, IMG_SIZE])
    img1 = tf.expand_dims(img1 / 255.0, 0)
    
    img2 = tf.io.read_file(path2)
    img2 = tf.image.decode_jpeg(img2, channels=3)
    img2 = tf.image.resize(img2, [IMG_SIZE, IMG_SIZE])
    img2 = tf.expand_dims(img2 / 255.0, 0)
    
    pred = model.predict([img1, img2], verbose=0)[0][0]
    
    print(f"\nPair {i+1}: {pair[0]} + {pair[1]}")
    print(f"  True label: {'Compatible' if label == 1 else 'Incompatible'}")
    print(f"  Predicted: {pred:.3f} ({'Compatible' if pred > 0.5 else 'Incompatible'})")

print("\n" + "=" * 80)
print("🎉 COMPATIBILITY MODEL TRAINING COMPLETE!")
print("=" * 80)
print(f"Finished: {datetime.now()}")
