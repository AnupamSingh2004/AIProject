#!/usr/bin/env python3
"""
Smart Outfit Compatibility Training
Uses color theory and style matching to create meaningful training pairs
"""

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
BATCH_SIZE = 16
EPOCHS = 25
FEATURE_DIM = 128

print("=" * 80)
print("👔 SMART OUTFIT COMPATIBILITY MODEL TRAINING")
print("=" * 80)
print(f"Started: {datetime.now()}")

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU: {'Yes' if gpus else 'No (using CPU)'}")

# Load data
print("\n📂 Loading data...")
train_df = pd.read_csv(PROCESSED / 'train.csv')
val_df = pd.read_csv(PROCESSED / 'val.csv')

print(f"✅ Train: {len(train_df):,} items")
print(f"✅ Val: {len(val_df):,} items")

# Color compatibility rules (based on color theory)
COLOR_COMPATIBILITY = {
    'Black': {'compatible': ['White', 'Grey', 'Silver', 'Gold', 'Red', 'Blue', 'Pink'], 'clash': []},
    'White': {'compatible': ['Black', 'Navy Blue', 'Blue', 'Red', 'Grey', 'Beige'], 'clash': []},
    'Blue': {'compatible': ['White', 'Beige', 'Brown', 'Grey', 'Orange', 'Yellow'], 'clash': ['Green', 'Purple']},
    'Navy Blue': {'compatible': ['White', 'Beige', 'Brown', 'Grey', 'Red'], 'clash': ['Black']},
    'Grey': {'compatible': ['White', 'Black', 'Blue', 'Navy Blue', 'Pink', 'Yellow'], 'clash': []},
    'Brown': {'compatible': ['Beige', 'White', 'Blue', 'Green', 'Orange'], 'clash': ['Black', 'Grey']},
    'Beige': {'compatible': ['White', 'Brown', 'Blue', 'Navy Blue', 'Green'], 'clash': []},
    'Red': {'compatible': ['White', 'Black', 'Grey', 'Beige'], 'clash': ['Pink', 'Orange', 'Purple']},
    'Green': {'compatible': ['Brown', 'Beige', 'White', 'Khaki'], 'clash': ['Blue', 'Red']},
    'Yellow': {'compatible': ['Grey', 'Blue', 'White', 'Purple'], 'clash': ['Orange', 'Green']},
    'Orange': {'compatible': ['Blue', 'White', 'Brown', 'Beige'], 'clash': ['Red', 'Yellow', 'Pink']},
    'Pink': {'compatible': ['Grey', 'White', 'Black', 'Beige'], 'clash': ['Red', 'Orange']},
    'Purple': {'compatible': ['Yellow', 'White', 'Grey'], 'clash': ['Red', 'Orange', 'Blue']},
}

# Style compatibility
STYLE_COMPAT = {
    'Formal': ['Formal', 'Smart Casual'],
    'Casual': ['Casual', 'Smart Casual'],
    'Sports': ['Sports', 'Casual'],
    'Ethnic': ['Ethnic'],
    'Party': ['Party', 'Formal', 'Smart Casual'],
}

def are_colors_compatible(color1, color2):
    """Check if two colors are compatible."""
    if pd.isna(color1) or pd.isna(color2):
        return 0.5  # Unknown
    
    color1 = str(color1).strip()
    color2 = str(color2).strip()
    
    if color1 == color2:
        return 0.7  # Same color (monochromatic - moderate)
    
    if color1 in COLOR_COMPATIBILITY:
        if color2 in COLOR_COMPATIBILITY[color1]['compatible']:
            return 1.0  # Highly compatible
        elif color2 in COLOR_COMPATIBILITY[color1]['clash']:
            return 0.0  # Clash
    
    return 0.5  # Neutral

def are_styles_compatible(style1, style2):
    """Check if two usage styles are compatible."""
    if pd.isna(style1) or pd.isna(style2):
        return 0.5
    
    style1 = str(style1).strip()
    style2 = str(style2).strip()
    
    if style1 == style2:
        return 1.0  # Perfect match
    
    if style1 in STYLE_COMPAT:
        if style2 in STYLE_COMPAT[style1]:
            return 0.8  # Compatible
    
    return 0.3  # Likely incompatible

def create_smart_outfit_pairs(df, num_pairs=5000):
    """Create smart outfit pairs using color and style theory."""
    print(f"\n🎨 Creating {num_pairs} smart outfit pairs...")
    
    # Separate by category
    topwear = df[df['main_category'] == 'Topwear']
    bottomwear = df[df['main_category'] == 'Bottomwear']
    footwear = df[df['main_category'] == 'Footwear']
    
    print(f"   Topwear: {len(topwear)}")
    print(f"   Bottomwear: {len(bottomwear)}")
    print(f"   Footwear: {len(footwear)}")
    
    pairs = []
    labels = []
    
    # Generate positive pairs (complementary colors + matching styles)
    print("   Creating positive pairs (good outfits)...")
    num_positive = num_pairs // 2
    attempts = 0
    max_attempts = num_positive * 10
    
    while len([l for l in labels if l == 1]) < num_positive and attempts < max_attempts:
        attempts += 1
        
        if len(topwear) == 0 or len(bottomwear) == 0:
            break
        
        top = topwear.sample(1).iloc[0]
        bottom = bottomwear.sample(1).iloc[0]
        
        # Check compatibility
        color_compat = are_colors_compatible(top['baseColour'], bottom['baseColour'])
        style_compat = are_styles_compatible(top['usage'], bottom['usage'])
        
        # Accept if good compatibility
        if color_compat >= 0.7 and style_compat >= 0.7:
            pairs.append((top['id'], bottom['id']))
            labels.append(1)
    
    print(f"   ✓ Created {sum(labels)} positive pairs")
    
    # Generate negative pairs (clashing colors OR style mismatch)
    print("   Creating negative pairs (bad outfits)...")
    num_negative = num_pairs - len(labels)
    attempts = 0
    max_attempts = num_negative * 10
    
    while len([l for l in labels if l == 0]) < num_negative and attempts < max_attempts:
        attempts += 1
        
        strategy = random.choice(['color_clash', 'style_clash', 'same_category'])
        
        if strategy == 'same_category':
            # Same category (not a valid outfit)
            if random.random() < 0.5 and len(topwear) > 1:
                item1, item2 = topwear.sample(2).iloc
            elif len(bottomwear) > 1:
                item1, item2 = bottomwear.sample(2).iloc
            else:
                continue
            
            pairs.append((item1['id'], item2['id']))
            labels.append(0)
        
        else:
            # Different categories but bad combination
            if len(topwear) == 0 or len(bottomwear) == 0:
                break
            
            top = topwear.sample(1).iloc[0]
            bottom = bottomwear.sample(1).iloc[0]
            
            color_compat = are_colors_compatible(top['baseColour'], bottom['baseColour'])
            style_compat = are_styles_compatible(top['usage'], bottom['usage'])
            
            # Accept if poor compatibility
            if strategy == 'color_clash' and color_compat <= 0.3:
                pairs.append((top['id'], bottom['id']))
                labels.append(0)
            elif strategy == 'style_clash' and style_compat <= 0.4:
                pairs.append((top['id'], bottom['id']))
                labels.append(0)
    
    print(f"   ✓ Created {len([l for l in labels if l == 0])} negative pairs")
    print(f"✅ Total pairs: {len(pairs)} (positive: {sum(labels)}, negative: {len(labels) - sum(labels)})")
    
    return pairs, labels

# Create smart pairs
train_pairs, train_labels = create_smart_outfit_pairs(train_df, num_pairs=8000)
val_pairs, val_labels = create_smart_outfit_pairs(val_df, num_pairs=1200)

# Create TensorFlow dataset
def create_pair_dataset(pairs, labels, training=True):
    """Create dataset of image pairs."""
    def load_pair(path1, path2, label):
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
            img1 = tf.image.random_brightness(img1, 0.1)
            img2 = tf.image.random_brightness(img2, 0.1)
        
        return (img1, img2), label
    
    # Create paths
    paths1 = [str(RAW / 'images' / f'{p[0]}.jpg') for p in pairs]
    paths2 = [str(RAW / 'images' / f'{p[1]}.jpg') for p in pairs]
    labels_array = np.array(labels, dtype=np.float32)
    
    ds = tf.data.Dataset.from_tensor_slices((paths1, paths2, labels_array))
    ds = ds.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(2000)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return ds

print("\n📊 Creating datasets...")
train_ds = create_pair_dataset(train_pairs, train_labels, training=True)
val_ds = create_pair_dataset(val_pairs, val_labels, training=False)

# Build model (same architecture)
print("\n🏗️  Building compatibility model...")
from tensorflow.keras import layers, models

def build_feature_extractor():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling=None
    )
    base.trainable = False
    
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    features = layers.Dense(FEATURE_DIM, activation=None)(x)
    features = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(features)
    
    return models.Model(inputs, features, name='feature_extractor')

feature_extractor = build_feature_extractor()

input1 = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='item1')
input2 = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='item2')

feat1 = feature_extractor(input1)
feat2 = feature_extractor(input2)

concat = layers.Concatenate()([feat1, feat2])
diff = layers.Subtract()([feat1, feat2])
prod = layers.Multiply()([feat1, feat2])

combined = layers.Concatenate()([concat, diff, prod])

x = layers.Dense(256, activation='relu')(combined)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation='relu')(x)

output = layers.Dense(1, activation='sigmoid', name='compatibility')(x)

model = models.Model(inputs=[input1, input2], outputs=output)

print(f"✅ Model: {model.count_params():,} params")

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# Train
print("\n" + "=" * 80)
print("🚀 TRAINING (Smart Color & Style Pairs)")
print("=" * 80)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        str(MODELS / 'outfit_compatibility_smart.keras'),
        save_best_only=True, monitor='val_auc', mode='max', verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=7, restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1
    )
]

history = model.fit(
    train_ds, validation_data=val_ds,
    epochs=EPOCHS, callbacks=callbacks, verbose=1
)

# Save
model.save(MODELS / 'outfit_compatibility_final.keras')
print(f"\n💾 Saved: {MODELS / 'outfit_compatibility_final.keras'}")

with open(MODELS / 'compatibility_smart_history.json', 'w') as f:
    json.dump({k: [float(v) for v in history.history[k]] for k in history.history}, f, indent=2)

print("\n" + "=" * 80)
print("🎉 SMART COMPATIBILITY TRAINING COMPLETE!")
print("=" * 80)
print(f"Finished: {datetime.now()}")
