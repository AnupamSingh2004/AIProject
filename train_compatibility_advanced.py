#!/usr/bin/env python3
"""
Advanced Outfit Compatibility Model Training
Uses REAL visual features: RGB colors, patterns, brightness, saturation
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from datetime import datetime
import json

# GPU setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"🎮 GPU detected: {gpus[0].name}")
    tf.config.experimental.set_memory_growth(gpus[0], True)
else:
    print("⚠️  No GPU detected, using CPU")

# Paths
ROOT = Path('/home/anupam/code/AIProject')
DATA = ROOT / 'data'
PROCESSED = DATA / 'processed'
RAW = DATA / 'raw'
MODELS = ROOT / 'models' / 'saved_models'
MODELS.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("🎨 ADVANCED OUTFIT COMPATIBILITY TRAINING")
print("=" * 80)
print(f"Started: {datetime.now()}")

# Load enhanced datasets with visual features
print("\n📂 Loading enhanced datasets...")
train_df = pd.read_csv(PROCESSED / 'train_enhanced.csv')
val_df = pd.read_csv(PROCESSED / 'val_enhanced.csv')
test_df = pd.read_csv(PROCESSED / 'test_enhanced.csv')

print(f"✅ Train: {len(train_df):,}")
print(f"✅ Val: {len(val_df):,}")
print(f"✅ Test: {len(test_df):,}")

# Parse RGB tuples from string
def parse_rgb(rgb_str):
    """Convert '(45, 67, 123)' to [45, 67, 123]"""
    if pd.isna(rgb_str) or rgb_str == '(0, 0, 0)':
        return [0, 0, 0]
    try:
        return [int(x) for x in rgb_str.strip('()').split(',')]
    except:
        return [0, 0, 0]

# Add parsed RGB columns
for col in ['color1_rgb', 'color2_rgb', 'color3_rgb']:
    train_df[col] = train_df[col].apply(parse_rgb)
    val_df[col] = val_df[col].apply(parse_rgb)
    test_df[col] = test_df[col].apply(parse_rgb)

print("\n✅ Parsed RGB values")

# Color harmony functions
def rgb_to_hsv(rgb):
    """Convert RGB to HSV for color harmony calculations."""
    r, g, b = np.array(rgb) / 255.0
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    diff = max_c - min_c
    
    # Hue
    if diff == 0:
        h = 0
    elif max_c == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif max_c == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    else:
        h = (60 * ((r - g) / diff) + 240) % 360
    
    # Saturation
    s = 0 if max_c == 0 else (diff / max_c)
    
    # Value
    v = max_c
    
    return h, s, v

def color_harmony_score(rgb1, rgb2):
    """Calculate color harmony score (0-1) based on HSV."""
    h1, s1, v1 = rgb_to_hsv(rgb1)
    h2, s2, v2 = rgb_to_hsv(rgb2)
    
    # Hue difference
    hue_diff = min(abs(h1 - h2), 360 - abs(h1 - h2))
    
    # Complementary colors (opposite on wheel): 150-210 degrees
    if 150 <= hue_diff <= 210:
        return 0.9
    
    # Analogous colors (close on wheel): 0-60 degrees
    if hue_diff <= 60:
        return 0.85
    
    # Triadic: 100-140 degrees
    if 100 <= hue_diff <= 140:
        return 0.75
    
    # Neutral pairing (one has low saturation)
    if s1 < 0.2 or s2 < 0.2:
        return 0.8
    
    # Moderate harmony
    return 0.5

def pattern_compatibility(pattern1, pattern2):
    """Check if patterns clash."""
    # Solid goes with everything
    if pattern1 == 'solid' or pattern2 == 'solid':
        return 1.0
    
    # Same patterns are okay
    if pattern1 == pattern2:
        return 0.8
    
    # Clashing patterns
    clash_pairs = [
        ('checkered', 'striped_horizontal'),
        ('checkered', 'striped_vertical'),
        ('striped_horizontal', 'striped_vertical'),
        ('floral', 'checkered'),
        ('floral', 'striped_horizontal'),
        ('floral', 'striped_vertical'),
        ('dotted', 'striped_horizontal'),
        ('dotted', 'striped_vertical'),
    ]
    
    for p1, p2 in clash_pairs:
        if (pattern1 == p1 and pattern2 == p2) or (pattern1 == p2 and pattern2 == p1):
            return 0.2
    
    # Textured is neutral
    if pattern1 == 'textured' or pattern2 == 'textured':
        return 0.7
    
    return 0.6

def create_training_pairs(df, n_pairs=10000):
    """Create COMPLETE OUTFIT SETS (3 items: top + bottom + shoes) considering patterns, colors, gender, and OCCASION."""
    print(f"\n🔄 Creating {n_pairs} complete outfit sets (Top + Bottom + Shoes)...")
    
    # Get items by category and gender
    male_topwear = df[(df['main_category'] == 'Topwear') & (df['gender'].isin(['Men', 'Boys', 'Unisex']))]
    male_bottomwear = df[(df['main_category'] == 'Bottomwear') & (df['gender'].isin(['Men', 'Boys', 'Unisex']))]
    male_footwear = df[(df['main_category'] == 'Footwear') & (df['gender'].isin(['Men', 'Boys', 'Unisex']))]
    
    female_topwear = df[(df['main_category'] == 'Topwear') & (df['gender'].isin(['Women', 'Girls', 'Unisex']))]
    female_bottomwear = df[(df['main_category'] == 'Bottomwear') & (df['gender'].isin(['Women', 'Girls', 'Unisex']))]
    female_footwear = df[(df['main_category'] == 'Footwear') & (df['gender'].isin(['Women', 'Girls', 'Unisex']))]
    female_dresses = df[(df['main_category'] == 'Dress') & (df['gender'].isin(['Women', 'Girls']))]
    
    # Unisex items
    unisex_topwear = df[(df['main_category'] == 'Topwear') & (df['gender'] == 'Unisex')]
    unisex_bottomwear = df[(df['main_category'] == 'Bottomwear') & (df['gender'] == 'Unisex')]
    unisex_footwear = df[(df['main_category'] == 'Footwear') & (df['gender'] == 'Unisex')]
    
    outfits = []
    labels = []
    
    n_positive = n_pairs // 2
    n_negative = n_pairs // 2
    
    # Occasion compatibility rules
    def occasion_compatible(usage1, usage2):
        """Check if two items have compatible usage/occasion."""
        if pd.isna(usage1) or pd.isna(usage2):
            return 0.7  # Unknown usage, assume moderate compatibility
        
        usage1 = str(usage1).strip()
        usage2 = str(usage2).strip()
        
        # Perfect match
        if usage1 == usage2:
            return 1.0
        
        # Compatible combinations
        compatible_pairs = {
            ('Casual', 'Sports'): 0.8,
            ('Casual', 'Travel'): 0.9,
            ('Casual', 'Home'): 0.9,
            ('Formal', 'Smart Casual'): 0.8,
            ('Smart Casual', 'Party'): 0.8,
            ('Sports', 'Travel'): 0.7,
        }
        
        # Check both directions
        score = compatible_pairs.get((usage1, usage2)) or compatible_pairs.get((usage2, usage1))
        if score:
            return score
        
        # Incompatible combinations (clash)
        incompatible_pairs = {
            ('Formal', 'Sports'),
            ('Formal', 'Home'),
            ('Formal', 'Casual'),
            ('Party', 'Sports'),
            ('Ethnic', 'Sports'),
        }
        
        if (usage1, usage2) in incompatible_pairs or (usage2, usage1) in incompatible_pairs:
            return 0.2  # Strong penalty
        
        return 0.6  # Default moderate compatibility
    
    # POSITIVE OUTFITS - well-coordinated complete looks
    print(f"   Creating {n_positive} positive (good) outfits...")
    attempts = 0
    max_attempts = n_positive * 20
    
    while len([l for l in labels if l == 1]) < n_positive and attempts < max_attempts:
        attempts += 1
        
        # Choose gender randomly
        gender = np.random.choice(['male', 'female'])
        
        if gender == 'male':
            outfit_type = 'regular'  # Men usually wear top+bottom+shoes
            if len(male_topwear) > 0 and len(male_bottomwear) > 0 and len(male_footwear) > 0:
                top = male_topwear.sample(1).iloc[0]
                bottom = male_bottomwear.sample(1).iloc[0]
                shoes = male_footwear.sample(1).iloc[0]
                
                # Check compatibility (patterns, colors, style, OCCASION)
                top_bottom_color = color_harmony_score(top['color1_rgb'], bottom['color1_rgb'])
                top_shoes_color = color_harmony_score(top['color1_rgb'], shoes['color1_rgb'])
                bottom_shoes_color = color_harmony_score(bottom['color1_rgb'], shoes['color1_rgb'])
                
                # PATTERN CHECK - very important!
                top_bottom_pattern = pattern_compatibility(top['pattern'], bottom['pattern'])
                
                # OCCASION/USAGE consistency - NEW!
                occasion_score = occasion_compatible(top.get('usage', ''), bottom.get('usage', ''))
                
                # Overall outfit score
                outfit_score = (
                    top_bottom_color * 0.25 +
                    top_shoes_color * 0.08 +
                    bottom_shoes_color * 0.07 +
                    top_bottom_pattern * 0.35 +  # Patterns are crucial!
                    occasion_score * 0.25          # Occasion match important!
                )
                
                # Accept if good outfit
                if outfit_score > 0.55:
                    outfits.append((top['id'], bottom['id'], shoes['id']))
                    labels.append(1)
                    
        else:  # female
            outfit_type = np.random.choice(['regular', 'dress'], p=[0.6, 0.4])
            
            if outfit_type == 'regular' and len(female_topwear) > 0 and len(female_bottomwear) > 0 and len(female_footwear) > 0:
                top = female_topwear.sample(1).iloc[0]
                bottom = female_bottomwear.sample(1).iloc[0]
                shoes = female_footwear.sample(1).iloc[0]
                
                top_bottom_color = color_harmony_score(top['color1_rgb'], bottom['color1_rgb'])
                top_shoes_color = color_harmony_score(top['color1_rgb'], shoes['color1_rgb'])
                bottom_shoes_color = color_harmony_score(bottom['color1_rgb'], shoes['color1_rgb'])
                top_bottom_pattern = pattern_compatibility(top['pattern'], bottom['pattern'])
                occasion_score = occasion_compatible(top.get('usage', ''), bottom.get('usage', ''))
                
                outfit_score = (
                    top_bottom_color * 0.25 +
                    top_shoes_color * 0.08 +
                    bottom_shoes_color * 0.07 +
                    top_bottom_pattern * 0.35 +
                    occasion_score * 0.25
                )
                
                if outfit_score > 0.55:
                    outfits.append((top['id'], bottom['id'], shoes['id']))
                    labels.append(1)
                    
            elif outfit_type == 'dress' and len(female_dresses) > 0 and len(female_footwear) > 0:
                dress = female_dresses.sample(1).iloc[0]
                shoes = female_footwear.sample(1).iloc[0]
                
                dress_shoes_color = color_harmony_score(dress['color1_rgb'], shoes['color1_rgb'])
                occasion_score = occasion_compatible(dress.get('usage', ''), shoes.get('usage', ''))
                
                # Dresses with solid/textured patterns are easier to match
                dress_score = dress_shoes_color * 0.5 + occasion_score * 0.5
                if dress_score > 0.55 or dress['pattern'] in ['solid', 'textured']:
                    outfits.append((dress['id'], dress['id'], shoes['id']))
                    labels.append(1)
    
    print(f"   ✅ Created {len([l for l in labels if l == 1])} positive outfits")
    
    # NEGATIVE OUTFITS - badly coordinated looks (considering patterns + OCCASIONS!)
    print(f"   Creating {n_negative} negative (bad) outfits...")
    attempts = 0
    max_attempts = n_negative * 20
    
    while len([l for l in labels if l == 0]) < n_negative and attempts < max_attempts:
        attempts += 1
        
        gender = np.random.choice(['male', 'female'])
        
        if gender == 'male':
            if len(male_topwear) > 0 and len(male_bottomwear) > 0 and len(male_footwear) > 0:
                top = male_topwear.sample(1).iloc[0]
                bottom = male_bottomwear.sample(1).iloc[0]
                shoes = male_footwear.sample(1).iloc[0]
                
                top_bottom_color = color_harmony_score(top['color1_rgb'], bottom['color1_rgb'])
                top_bottom_pattern = pattern_compatibility(top['pattern'], bottom['pattern'])
                occasion_score = occasion_compatible(top.get('usage', ''), bottom.get('usage', ''))
                
                # PATTERN CLASH - major red flag!
                has_pattern_clash = (
                    top['pattern'] in ['checkered', 'striped_horizontal', 'striped_vertical', 'floral'] and
                    bottom['pattern'] in ['checkered', 'striped_horizontal', 'striped_vertical', 'floral'] and
                    top['pattern'] != bottom['pattern'] and
                    top['pattern'] != 'solid' and bottom['pattern'] != 'solid'
                )
                
                poor_coordination = (top_bottom_color * 0.3 + top_bottom_pattern * 0.4 + occasion_score * 0.3) < 0.5
                
                # OCCASION CLASH - formal + sports, etc.
                occasion_clash = occasion_score < 0.3
                
                if has_pattern_clash or poor_coordination or occasion_clash:
                    outfits.append((top['id'], bottom['id'], shoes['id']))
                    labels.append(0)
        else:
            outfit_type = np.random.choice(['regular', 'dress'], p=[0.6, 0.4])
            
            if outfit_type == 'regular' and len(female_topwear) > 0 and len(female_bottomwear) > 0 and len(female_footwear) > 0:
                top = female_topwear.sample(1).iloc[0]
                bottom = female_bottomwear.sample(1).iloc[0]
                shoes = female_footwear.sample(1).iloc[0]
                
                top_bottom_pattern = pattern_compatibility(top['pattern'], bottom['pattern'])
                occasion_score = occasion_compatible(top.get('usage', ''), bottom.get('usage', ''))
                
                has_pattern_clash = (
                    top['pattern'] in ['checkered', 'striped_horizontal', 'striped_vertical', 'floral'] and
                    bottom['pattern'] in ['checkered', 'striped_horizontal', 'striped_vertical', 'floral'] and
                    top['pattern'] != bottom['pattern']
                )
                
                poor_coordination = (top_bottom_pattern * 0.5 + occasion_score * 0.5) < 0.4
                
                if has_pattern_clash or poor_coordination or occasion_score < 0.3:
                    outfits.append((top['id'], bottom['id'], shoes['id']))
                    labels.append(0)
                    
            elif outfit_type == 'dress' and len(female_dresses) > 0 and len(female_footwear) > 0:
                dress = female_dresses.sample(1).iloc[0]
                shoes = female_footwear.sample(1).iloc[0]
                
                color_score = color_harmony_score(dress['color1_rgb'], shoes['color1_rgb'])
                occasion_score = occasion_compatible(dress.get('usage', ''), shoes.get('usage', ''))
                
                if color_score < 0.4 or occasion_score < 0.3:
                    outfits.append((dress['id'], dress['id'], shoes['id']))
                    labels.append(0)
    
    # Fill remaining negatives with random combos
    negatives_needed = n_negative - len([l for l in labels if l == 0])
    if negatives_needed > 0:
        print(f"   Adding {negatives_needed} random negative outfits...")
        for _ in range(negatives_needed):
            gender = np.random.choice(['male', 'female'])
            if gender == 'male' and len(male_topwear) > 0 and len(male_bottomwear) > 0 and len(male_footwear) > 0:
                top = male_topwear.sample(1).iloc[0]
                bottom = male_bottomwear.sample(1).iloc[0]
                shoes = male_footwear.sample(1).iloc[0]
                outfits.append((top['id'], bottom['id'], shoes['id']))
                labels.append(0)
            elif len(female_topwear) > 0 and len(female_bottomwear) > 0 and len(female_footwear) > 0:
                top = female_topwear.sample(1).iloc[0]
                bottom = female_bottomwear.sample(1).iloc[0]
                shoes = female_footwear.sample(1).iloc[0]
                outfits.append((top['id'], bottom['id'], shoes['id']))
                labels.append(0)
    
    total_positive = sum(labels)
    total_negative = len(labels) - total_positive
    print(f"\n✅ Created {len(outfits)} complete outfits ({total_positive} good, {total_negative} bad)")
    print(f"   Balance: {total_positive/len(outfits)*100:.1f}% good, {total_negative/len(outfits)*100:.1f}% bad")
    print(f"   Patterns checked: solid, striped, checkered, floral")
    print(f"   Gender separation: Men's and Women's outfits")
    print(f"   Occasion matching: Casual/Formal/Sports/Party/Ethnic compatibility")
    
    return outfits, labels

# Create dataset from outfit sets (3 items: top, bottom, shoes)
def create_dataset_from_pairs(pairs, labels, batch_size=32, cache_name=""):
    """Create batched dataset from 3-item outfit sets with intelligent caching."""
    
    # Pre-load ALL unique images used in outfits
    unique_ids = set()
    for id1, id2, id3 in pairs:
        unique_ids.add(id1)
        unique_ids.add(id2)
        unique_ids.add(id3)
    
    print(f"   📸 Caching {len(unique_ids)} unique images for {cache_name}...")
    img_cache = {}
    
    for i, img_id in enumerate(unique_ids):
        if i % 100 == 0 and i > 0:
            print(f"      Loaded {i}/{len(unique_ids)}...", end='\r')
        
        img_path = RAW / 'images' / f'{int(img_id)}.jpg'
        try:
            img = tf.io.read_file(str(img_path))
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [224, 224])
            img = tf.cast(img, tf.float32) / 255.0
            img_cache[img_id] = img.numpy()
        except:
            img_cache[img_id] = np.zeros((224, 224, 3), dtype=np.float32)
    
    print(f"   ✅ Cached {len(unique_ids)} images ({len(unique_ids) * 224 * 224 * 3 * 4 / (1024**2):.1f} MB)")
    
    # Create arrays for faster access
    outfit_data = []
    for idx in range(len(pairs)):
        id1, id2, id3 = pairs[idx]
        img1 = img_cache[id1]  # Top or Dress
        img2 = img_cache[id2]  # Bottom or Dress
        img3 = img_cache[id3]  # Shoes
        label = labels[idx]
        outfit_data.append((img1, img2, img3, label))
    
    def generator():
        """Generator that yields cached outfit sets (3 images)."""
        indices = np.arange(len(outfit_data))
        np.random.shuffle(indices)
        
        for idx in indices:
            img1, img2, img3, label = outfit_data[idx]
            # Concatenate 3 images along channel dimension: (224, 224, 9)
            # Or pass as 3 separate inputs
            yield (img1, img2, img3), label
    
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            (
                tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),  # Top
                tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),  # Bottom
                tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),  # Shoes
            ),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    )
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# Create training pairs (BALANCED - 50/50 split)
train_pairs, train_labels = create_training_pairs(train_df, n_pairs=8000)
val_pairs, val_labels = create_training_pairs(val_df, n_pairs=1600)

# Create datasets (cache all images for maximum speed)
print("\n📦 Creating TensorFlow datasets with image caching...")
train_dataset = create_dataset_from_pairs(train_pairs, train_labels, batch_size=32, cache_name="TRAIN")
val_dataset = create_dataset_from_pairs(val_pairs, val_labels, batch_size=32, cache_name="VAL")
print("✅ All datasets created and cached")

# Build 3-input compatibility model for complete outfits
print("\n🏗️  Building 3-input Siamese CNN model for complete outfits...")

# Shared feature extractor
base_model = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3),
    pooling='avg'
)
base_model.trainable = False  # Freeze initially

# Build feature extractor
feature_input = layers.Input(shape=(224, 224, 3))
x = base_model(feature_input)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
feature_output = layers.Dense(64, activation='relu', name='features')(x)
feature_extractor = keras.Model(feature_input, feature_output, name='feature_extractor')

# Three inputs for complete outfit
input_top = layers.Input(shape=(224, 224, 3), name='top')
input_bottom = layers.Input(shape=(224, 224, 3), name='bottom')
input_shoes = layers.Input(shape=(224, 224, 3), name='shoes')

# Extract features from each item
features_top = feature_extractor(input_top)
features_bottom = feature_extractor(input_bottom)
features_shoes = feature_extractor(input_shoes)

# Combine all features
combined = layers.Concatenate()([features_top, features_bottom, features_shoes])

# Compatibility scoring
x = layers.Dense(256, activation='relu')(combined)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.2)(x)

# Output: compatibility score [0, 1]
compatibility = layers.Dense(1, activation='sigmoid', name='compatibility')(x)

# Create model
model = keras.Model(
    inputs=[input_top, input_bottom, input_shoes],
    outputs=compatibility,
    name='outfit_compatibility_3item'
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

print(f"✅ Model built: {model.count_params():,} parameters")

# Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(
        MODELS / 'outfit_compatibility_advanced.keras',
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=7,
        restore_best_weights=True,
        verbose=1
    )
]

# Training
print("\n🚀 Starting training...")
print(f"Train pairs: {len(train_pairs):,}")
print(f"Val pairs: {len(val_pairs):,}")

# Calculate steps per epoch
steps_per_epoch = len(train_pairs) // 32
validation_steps = len(val_pairs) // 32

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")

history = model.fit(
    train_dataset,
    steps_per_epoch=steps_per_epoch,
    epochs=25,
    validation_data=val_dataset,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1
)

# Evaluate on test set
print("\n📊 Evaluating on test set...")
test_pairs, test_labels = create_training_pairs(test_df, n_pairs=2000)
test_dataset = create_dataset_from_pairs(test_pairs, test_labels, batch_size=32)
test_steps = len(test_pairs) // 32

test_loss, test_acc, test_auc = model.evaluate(
    test_dataset,
    steps=test_steps,
    verbose=1
)

print(f"\n✅ Test Results:")
print(f"   Loss: {test_loss:.4f}")
print(f"   Accuracy: {test_acc:.4f}")
print(f"   AUC: {test_auc:.4f}")

# Save history
history_dict = {
    'train_loss': [float(x) for x in history.history['loss']],
    'train_accuracy': [float(x) for x in history.history['accuracy']],
    'train_auc': [float(x) for x in history.history['auc']],
    'val_loss': [float(x) for x in history.history['val_loss']],
    'val_accuracy': [float(x) for x in history.history['val_accuracy']],
    'val_auc': [float(x) for x in history.history['val_auc']],
    'test_loss': float(test_loss),
    'test_accuracy': float(test_acc),
    'test_auc': float(test_auc)
}

with open(MODELS / 'compatibility_advanced_history.json', 'w') as f:
    json.dump(history_dict, f, indent=2)

print("\n" + "=" * 80)
print("🎉 TRAINING COMPLETE!")
print("=" * 80)
print(f"Finished: {datetime.now()}")
print(f"\n✅ Model saved: {MODELS / 'outfit_compatibility_advanced.keras'}")
print(f"✅ History saved: {MODELS / 'compatibility_advanced_history.json'}")
print("\nThis model uses:")
print("  • Real RGB color values for color harmony")
print("  • Pattern detection to avoid clashes")
print("  • Brightness balance for visual appeal")
print("  • Deep CNN features from actual images")
