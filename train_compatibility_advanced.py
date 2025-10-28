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
    """Calculate color harmony score (0-1) based on HSV - STRICT VERSION."""
    h1, s1, v1 = rgb_to_hsv(rgb1)
    h2, s2, v2 = rgb_to_hsv(rgb2)
    
    # Hue difference
    hue_diff = min(abs(h1 - h2), 360 - abs(h1 - h2))
    
    # Monochromatic (same hue, different values) - BEST
    if hue_diff <= 15 and abs(v1 - v2) > 0.15:
        return 1.0
    
    # Analogous colors (close on wheel): 15-45 degrees
    if 15 < hue_diff <= 45:
        return 0.95
    
    # Complementary colors (opposite on wheel): 165-195 degrees
    if 165 <= hue_diff <= 195:
        return 0.9
    
    # Triadic: 115-125 degrees
    if 115 <= hue_diff <= 125:
        return 0.85
    
    # Neutral pairing (one has low saturation < 0.15)
    if s1 < 0.15 or s2 < 0.15:
        return 0.9  # Neutrals go with everything
    
    # Both neutrals (black/white/gray)
    if s1 < 0.15 and s2 < 0.15:
        return 0.95
    
    # Moderate harmony (45-90 degrees)
    if 45 < hue_diff <= 90:
        return 0.65
    
    # Poor harmony (clashing colors)
    if 90 < hue_diff < 165:
        return 0.3  # HARSH PENALTY
    
    # Very poor
    return 0.2

def pattern_compatibility(pattern1, pattern2):
    """Check if patterns clash - STRICT VERSION."""
    # Solid goes with everything
    if pattern1 == 'solid' or pattern2 == 'solid':
        return 1.0
    
    # Textured is neutral (not too busy)
    if pattern1 == 'textured' or pattern2 == 'textured':
        return 0.9
    
    # Same patterns are okay
    if pattern1 == pattern2:
        return 0.85
    
    # MAJOR CLASHING PATTERNS - HARSH PENALTY
    major_clash_pairs = [
        ('checkered', 'striped_horizontal'),
        ('checkered', 'striped_vertical'),
        ('striped_horizontal', 'striped_vertical'),
        ('floral', 'checkered'),
        ('floral', 'striped_horizontal'),
        ('floral', 'striped_vertical'),
        ('dotted', 'checkered'),
    ]
    
    for p1, p2 in major_clash_pairs:
        if (pattern1 == p1 and pattern2 == p2) or (pattern1 == p2 and pattern2 == p1):
            return 0.05  # VERY BAD - almost impossible to make work
    
    # Minor clashes
    minor_clash_pairs = [
        ('dotted', 'striped_horizontal'),
        ('dotted', 'striped_vertical'),
        ('floral', 'dotted'),
    ]
    
    for p1, p2 in minor_clash_pairs:
        if (pattern1 == p1 and pattern2 == p2) or (pattern1 == p2 and pattern2 == p1):
            return 0.3  # BAD but not terrible
    
    # Two different busy patterns (not solid/textured)
    return 0.5

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
    def occasion_compatible(usage1, usage2, usage3=None):
        """Check if items have compatible usage/occasion - STRICT VERSION."""
        usages = [u for u in [usage1, usage2, usage3] if not pd.isna(u)]
        if not usages:
            return 0.5  # Unknown usage, assume low-medium compatibility
        
        usages = [str(u).strip() for u in usages]
        
        # All same usage - PERFECT
        if len(set(usages)) == 1:
            return 1.0
        
        # Compatible combinations with higher standards
        compatible_sets = [
            {'Casual', 'Travel'},
            {'Casual', 'Home'},
            {'Sports', 'Casual'},
            {'Sports', 'Travel'},
            {'Formal', 'Smart Casual'},
            {'Smart Casual', 'Party'},
        ]
        
        usage_set = set(usages)
        for comp_set in compatible_sets:
            if usage_set.issubset(comp_set):
                return 0.85  # Good compatibility
        
        # INCOMPATIBLE combinations (clash) - HARSH PENALTY
        incompatible_sets = [
            {'Formal', 'Sports'},
            {'Formal', 'Home'},
            {'Formal', 'Casual'},
            {'Party', 'Sports'},
            {'Ethnic', 'Sports'},
            {'Ethnic', 'Casual'},
        ]
        
        for incomp_set in incompatible_sets:
            if len(usage_set & incomp_set) == len(usage_set):  # All items in incompatible set
                return 0.1  # VERY BAD
        
        return 0.5  # Default moderate compatibility
    
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
                top_shoes_pattern = pattern_compatibility(top['pattern'], shoes['pattern'])
                bottom_shoes_pattern = pattern_compatibility(bottom['pattern'], shoes['pattern'])
                
                # OCCASION/USAGE consistency - STRICT!
                occasion_score = occasion_compatible(
                    top.get('usage', ''), 
                    bottom.get('usage', ''),
                    shoes.get('usage', '')
                )
                
                # Overall outfit score with STRICTER weights
                outfit_score = (
                    top_bottom_color * 0.20 +
                    top_shoes_color * 0.05 +
                    bottom_shoes_color * 0.05 +
                    top_bottom_pattern * 0.30 +     # Patterns are crucial!
                    top_shoes_pattern * 0.05 +
                    bottom_shoes_pattern * 0.05 +
                    occasion_score * 0.30            # Occasion match very important!
                )
                
                # Accept if GOOD outfit (raised threshold)
                if outfit_score > 0.70:  # STRICTER: was 0.55, now 0.70
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
                top_shoes_pattern = pattern_compatibility(top['pattern'], shoes['pattern'])
                bottom_shoes_pattern = pattern_compatibility(bottom['pattern'], shoes['pattern'])
                occasion_score = occasion_compatible(
                    top.get('usage', ''), 
                    bottom.get('usage', ''),
                    shoes.get('usage', '')
                )
                
                outfit_score = (
                    top_bottom_color * 0.20 +
                    top_shoes_color * 0.05 +
                    bottom_shoes_color * 0.05 +
                    top_bottom_pattern * 0.30 +
                    top_shoes_pattern * 0.05 +
                    bottom_shoes_pattern * 0.05 +
                    occasion_score * 0.30
                )
                
                if outfit_score > 0.70:  # STRICTER
                    outfits.append((top['id'], bottom['id'], shoes['id']))
                    labels.append(1)
                    
            elif outfit_type == 'dress' and len(female_dresses) > 0 and len(female_footwear) > 0:
                dress = female_dresses.sample(1).iloc[0]
                shoes = female_footwear.sample(1).iloc[0]
                
                dress_shoes_color = color_harmony_score(dress['color1_rgb'], shoes['color1_rgb'])
                dress_shoes_pattern = pattern_compatibility(dress['pattern'], shoes['pattern'])
                occasion_score = occasion_compatible(
                    dress.get('usage', ''), 
                    shoes.get('usage', '')
                )
                
                # Dresses scoring
                dress_score = (
                    dress_shoes_color * 0.35 + 
                    dress_shoes_pattern * 0.30 +
                    occasion_score * 0.35
                )
                
                if dress_score > 0.70:  # STRICTER
                    outfits.append((dress['id'], dress['id'], shoes['id']))
                    labels.append(1)
    
    print(f"   ✅ Created {len([l for l in labels if l == 1])} positive outfits")
    
    # NEGATIVE OUTFITS - badly coordinated looks (EXPLICIT bad examples)
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
                occasion_score = occasion_compatible(
                    top.get('usage', ''), 
                    bottom.get('usage', ''),
                    shoes.get('usage', '')
                )
                
                # PATTERN CLASH - major red flag!
                has_major_pattern_clash = (
                    top['pattern'] in ['checkered', 'striped_horizontal', 'striped_vertical', 'floral'] and
                    bottom['pattern'] in ['checkered', 'striped_horizontal', 'striped_vertical', 'floral'] and
                    top['pattern'] != bottom['pattern'] and
                    top['pattern'] not in ['solid', 'textured'] and 
                    bottom['pattern'] not in ['solid', 'textured']
                )
                
                # COLOR CLASH - clashing hues
                has_color_clash = top_bottom_color < 0.35
                
                # OCCASION CLASH - formal + sports, etc.
                has_occasion_clash = occasion_score < 0.25
                
                # Overall bad outfit score
                bad_outfit_score = (
                    top_bottom_color * 0.20 +
                    top_bottom_pattern * 0.50 +
                    occasion_score * 0.30
                )
                
                # Accept as negative if it has explicit problems
                if has_major_pattern_clash or has_color_clash or has_occasion_clash or bad_outfit_score < 0.45:
                    outfits.append((top['id'], bottom['id'], shoes['id']))
                    labels.append(0)
        else:
            outfit_type = np.random.choice(['regular', 'dress'], p=[0.6, 0.4])
            
            if outfit_type == 'regular' and len(female_topwear) > 0 and len(female_bottomwear) > 0 and len(female_footwear) > 0:
                top = female_topwear.sample(1).iloc[0]
                bottom = female_bottomwear.sample(1).iloc[0]
                shoes = female_footwear.sample(1).iloc[0]
                
                top_bottom_color = color_harmony_score(top['color1_rgb'], bottom['color1_rgb'])
                top_bottom_pattern = pattern_compatibility(top['pattern'], bottom['pattern'])
                occasion_score = occasion_compatible(
                    top.get('usage', ''), 
                    bottom.get('usage', ''),
                    shoes.get('usage', '')
                )
                
                has_major_pattern_clash = (
                    top['pattern'] in ['checkered', 'striped_horizontal', 'striped_vertical', 'floral'] and
                    bottom['pattern'] in ['checkered', 'striped_horizontal', 'striped_vertical', 'floral'] and
                    top['pattern'] != bottom['pattern'] and
                    top['pattern'] not in ['solid', 'textured'] and 
                    bottom['pattern'] not in ['solid', 'textured']
                )
                
                has_color_clash = top_bottom_color < 0.35
                has_occasion_clash = occasion_score < 0.25
                
                bad_outfit_score = (
                    top_bottom_color * 0.20 +
                    top_bottom_pattern * 0.50 +
                    occasion_score * 0.30
                )
                
                if has_major_pattern_clash or has_color_clash or has_occasion_clash or bad_outfit_score < 0.45:
                    outfits.append((top['id'], bottom['id'], shoes['id']))
                    labels.append(0)
                    
            elif outfit_type == 'dress' and len(female_dresses) > 0 and len(female_footwear) > 0:
                dress = female_dresses.sample(1).iloc[0]
                shoes = female_footwear.sample(1).iloc[0]
                
                color_score = color_harmony_score(dress['color1_rgb'], shoes['color1_rgb'])
                pattern_score = pattern_compatibility(dress['pattern'], shoes['pattern'])
                occasion_score = occasion_compatible(
                    dress.get('usage', ''), 
                    shoes.get('usage', '')
                )
                
                bad_dress_score = color_score * 0.4 + pattern_score * 0.3 + occasion_score * 0.3
                
                if color_score < 0.35 or occasion_score < 0.25 or bad_dress_score < 0.45:
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

# Shared feature extractor (MobileNetV2 base)
base_model = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3),
    pooling='avg'
)
base_model.trainable = False  # Freeze initially

# Build IMPROVED feature extractor with deeper understanding
feature_input = layers.Input(shape=(224, 224, 3))
x = base_model(feature_input)
x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
x = layers.BatchNormalization()(x)
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

# IMPROVED FEATURE FUSION - Multiple interaction types
# 1. Direct concatenation (preserve individual features)
concat_features = layers.Concatenate()([features_top, features_bottom, features_shoes])

# 2. Pairwise differences (capture contrast)
diff_top_bottom = layers.Subtract()([features_top, features_bottom])
diff_top_shoes = layers.Subtract()([features_top, features_shoes])
diff_bottom_shoes = layers.Subtract()([features_bottom, features_shoes])
diff_features = layers.Concatenate()([
    layers.Lambda(lambda x: tf.abs(x))(diff_top_bottom),
    layers.Lambda(lambda x: tf.abs(x))(diff_top_shoes),
    layers.Lambda(lambda x: tf.abs(x))(diff_bottom_shoes)
])

# 3. Pairwise products (capture co-occurrence)
prod_top_bottom = layers.Multiply()([features_top, features_bottom])
prod_top_shoes = layers.Multiply()([features_top, features_shoes])
prod_bottom_shoes = layers.Multiply()([features_bottom, features_shoes])
prod_features = layers.Concatenate()([prod_top_bottom, prod_top_shoes, prod_bottom_shoes])

# 4. Attention-like mechanism (learn what matters most)
avg_features = layers.Average()([features_top, features_bottom, features_shoes])
max_features = layers.Maximum()([features_top, features_bottom, features_shoes])

# Combine ALL interaction types
combined = layers.Concatenate()([
    concat_features,    # Direct features (192)
    diff_features,      # Differences (192)
    prod_features,      # Products (192)
    avg_features,       # Average (64)
    max_features        # Maximum (64)
])  # Total: 704 features

# DEEPER compatibility scoring network
x = layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(combined)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)

x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)

x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.2)(x)

# Output: compatibility score [0, 1]
compatibility = layers.Dense(1, activation='sigmoid', name='compatibility')(x)

# Create model
model = keras.Model(
    inputs=[input_top, input_bottom, input_shoes],
    outputs=compatibility,
    name='outfit_compatibility_3item_advanced'
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc'), keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall')]
)

print(f"✅ Model built: {model.count_params():,} parameters")

# Callbacks with improved training strategy
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
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=10,  # More patience for better convergence
        restore_best_weights=True,
        verbose=1
    ),
    # Learning rate warmup
    keras.callbacks.LearningRateScheduler(
        lambda epoch: 0.0001 * min(1.0, (epoch + 1) / 5),  # Warmup first 5 epochs
        verbose=0
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
    epochs=50,  # More epochs with early stopping
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

test_results = model.evaluate(
    test_dataset,
    steps=test_steps,
    verbose=1
)

# Unpack all metrics
test_loss = test_results[0]
test_acc = test_results[1]
test_auc = test_results[2]
test_precision = test_results[3]
test_recall = test_results[4]

print(f"\n✅ Test Results:")
print(f"   Loss: {test_loss:.4f}")
print(f"   Accuracy: {test_acc:.4f}")
print(f"   AUC: {test_auc:.4f}")
print(f"   Precision: {test_precision:.4f}")
print(f"   Recall: {test_recall:.4f}")
print(f"   F1-Score: {2 * (test_precision * test_recall) / (test_precision + test_recall):.4f}")

# Save history with additional metrics
history_dict = {
    'train_loss': [float(x) for x in history.history['loss']],
    'train_accuracy': [float(x) for x in history.history['accuracy']],
    'train_auc': [float(x) for x in history.history['auc']],
    'train_precision': [float(x) for x in history.history['precision']],
    'train_recall': [float(x) for x in history.history['recall']],
    'val_loss': [float(x) for x in history.history['val_loss']],
    'val_accuracy': [float(x) for x in history.history['val_accuracy']],
    'val_auc': [float(x) for x in history.history['val_auc']],
    'val_precision': [float(x) for x in history.history['val_precision']],
    'val_recall': [float(x) for x in history.history['val_recall']],
    'test_loss': float(test_loss),
    'test_accuracy': float(test_acc),
    'test_auc': float(test_auc),
    'test_precision': float(test_precision),
    'test_recall': float(test_recall)
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
print("  • Pattern detection to avoid clashes (striped, checkered, floral, etc.)")
print("  • Occasion matching (Casual, Formal, Sports, Party, Ethnic)")
print("  • Gender-specific outfit rules (Men's vs Women's)")
print("  • Deep CNN features from actual images")
print("  • Strict compatibility scoring (70% threshold for good outfits)")
print("\n⚠️  NOTE: Skin tone compatibility is NOT yet integrated.")
print("   Skin tone analysis exists in database but not in this training.")
print("   Future enhancement: Match outfit colors to user's skin undertone.")
