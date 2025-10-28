#!/usr/bin/env python3
"""
Extract Visual Features from Clothing Images
- Extracts dominant RGB colors using K-means clustering
- Detects patterns (solid, striped, plaid, floral, checkered, dotted)
- Analyzes color diversity and brightness
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from sklearn.cluster import KMeans
from tqdm import tqdm
import json
from datetime import datetime

# Paths
ROOT = Path('/home/anupam/code/AIProject')
DATA = ROOT / 'data'
PROCESSED = DATA / 'processed'
RAW = DATA / 'raw'

print("=" * 80)
print("🎨 VISUAL FEATURE EXTRACTION")
print("=" * 80)
print(f"Started: {datetime.now()}")

def extract_dominant_colors(image_path, n_colors=3):
    """Extract dominant colors using K-means clustering."""
    try:
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize for faster processing
        img_small = cv2.resize(img_rgb, (100, 100))
        
        # Reshape to list of pixels
        pixels = img_small.reshape(-1, 3)
        
        # Apply K-means
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get colors and their frequencies
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        counts = np.bincount(labels)
        
        # Sort by frequency
        sorted_indices = np.argsort(-counts)
        sorted_colors = colors[sorted_indices]
        sorted_percentages = counts[sorted_indices] / counts.sum()
        
        return {
            'color1_rgb': tuple(sorted_colors[0]),
            'color1_pct': float(sorted_percentages[0]),
            'color2_rgb': tuple(sorted_colors[1]) if len(sorted_colors) > 1 else (0, 0, 0),
            'color2_pct': float(sorted_percentages[1]) if len(sorted_colors) > 1 else 0.0,
            'color3_rgb': tuple(sorted_colors[2]) if len(sorted_colors) > 2 else (0, 0, 0),
            'color3_pct': float(sorted_percentages[2]) if len(sorted_colors) > 2 else 0.0,
        }
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def detect_pattern(image_path):
    """Detect pattern type using edge detection and frequency analysis."""
    try:
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            return 'unknown'
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize
        gray = cv2.resize(gray, (224, 224))
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (224 * 224)
        
        # Analyze color variance
        std_dev = np.std(gray)
        
        # Calculate horizontal and vertical gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        horizontal_edges = np.sum(np.abs(sobelx))
        vertical_edges = np.sum(np.abs(sobely))
        
        # Pattern detection logic
        if edge_density < 0.05 and std_dev < 30:
            return 'solid'
        
        # Check for stripes (strong directional edges)
        edge_ratio = abs(horizontal_edges - vertical_edges) / (horizontal_edges + vertical_edges + 1e-6)
        
        if edge_ratio > 0.3 and edge_density > 0.1:
            if horizontal_edges > vertical_edges:
                return 'striped_horizontal'
            else:
                return 'striped_vertical'
        
        # Check for plaid/checkered (both horizontal and vertical)
        if edge_ratio < 0.2 and edge_density > 0.15:
            return 'checkered'
        
        # Check for dots/polka
        # Use blob detection
        detector = cv2.SimpleBlobDetector_create()
        keypoints = detector.detect(gray)
        
        if len(keypoints) > 10:
            return 'dotted'
        
        # High texture variance suggests floral or complex pattern
        if std_dev > 50 and edge_density > 0.2:
            return 'floral'
        
        # Medium complexity
        if edge_density > 0.1:
            return 'textured'
        
        return 'solid'
        
    except Exception as e:
        print(f"Error detecting pattern for {image_path}: {e}")
        return 'unknown'

def calculate_color_metrics(color_data):
    """Calculate additional color metrics."""
    try:
        c1 = np.array(color_data['color1_rgb'])
        c2 = np.array(color_data['color2_rgb'])
        c3 = np.array(color_data['color3_rgb'])
        
        # Average brightness
        brightness = float(np.mean([np.mean(c1), np.mean(c2), np.mean(c3)]))
        
        # Color diversity (how different are the colors)
        color_diversity = float(np.linalg.norm(c1 - c2) + np.linalg.norm(c2 - c3) + np.linalg.norm(c1 - c3)) / 3
        
        # Determine if warm or cool
        # Warm colors have more red, cool colors have more blue
        avg_red = float((c1[0] + c2[0] + c3[0]) / 3)
        avg_blue = float((c1[2] + c2[2] + c3[2]) / 3)
        
        temperature = 'warm' if avg_red > avg_blue else 'cool'
        
        # Saturation (distance from gray)
        def saturation(color):
            mean_val = np.mean(color)
            return np.std(color - mean_val)
        
        avg_saturation = float((saturation(c1) + saturation(c2) + saturation(c3)) / 3)
        
        return {
            'brightness': brightness,
            'color_diversity': color_diversity,
            'temperature': temperature,
            'saturation': avg_saturation
        }
    except:
        return {
            'brightness': 128.0,
            'color_diversity': 0.0,
            'temperature': 'neutral',
            'saturation': 0.0
        }

# Load existing data
print("\n📂 Loading datasets...")
train_df = pd.read_csv(PROCESSED / 'train.csv')
val_df = pd.read_csv(PROCESSED / 'val.csv')
test_df = pd.read_csv(PROCESSED / 'test.csv')

print(f"✅ Train: {len(train_df):,}")
print(f"✅ Val: {len(val_df):,}")
print(f"✅ Test: {len(test_df):,}")

def process_dataset(df, name):
    """Process a dataset and add visual features."""
    print(f"\n🎨 Processing {name} dataset...")
    
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting features"):
        image_path = RAW / 'images' / f"{row['id']}.jpg"
        
        # Extract colors
        color_data = extract_dominant_colors(image_path)
        
        if color_data is None:
            # Use defaults if image not found
            color_data = {
                'color1_rgb': (128, 128, 128),
                'color1_pct': 1.0,
                'color2_rgb': (0, 0, 0),
                'color2_pct': 0.0,
                'color3_rgb': (0, 0, 0),
                'color3_pct': 0.0,
            }
        
        # Detect pattern
        pattern = detect_pattern(image_path)
        
        # Calculate metrics
        metrics = calculate_color_metrics(color_data)
        
        # Combine all features
        result = {
            'id': row['id'],
            **color_data,
            'pattern': pattern,
            **metrics
        }
        
        results.append(result)
    
    # Create features dataframe
    features_df = pd.DataFrame(results)
    
    # Merge with original
    enhanced_df = df.merge(features_df, on='id', how='left')
    
    return enhanced_df

# Process all datasets
train_enhanced = process_dataset(train_df, 'TRAIN')
val_enhanced = process_dataset(val_df, 'VAL')
test_enhanced = process_dataset(test_df, 'TEST')

# Save enhanced datasets
print("\n💾 Saving enhanced datasets...")
train_enhanced.to_csv(PROCESSED / 'train_enhanced.csv', index=False)
val_enhanced.to_csv(PROCESSED / 'val_enhanced.csv', index=False)
test_enhanced.to_csv(PROCESSED / 'test_enhanced.csv', index=False)

print(f"✅ Saved: {PROCESSED / 'train_enhanced.csv'}")
print(f"✅ Saved: {PROCESSED / 'val_enhanced.csv'}")
print(f"✅ Saved: {PROCESSED / 'test_enhanced.csv'}")

# Print sample
print("\n📊 Sample enhanced data:")
print(train_enhanced[['id', 'articleType', 'baseColour', 'color1_rgb', 'pattern', 'temperature']].head(10))

# Statistics
print("\n📈 Pattern distribution:")
print(train_enhanced['pattern'].value_counts())

print("\n📈 Temperature distribution:")
print(train_enhanced['temperature'].value_counts())

print(f"\n📈 Average brightness: {train_enhanced['brightness'].mean():.2f}")
print(f"📈 Average color diversity: {train_enhanced['color_diversity'].mean():.2f}")
print(f"📈 Average saturation: {train_enhanced['saturation'].mean():.2f}")

print("\n" + "=" * 80)
print("🎉 FEATURE EXTRACTION COMPLETE!")
print("=" * 80)
print(f"Finished: {datetime.now()}")
print("\nNew columns added:")
print("  - color1_rgb, color2_rgb, color3_rgb (actual RGB values)")
print("  - color1_pct, color2_pct, color3_pct (percentages)")
print("  - pattern (solid, striped, checkered, floral, etc.)")
print("  - brightness, color_diversity, temperature, saturation")
