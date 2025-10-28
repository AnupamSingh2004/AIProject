"""
Outfit Compatibility Model
Learns to score outfit combinations based on color harmony and style.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import List, Tuple, Dict
import random


class OutfitCompatibilityModel:
    """
    Model that learns outfit compatibility from color and style features.
    Uses a siamese-style network to learn outfit scoring.
    """
    
    def __init__(self, feature_dim: int = 128):
        """
        Initialize the compatibility model.
        
        Args:
            feature_dim: Dimension of feature embeddings
        """
        self.feature_dim = feature_dim
        self.model = None
        
    def build_feature_extractor(self, img_size: Tuple[int, int] = (224, 224)) -> keras.Model:
        """
        Build a CNN feature extractor for clothing items.
        
        Args:
            img_size: Input image size
            
        Returns:
            Feature extractor model
        """
        inputs = layers.Input(shape=(*img_size, 3))
        
        # Use MobileNetV2 as backbone
        base_model = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=(*img_size, 3),
            pooling=None
        )
        base_model.trainable = False
        
        # Apply preprocessing and base model
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Feature embedding
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        features = layers.Dense(self.feature_dim, activation=None, name='features')(x)
        
        # L2 normalize features using Keras layer
        features = layers.Lambda(lambda x: keras.ops.normalize(x, axis=1))(features)
        
        model = keras.Model(inputs, features, name='feature_extractor')
        
        return model
    
    def build_compatibility_model(self, img_size: Tuple[int, int] = (224, 224)) -> keras.Model:
        """
        Build the outfit compatibility model.
        Takes two clothing items and predicts compatibility score.
        
        Args:
            img_size: Input image size
            
        Returns:
            Compatibility model
        """
        print("🏗️  Building outfit compatibility model...")
        
        # Feature extractor (shared weights)
        feature_extractor = self.build_feature_extractor(img_size)
        
        # Inputs for two clothing items
        input_item1 = layers.Input(shape=(*img_size, 3), name='item1')
        input_item2 = layers.Input(shape=(*img_size, 3), name='item2')
        
        # Extract features
        features1 = feature_extractor(input_item1)
        features2 = feature_extractor(input_item2)
        
        # Combine features
        # Option 1: Concatenate
        combined = layers.Concatenate()([features1, features2])
        
        # Option 2: Element-wise operations (can also use)
        difference = layers.Subtract()([features1, features2])
        product = layers.Multiply()([features1, features2])
        
        # Combine all representations
        x = layers.Concatenate()([combined, difference, product])
        
        # Compatibility scoring layers
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        
        # Output: compatibility score [0, 1]
        compatibility_score = layers.Dense(1, activation='sigmoid', name='compatibility')(x)
        
        # Create model
        model = keras.Model(
            inputs=[input_item1, input_item2],
            outputs=compatibility_score,
            name='outfit_compatibility'
        )
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        self.model = model
        self.feature_extractor = feature_extractor
        
        print(f"✅ Model built successfully!")
        print(f"Total parameters: {model.count_params():,}")
        
        return model
    
    def create_outfit_pairs_dataset(self, 
                                   df: pd.DataFrame,
                                   num_positive: int = 5000,
                                   num_negative: int = 5000) -> pd.DataFrame:
        """
        Create training dataset of outfit pairs with compatibility labels.
        
        Args:
            df: DataFrame with clothing items
            num_positive: Number of positive (compatible) pairs
            num_negative: Number of negative (incompatible) pairs
            
        Returns:
            DataFrame with outfit pairs
        """
        print("\n📊 Creating outfit pairs dataset...")
        
        pairs = []
        
        # Create positive pairs (same style or complementary colors)
        print("Creating positive pairs...")
        for _ in range(num_positive):
            # Strategy: Pick items from same style or occasion
            if 'main_category' in df.columns:
                # Pick top and bottom from compatible styles
                tops = df[df['main_category'].isin(['Topwear', 'Dress'])]
                bottoms = df[df['main_category'] == 'Bottomwear']
                
                if len(tops) > 0 and len(bottoms) > 0:
                    top = tops.sample(1).iloc[0]
                    bottom = bottoms.sample(1).iloc[0]
                    
                    pairs.append({
                        'item1_path': top['image_path'],
                        'item2_path': bottom['image_path'],
                        'item1_category': top.get('main_category', 'Unknown'),
                        'item2_category': bottom.get('main_category', 'Unknown'),
                        'compatible': 1
                    })
        
        # Create negative pairs (clashing colors or incompatible items)
        print("Creating negative pairs...")
        for _ in range(num_negative):
            # Strategy: Pick random incompatible items
            item1, item2 = df.sample(2).values
            
            pairs.append({
                'item1_path': item1[df.columns.get_loc('image_path')],
                'item2_path': item2[df.columns.get_loc('image_path')],
                'item1_category': item1[df.columns.get_loc('main_category')] if 'main_category' in df.columns else 'Unknown',
                'item2_category': item2[df.columns.get_loc('main_category')] if 'main_category' in df.columns else 'Unknown',
                'compatible': 0
            })
        
        pairs_df = pd.DataFrame(pairs)
        
        print(f"✅ Created {len(pairs_df)} outfit pairs")
        print(f"   Positive pairs: {sum(pairs_df['compatible'] == 1)}")
        print(f"   Negative pairs: {sum(pairs_df['compatible'] == 0)}")
        
        return pairs_df
    
    def create_tf_dataset(self, 
                         pairs_df: pd.DataFrame,
                         img_size: Tuple[int, int] = (224, 224),
                         batch_size: int = 32) -> tf.data.Dataset:
        """
        Create TensorFlow dataset from pairs DataFrame.
        
        Args:
            pairs_df: DataFrame with outfit pairs
            img_size: Image size
            batch_size: Batch size
            
        Returns:
            TensorFlow dataset
        """
        def parse_pair(item1_path, item2_path, label):
            """Load and preprocess image pair."""
            # Load images
            img1 = tf.io.read_file(item1_path)
            img1 = tf.image.decode_jpeg(img1, channels=3)
            img1 = tf.image.resize(img1, img_size)
            img1 = img1 / 255.0
            
            img2 = tf.io.read_file(item2_path)
            img2 = tf.image.decode_jpeg(img2, channels=3)
            img2 = tf.image.resize(img2, img_size)
            img2 = img2 / 255.0
            
            return (img1, img2), label
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((
            pairs_df['item1_path'].values,
            pairs_df['item2_path'].values,
            pairs_df['compatible'].values.astype(np.float32)
        ))
        
        dataset = dataset.map(parse_pair, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def train(self,
             train_dataset,
             val_dataset,
             epochs: int = 20,
             callbacks: list = None) -> keras.callbacks.History:
        """
        Train the compatibility model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of epochs
            callbacks: List of callbacks
            
        Returns:
            Training history
        """
        print(f"\n🚀 Starting training for {epochs} epochs...")
        
        if callbacks is None:
            callbacks = []
        
        callbacks.extend([
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ])
        
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Training complete!")
        
        return history
    
    def predict_compatibility(self, 
                             item1_path: str,
                             item2_path: str,
                             img_size: Tuple[int, int] = (224, 224)) -> float:
        """
        Predict compatibility score for two clothing items.
        
        Args:
            item1_path: Path to first item image
            item2_path: Path to second item image
            img_size: Image size
            
        Returns:
            Compatibility score [0, 1]
        """
        # Load and preprocess images
        img1 = tf.io.read_file(item1_path)
        img1 = tf.image.decode_jpeg(img1, channels=3)
        img1 = tf.image.resize(img1, img_size)
        img1 = tf.expand_dims(img1 / 255.0, 0)
        
        img2 = tf.io.read_file(item2_path)
        img2 = tf.image.decode_jpeg(img2, channels=3)
        img2 = tf.image.resize(img2, img_size)
        img2 = tf.expand_dims(img2 / 255.0, 0)
        
        # Predict
        score = self.model.predict([img1, img2], verbose=0)[0][0]
        
        return float(score)
    
    def save_model(self, save_path: str):
        """Save the model."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save(str(save_path))
        print(f"✅ Model saved to {save_path}")
    
    def load_model(self, model_path: str):
        """Load a saved model."""
        self.model = keras.models.load_model(model_path)
        print(f"✅ Model loaded from {model_path}")


def build_compatibility_model(img_size: Tuple[int, int] = (224, 224)) -> keras.Model:
    """Helper function to build and return the compatibility model."""
    model_instance = OutfitCompatibilityModel(feature_dim=128)
    return model_instance.build_compatibility_model(img_size)

def main():
    """Main training script for outfit compatibility model."""
    print("=" * 70)
    print("👔 Outfit Compatibility Model Training")
    print("=" * 70)
    
    # Paths
    processed_dir = Path("../data/processed")
    models_dir = Path("../models/saved_models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    train_df = pd.read_csv(processed_dir / 'train.csv')
    val_df = pd.read_csv(processed_dir / 'val.csv')
    
    # Create model
    model = OutfitCompatibilityModel(feature_dim=128)
    model.build_compatibility_model(img_size=(224, 224))
    model.model.summary()
    
    # Create outfit pairs datasets
    train_pairs = model.create_outfit_pairs_dataset(train_df, num_positive=3000, num_negative=3000)
    val_pairs = model.create_outfit_pairs_dataset(val_df, num_positive=500, num_negative=500)
    
    # Save pairs for reference
    train_pairs.to_csv(processed_dir / 'train_outfit_pairs.csv', index=False)
    val_pairs.to_csv(processed_dir / 'val_outfit_pairs.csv', index=False)
    
    # Create TensorFlow datasets
    train_dataset = model.create_tf_dataset(train_pairs, batch_size=32)
    val_dataset = model.create_tf_dataset(val_pairs, batch_size=32)
    
    # Train model
    history = model.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=20
    )
    
    # Save model
    model.save_model(str(models_dir / 'outfit_compatibility.keras'))
    
    print("\n" + "=" * 70)
    print("✅ Training Pipeline Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
