"""
Clothing Classification Model
CNN model for classifying clothing items into categories.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Tuple, Dict
import matplotlib.pyplot as plt


class ClothingClassificationModel:
    """CNN model for clothing classification."""
    
    def __init__(self, 
                 num_classes: int,
                 img_size: Tuple[int, int] = (224, 224),
                 model_name: str = 'efficientnet',
                 use_pretrained: bool = True):
        """
        Initialize the classification model.
        
        Args:
            num_classes: Number of clothing categories
            img_size: Input image size
            model_name: Base model architecture ('efficientnet', 'mobilenet', 'resnet')
            use_pretrained: Whether to use ImageNet pretrained weights
        """
        self.num_classes = num_classes
        self.img_size = img_size
        self.model_name = model_name
        self.use_pretrained = use_pretrained
        
        self.model = None
        self.history = None
        
    def build_model(self) -> keras.Model:
        """
        Build the CNN model with transfer learning.
        
        Returns:
            Compiled Keras model
        """
        print(f"🏗️  Building {self.model_name} model...")
        
        # Input layer
        inputs = layers.Input(shape=(*self.img_size, 3))
        
        # Data augmentation layers (applied during training only)
        augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ], name="augmentation")
        
        x = augmentation(inputs)
        
        # Preprocessing
        if self.model_name == 'efficientnet':
            preprocess = tf.keras.applications.efficientnet.preprocess_input
            base_model = EfficientNetB0(
                include_top=False,
                weights='imagenet' if self.use_pretrained else None,
                input_shape=(*self.img_size, 3)
            )
        elif self.model_name == 'mobilenet':
            preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
            base_model = MobileNetV2(
                include_top=False,
                weights='imagenet' if self.use_pretrained else None,
                input_shape=(*self.img_size, 3)
            )
        elif self.model_name == 'resnet':
            preprocess = tf.keras.applications.resnet50.preprocess_input
            base_model = ResNet50(
                include_top=False,
                weights='imagenet' if self.use_pretrained else None,
                input_shape=(*self.img_size, 3)
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        x = preprocess(x)
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Add base model
        x = base_model(x, training=False)
        
        # Global pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = keras.Model(inputs, outputs)
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        
        self.model = model
        self.base_model = base_model
        
        print(f"✅ Model built successfully!")
        print(f"Total parameters: {model.count_params():,}")
        
        return model
    
    def unfreeze_base_model(self, layers_to_unfreeze: int = -30):
        """
        Unfreeze base model layers for fine-tuning.
        
        Args:
            layers_to_unfreeze: Number of layers to unfreeze from the end (-30 = last 30 layers)
        """
        print(f"\n🔓 Unfreezing base model layers for fine-tuning...")
        
        self.base_model.trainable = True
        
        # Freeze all layers except the last few
        for layer in self.base_model.layers[:layers_to_unfreeze]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        
        trainable_params = sum([tf.size(var).numpy() for var in self.model.trainable_variables])
        print(f"Trainable parameters: {trainable_params:,}")
    
    def create_data_generators(self, 
                               train_csv: str,
                               val_csv: str,
                               batch_size: int = 32) -> Tuple:
        """
        Create TensorFlow data generators from CSV files.
        
        Args:
            train_csv: Path to training CSV
            val_csv: Path to validation CSV
            batch_size: Batch size for training
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        print("\n📦 Creating data generators...")
        
        # Load CSVs
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        
        print(f"Training samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
        
        def parse_image(image_path, label):
            """Load and preprocess image."""
            # Read image file
            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, self.img_size)
            image = image / 255.0  # Normalize to [0, 1]
            return image, label
        
        # Create training dataset
        train_paths = train_df['image_path'].values
        train_labels = train_df['category_label'].values
        
        train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
        train_dataset = train_dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.shuffle(buffer_size=1000)
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        
        # Create validation dataset
        val_paths = val_df['image_path'].values
        val_labels = val_df['category_label'].values
        
        val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
        val_dataset = val_dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(batch_size)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
        
        print("✅ Data generators created!")
        
        return train_dataset, val_dataset
    
    def train(self,
             train_dataset,
             val_dataset,
             epochs: int = 20,
             callbacks: list = None) -> keras.callbacks.History:
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of training epochs
            callbacks: List of Keras callbacks
            
        Returns:
            Training history
        """
        print(f"\n🚀 Starting training for {epochs} epochs...")
        
        if callbacks is None:
            callbacks = []
        
        # Add default callbacks
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
        
        self.history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Training complete!")
        
        return self.history
    
    def evaluate(self, test_dataset) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Dictionary of metrics
        """
        print("\n📊 Evaluating model...")
        
        results = self.model.evaluate(test_dataset, verbose=1)
        
        metrics = {
            'loss': results[0],
            'accuracy': results[1],
            'top_3_accuracy': results[2]
        }
        
        print("\nTest Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def plot_training_history(self, save_path: str = None):
        """
        Plot training history.
        
        Args:
            save_path: Path to save plot (optional)
        """
        if self.history is None:
            print("No training history available.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot accuracy
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot loss
        axes[1].plot(self.history.history['loss'], label='Train Loss')
        axes[1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Model Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def save_model(self, save_path: str):
        """
        Save the trained model.
        
        Args:
            save_path: Path to save model
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save(str(save_path))
        print(f"✅ Model saved to {save_path}")
    
    def load_model(self, model_path: str):
        """
        Load a trained model.
        
        Args:
            model_path: Path to saved model
        """
        self.model = keras.models.load_model(model_path)
        print(f"✅ Model loaded from {model_path}")
    
    def predict(self, image_path: str, label_mapping: Dict[int, str] = None) -> Dict:
        """
        Predict clothing category for an image.
        
        Args:
            image_path: Path to image file
            label_mapping: Mapping from indices to category names
            
        Returns:
            Dictionary with predictions
        """
        # Load and preprocess image
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.img_size)
        img = img / 255.0
        img = tf.expand_dims(img, 0)
        
        # Predict
        predictions = self.model.predict(img, verbose=0)
        
        # Get top predictions
        top_k = 3
        top_indices = np.argsort(predictions[0])[-top_k:][::-1]
        
        results = {
            'top_predictions': []
        }
        
        for idx in top_indices:
            category = label_mapping.get(idx, f"Category_{idx}") if label_mapping else f"Category_{idx}"
            confidence = float(predictions[0][idx])
            
            results['top_predictions'].append({
                'category': category,
                'confidence': confidence
            })
        
        return results


def main():
    """Main training script."""
    print("=" * 70)
    print("👕 Clothing Classification Model Training")
    print("=" * 70)
    
    # Paths
    processed_dir = Path("../data/processed")
    models_dir = Path("../models/saved_models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Load label mapping
    with open(processed_dir / 'label_mapping.json', 'r') as f:
        label_mapping = json.load(f)
    
    num_classes = len(label_mapping)
    print(f"\nNumber of classes: {num_classes}")
    print(f"Classes: {list(label_mapping.keys())}")
    
    # Create model
    model = ClothingClassificationModel(
        num_classes=num_classes,
        img_size=(224, 224),
        model_name='efficientnet',
        use_pretrained=True
    )
    
    # Build model
    model.build_model()
    model.model.summary()
    
    # Create data generators
    train_dataset, val_dataset = model.create_data_generators(
        train_csv=str(processed_dir / 'train.csv'),
        val_csv=str(processed_dir / 'val.csv'),
        batch_size=32
    )
    
    # Phase 1: Train with frozen base
    print("\n" + "=" * 70)
    print("Phase 1: Training with frozen base model")
    print("=" * 70)
    
    model.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=10
    )
    
    # Phase 2: Fine-tune
    print("\n" + "=" * 70)
    print("Phase 2: Fine-tuning")
    print("=" * 70)
    
    model.unfreeze_base_model(layers_to_unfreeze=-30)
    
    model.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=10
    )
    
    # Evaluate on test set
    test_df = pd.read_csv(processed_dir / 'test.csv')
    test_paths = test_df['image_path'].values
    test_labels = test_df['category_label'].values
    
    test_dataset = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
    test_dataset = test_dataset.map(
        lambda x, y: (tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(x), channels=3), (224, 224)) / 255.0, y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    
    metrics = model.evaluate(test_dataset)
    
    # Plot training history
    model.plot_training_history(save_path=str(models_dir / 'training_history.png'))
    
    # Save model
    model.save_model(str(models_dir / 'clothing_classifier.keras'))
    
    print("\n" + "=" * 70)
    print("✅ Training Pipeline Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
