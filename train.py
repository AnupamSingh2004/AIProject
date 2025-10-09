"""
Master Training Script
Runs the complete training pipeline for the fashion recommendation system.
"""

import sys
from pathlib import Path
import argparse

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'scripts'))
sys.path.insert(0, str(project_root / 'models'))


def run_preprocessing(args):
    """Run data preprocessing."""
    print("\n" + "=" * 80)
    print("STEP 1: DATA PREPROCESSING")
    print("=" * 80)
    
    from preprocess_data import FashionDataPreprocessor
    
    preprocessor = FashionDataPreprocessor(
        data_dir=str(args.data_dir),
        output_dir=str(args.processed_dir),
        img_size=(args.img_size, args.img_size)
    )
    
    preprocessor.run_full_pipeline(
        extract_colors=args.extract_colors,
        color_sample_size=args.color_samples
    )
    
    print("\n✅ Preprocessing complete!\n")


def run_classifier_training(args):
    """Train clothing classification model."""
    print("\n" + "=" * 80)
    print("STEP 2: CLOTHING CLASSIFICATION MODEL TRAINING")
    print("=" * 80)
    
    import json
    import tensorflow as tf
    from clothing_classifier import ClothingClassificationModel
    import pandas as pd
    
    # Load label mapping
    with open(args.processed_dir / 'label_mapping.json', 'r') as f:
        label_mapping = json.load(f)
    
    num_classes = len(label_mapping)
    print(f"\nTraining classifier for {num_classes} clothing categories")
    
    # Create model
    model = ClothingClassificationModel(
        num_classes=num_classes,
        img_size=(args.img_size, args.img_size),
        model_name=args.model_architecture,
        use_pretrained=True
    )
    
    # Build model
    model.build_model()
    
    # Create data generators
    train_dataset, val_dataset = model.create_data_generators(
        train_csv=str(args.processed_dir / 'train.csv'),
        val_csv=str(args.processed_dir / 'val.csv'),
        batch_size=args.batch_size
    )
    
    # Phase 1: Train with frozen base
    print("\n📚 Phase 1: Training with frozen base model")
    model.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=args.epochs_phase1
    )
    
    # Phase 2: Fine-tune
    if args.fine_tune:
        print("\n🔥 Phase 2: Fine-tuning")
        model.unfreeze_base_model(layers_to_unfreeze=-30)
        model.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=args.epochs_phase2
        )
    
    # Evaluate
    test_df = pd.read_csv(args.processed_dir / 'test.csv')
    test_paths = test_df['image_path'].values
    test_labels = test_df['category_label'].values
    
    test_dataset = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
    test_dataset = test_dataset.map(
        lambda x, y: (tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(x), channels=3), (args.img_size, args.img_size)) / 255.0, y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    test_dataset = test_dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    
    metrics = model.evaluate(test_dataset)
    
    # Save model
    model_path = args.models_dir / 'clothing_classifier.keras'
    model.save_model(str(model_path))
    
    # Plot history
    plot_path = args.models_dir / 'classifier_training_history.png'
    model.plot_training_history(save_path=str(plot_path))
    
    print(f"\n✅ Classifier training complete!")
    print(f"   Test Accuracy: {metrics['accuracy']:.2%}")
    print(f"   Model saved to: {model_path}")


def run_compatibility_training(args):
    """Train outfit compatibility model."""
    print("\n" + "=" * 80)
    print("STEP 3: OUTFIT COMPATIBILITY MODEL TRAINING")
    print("=" * 80)
    
    import pandas as pd
    from outfit_compatibility_model import OutfitCompatibilityModel
    
    # Load data
    train_df = pd.read_csv(args.processed_dir / 'train.csv')
    val_df = pd.read_csv(args.processed_dir / 'val.csv')
    
    # Create model
    model = OutfitCompatibilityModel(feature_dim=128)
    model.build_compatibility_model(img_size=(args.img_size, args.img_size))
    
    # Create outfit pairs
    train_pairs = model.create_outfit_pairs_dataset(
        train_df,
        num_positive=args.num_positive_pairs,
        num_negative=args.num_negative_pairs
    )
    val_pairs = model.create_outfit_pairs_dataset(
        val_df,
        num_positive=args.num_positive_pairs // 6,
        num_negative=args.num_negative_pairs // 6
    )
    
    # Create datasets
    train_dataset = model.create_tf_dataset(train_pairs, batch_size=args.batch_size)
    val_dataset = model.create_tf_dataset(val_pairs, batch_size=args.batch_size)
    
    # Train
    history = model.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=args.compatibility_epochs
    )
    
    # Save model
    model_path = args.models_dir / 'outfit_compatibility.keras'
    model.save_model(str(model_path))
    
    print(f"\n✅ Compatibility model training complete!")
    print(f"   Model saved to: {model_path}")


def main():
    """Main training orchestrator."""
    parser = argparse.ArgumentParser(description='Train Fashion Recommendation System Models')
    
    # Paths
    parser.add_argument('--data_dir', type=Path, default=Path('data/raw'),
                       help='Raw dataset directory')
    parser.add_argument('--processed_dir', type=Path, default=Path('data/processed'),
                       help='Processed data directory')
    parser.add_argument('--models_dir', type=Path, default=Path('models/saved_models'),
                       help='Directory to save trained models')
    
    # Pipeline steps
    parser.add_argument('--skip_preprocessing', action='store_true',
                       help='Skip data preprocessing step')
    parser.add_argument('--skip_classifier', action='store_true',
                       help='Skip classifier training')
    parser.add_argument('--skip_compatibility', action='store_true',
                       help='Skip compatibility model training')
    
    # Preprocessing options
    parser.add_argument('--extract_colors', action='store_true',
                       help='Extract color features during preprocessing')
    parser.add_argument('--color_samples', type=int, default=5000,
                       help='Number of samples for color extraction')
    
    # Model options
    parser.add_argument('--img_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--model_architecture', type=str, default='efficientnet',
                       choices=['efficientnet', 'mobilenet', 'resnet'],
                       help='Base model architecture')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    
    # Classifier training
    parser.add_argument('--epochs_phase1', type=int, default=10,
                       help='Epochs for phase 1 (frozen base)')
    parser.add_argument('--epochs_phase2', type=int, default=10,
                       help='Epochs for phase 2 (fine-tuning)')
    parser.add_argument('--fine_tune', action='store_true', default=True,
                       help='Enable fine-tuning phase')
    
    # Compatibility training
    parser.add_argument('--compatibility_epochs', type=int, default=20,
                       help='Epochs for compatibility model')
    parser.add_argument('--num_positive_pairs', type=int, default=3000,
                       help='Number of positive outfit pairs')
    parser.add_argument('--num_negative_pairs', type=int, default=3000,
                       help='Number of negative outfit pairs')
    
    args = parser.parse_args()
    
    # Create directories
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("🎨 FASHION RECOMMENDATION SYSTEM - COMPLETE TRAINING PIPELINE")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data Directory: {args.data_dir}")
    print(f"  Processed Directory: {args.processed_dir}")
    print(f"  Models Directory: {args.models_dir}")
    print(f"  Image Size: {args.img_size}x{args.img_size}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Model Architecture: {args.model_architecture}")
    
    # Step 1: Preprocessing
    if not args.skip_preprocessing:
        try:
            run_preprocessing(args)
        except Exception as e:
            print(f"\n❌ Error in preprocessing: {e}")
            print("Continuing with next steps...")
    
    # Step 2: Classifier Training
    if not args.skip_classifier:
        try:
            run_classifier_training(args)
        except Exception as e:
            print(f"\n❌ Error in classifier training: {e}")
            print("Continuing with next steps...")
    
    # Step 3: Compatibility Training
    if not args.skip_compatibility:
        try:
            run_compatibility_training(args)
        except Exception as e:
            print(f"\n❌ Error in compatibility training: {e}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎉 TRAINING PIPELINE COMPLETE!")
    print("=" * 80)
    print("\nTrained Models:")
    print(f"  1. Clothing Classifier: {args.models_dir / 'clothing_classifier.keras'}")
    print(f"  2. Outfit Compatibility: {args.models_dir / 'outfit_compatibility.keras'}")
    print("\nNext Steps:")
    print("  1. Test the models using the demo application")
    print("  2. Run: streamlit run app/streamlit_app.py")
    print("  3. Upload your photo and clothing items")
    print("  4. Get personalized recommendations!")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
