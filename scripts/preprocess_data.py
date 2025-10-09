"""
Data Preprocessing Module
Prepares the Fashion Product Images dataset for model training.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import json
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2


class FashionDataPreprocessor:
    """Preprocesses the Kaggle Fashion Product Images dataset."""
    
    # Clothing category mapping
    CATEGORY_MAPPING = {
        'Topwear': ['Shirts', 'Tshirts', 'Tops', 'Sweaters', 'Sweatshirts', 'Jackets', 'Blazers'],
        'Bottomwear': ['Jeans', 'Trousers', 'Shorts', 'Skirts', 'Leggings', 'Capris'],
        'Dress': ['Dresses', 'Jumpsuit', 'Kurtas', 'Kurtis'],
        'Footwear': ['Shoes', 'Sandals', 'Flip Flops', 'Heels', 'Casual Shoes', 'Formal Shoes', 'Sports Shoes'],
        'Accessories': ['Watches', 'Bags', 'Belts', 'Sunglasses', 'Wallets', 'Scarves'],
    }
    
    def __init__(self, data_dir: str, output_dir: str = None, img_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the preprocessor.
        
        Args:
            data_dir: Directory containing the raw dataset
            output_dir: Directory to save processed data
            img_size: Target image size for model input
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir.parent / 'processed'
        self.img_size = img_size
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths
        self.styles_csv = self.data_dir / 'styles.csv'
        self.images_dir = self.data_dir / 'images'
        
    def load_and_clean_data(self) -> pd.DataFrame:
        """
        Load and clean the styles CSV file.
        
        Returns:
            Cleaned DataFrame
        """
        print("📂 Loading dataset...")
        
        # Load CSV
        df = pd.read_csv(self.styles_csv, on_bad_lines='skip')
        
        print(f"Initial dataset size: {len(df)} items")
        print(f"Columns: {df.columns.tolist()}")
        
        # Remove rows with missing critical data
        df = df.dropna(subset=['id'])
        
        # Add image path column
        df['image_path'] = df['id'].apply(lambda x: str(self.images_dir / f"{x}.jpg"))
        
        # Filter out items where image doesn't exist
        df['image_exists'] = df['image_path'].apply(lambda x: os.path.exists(x))
        df = df[df['image_exists']].copy()
        df = df.drop('image_exists', axis=1)
        
        print(f"Dataset after filtering: {len(df)} items")
        
        return df
    
    def create_category_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create simplified category labels for classification.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with category labels
        """
        print("\n🏷️  Creating category labels...")
        
        def get_main_category(row):
            """Determine main category from article type."""
            article_type = str(row.get('articleType', ''))
            
            for main_cat, sub_cats in self.CATEGORY_MAPPING.items():
                if article_type in sub_cats:
                    return main_cat
            
            return 'Other'
        
        df['main_category'] = df.apply(get_main_category, axis=1)
        
        # Create numerical labels
        category_to_idx = {cat: idx for idx, cat in enumerate(sorted(df['main_category'].unique()))}
        df['category_label'] = df['main_category'].map(category_to_idx)
        
        # Save label mapping
        label_mapping_path = self.output_dir / 'label_mapping.json'
        with open(label_mapping_path, 'w') as f:
            json.dump(category_to_idx, f, indent=2)
        
        print(f"Category distribution:")
        print(df['main_category'].value_counts())
        
        return df
    
    def extract_color_features(self, df: pd.DataFrame, sample_size: int = None) -> pd.DataFrame:
        """
        Extract dominant color features from images.
        
        Args:
            df: Input DataFrame
            sample_size: Number of samples to process (None for all)
            
        Returns:
            DataFrame with color features
        """
        print("\n🎨 Extracting color features...")
        
        if sample_size:
            df_sample = df.sample(n=min(sample_size, len(df))).copy()
        else:
            df_sample = df.copy()
        
        colors_rgb = []
        colors_hsv = []
        
        for _, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc="Processing images"):
            try:
                img_path = row['image_path']
                img = cv2.imread(img_path)
                
                if img is not None:
                    # Resize to speed up processing
                    img_small = cv2.resize(img, (100, 100))
                    
                    # Calculate median color (RGB)
                    pixels = img_small.reshape(-1, 3)
                    median_color_bgr = np.median(pixels, axis=0).astype(int)
                    median_color_rgb = median_color_bgr[::-1]  # BGR to RGB
                    
                    # Convert to HSV
                    img_hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
                    median_color_hsv = np.median(img_hsv.reshape(-1, 3), axis=0).astype(int)
                    
                    colors_rgb.append(tuple(median_color_rgb))
                    colors_hsv.append(tuple(median_color_hsv))
                else:
                    colors_rgb.append((0, 0, 0))
                    colors_hsv.append((0, 0, 0))
                    
            except Exception as e:
                colors_rgb.append((0, 0, 0))
                colors_hsv.append((0, 0, 0))
        
        df_sample['dominant_color_rgb'] = colors_rgb
        df_sample['dominant_color_hsv'] = colors_hsv
        
        return df_sample
    
    def create_train_val_test_split(self, df: pd.DataFrame, 
                                    train_ratio: float = 0.7,
                                    val_ratio: float = 0.15,
                                    test_ratio: float = 0.15) -> Dict[str, pd.DataFrame]:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            df: Input DataFrame
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            
        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        print("\n📊 Splitting dataset...")
        
        # First split: train and temp (val + test)
        train_df, temp_df = train_test_split(
            df,
            test_size=(1 - train_ratio),
            stratify=df['main_category'],
            random_state=42
        )
        
        # Second split: val and test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_size),
            stratify=temp_df['main_category'],
            random_state=42
        )
        
        print(f"Train set: {len(train_df)} samples")
        print(f"Validation set: {len(val_df)} samples")
        print(f"Test set: {len(test_df)} samples")
        
        # Save splits
        train_df.to_csv(self.output_dir / 'train.csv', index=False)
        val_df.to_csv(self.output_dir / 'val.csv', index=False)
        test_df.to_csv(self.output_dir / 'test.csv', index=False)
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess a single image for model input.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image array
        """
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize(self.img_size)
            img_array = np.array(img) / 255.0  # Normalize to [0, 1]
            return img_array
        except Exception as e:
            # Return blank image on error
            return np.zeros((*self.img_size, 3))
    
    def create_dataset_summary(self, splits: Dict[str, pd.DataFrame]):
        """
        Create and save dataset summary statistics.
        
        Args:
            splits: Dictionary of train/val/test splits
        """
        print("\n📈 Creating dataset summary...")
        
        summary = {
            'total_samples': sum(len(df) for df in splits.values()),
            'image_size': self.img_size,
            'splits': {
                name: {
                    'size': len(df),
                    'categories': df['main_category'].value_counts().to_dict()
                }
                for name, df in splits.items()
            }
        }
        
        summary_path = self.output_dir / 'dataset_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Summary saved to {summary_path}")
    
    def run_full_pipeline(self, extract_colors: bool = False, color_sample_size: int = 5000):
        """
        Run the complete preprocessing pipeline.
        
        Args:
            extract_colors: Whether to extract color features
            color_sample_size: Number of samples for color extraction
        """
        print("=" * 70)
        print("🚀 Fashion Dataset Preprocessing Pipeline")
        print("=" * 70)
        
        # Load and clean data
        df = self.load_and_clean_data()
        
        # Create category labels
        df = self.create_category_labels(df)
        
        # Extract color features (optional, can be slow)
        if extract_colors:
            df = self.extract_color_features(df, sample_size=color_sample_size)
        
        # Create splits
        splits = self.create_train_val_test_split(df)
        
        # Create summary
        self.create_dataset_summary(splits)
        
        print("\n" + "=" * 70)
        print("✅ Preprocessing Complete!")
        print("=" * 70)
        print(f"\nProcessed data saved to: {self.output_dir}")
        print("\nNext steps:")
        print("  1. Review the dataset summary at: dataset_summary.json")
        print("  2. Check label mappings at: label_mapping.json")
        print("  3. Use train.csv, val.csv, test.csv for model training")


def main():
    """Main preprocessing script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess Fashion Product Images dataset')
    parser.add_argument('--data_dir', type=str, default='../data/raw',
                       help='Directory containing raw dataset')
    parser.add_argument('--output_dir', type=str, default='../data/processed',
                       help='Directory to save processed data')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Target image size')
    parser.add_argument('--extract_colors', action='store_true',
                       help='Extract color features (slower)')
    parser.add_argument('--color_samples', type=int, default=5000,
                       help='Number of samples for color extraction')
    
    args = parser.parse_args()
    
    # Get absolute paths
    script_dir = Path(__file__).parent
    data_dir = (script_dir / args.data_dir).resolve()
    output_dir = (script_dir / args.output_dir).resolve()
    
    # Run preprocessing
    preprocessor = FashionDataPreprocessor(
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        img_size=(args.img_size, args.img_size)
    )
    
    preprocessor.run_full_pipeline(
        extract_colors=args.extract_colors,
        color_sample_size=args.color_samples
    )


if __name__ == "__main__":
    main()
