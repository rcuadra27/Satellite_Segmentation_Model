#!/usr/bin/env python3
"""
Script to download and train on real satellite data (DeepGlobe)
"""

import sys
from pathlib import Path
import subprocess
import zipfile
import shutil

def download_deepglobe():
    print("🌍 Downloading Real Satellite Data (DeepGlobe)")
    print("=" * 60)
    
    print("This will download ~6GB of real satellite images with segmentation masks.")
    print("The DeepGlobe dataset contains:")
    print("  - 803 real satellite images")
    print("  - 7 land cover classes")
    print("  - Much more diverse than our 20 synthetic images")
    
    response = input("\nDo you want to proceed? (y/n): ")
    if response.lower() != 'y':
        print("Skipping download.")
        return False
    
    # Check if kaggle is installed
    try:
        subprocess.run(['kaggle', '--version'], check=True, capture_output=True)
        print("✅ Kaggle CLI found")
    except:
        print("❌ Kaggle CLI not found. Please install it first:")
        print("   pip install kaggle")
        print("   Then get API credentials from: https://www.kaggle.com/settings")
        return False
    
    # Download dataset
    print("\n📥 Downloading DeepGlobe dataset...")
    try:
        subprocess.run([
            'kaggle', 'datasets', 'download', 
            '-d', 'balraj98/deepglobe-land-cover-classification-dataset',
            '-p', 'data/real/'
        ], check=True)
        print("✅ Download completed")
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")
        return False
    
    # Extract dataset
    print("\n📦 Extracting dataset...")
    zip_path = Path('data/real/deepglobe-land-cover-classification-dataset.zip')
    
    if not zip_path.exists():
        print("❌ Downloaded zip file not found")
        return False
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('data/real/')
    
    print("✅ Extraction completed")
    
    # Organize files
    print("\n📁 Organizing files...")
    extracted_dir = Path('data/real/deepglobe-land-cover-classification-dataset')
    
    if extracted_dir.exists():
        # Create organized directories
        images_dir = Path('data/real/images')
        masks_dir = Path('data/real/masks')
        images_dir.mkdir(exist_ok=True)
        masks_dir.mkdir(exist_ok=True)
        
        # Move files
        moved_images = 0
        moved_masks = 0
        
        for f in extracted_dir.glob('*sat.jpg'):
            shutil.move(str(f), str(images_dir / f.name))
            moved_images += 1
        
        for f in extracted_dir.glob('*mask.png'):
            shutil.move(str(f), str(masks_dir / f.name))
            moved_masks += 1
        
        print(f"✅ Organized {moved_images} images and {moved_masks} masks")
        
        # Clean up
        shutil.rmtree(extracted_dir)
        zip_path.unlink()
        
        print("✅ Cleanup completed")
        print(f"\n📊 Real dataset ready:")
        print(f"  Images: {images_dir}")
        print(f"  Masks: {masks_dir}")
        print(f"  Total: {moved_images} image-mask pairs")
        
        return True
    
    return False

def create_real_data_training_script():
    """Create a training script for real data"""
    
    script_content = '''#!/usr/bin/env python3
"""
Training script for real DeepGlobe satellite data
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from datetime import datetime

# Add src to path
project_root = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(project_root / 'src'))

from data_loader import SatelliteDataLoader, split_train_val_test
from models import unet_model
from utils import check_gpu_availability

def train_on_real_data():
    print("🌍 Training on Real Satellite Data (DeepGlobe)")
    print("=" * 60)
    
    check_gpu_availability()
    
    # Set up paths for real data
    images_dir = Path('data/raw/deepglobe_images')
    masks_dir = Path('data/raw/deepglobe_masks')
    
    if not images_dir.exists() or not masks_dir.exists():
        print("❌ DeepGlobe data not found. Please run download_deepglobe() first.")
        return False
    
    # Create data loader for real data
    data_loader = SatelliteDataLoader(
        image_dir=str(images_dir),
        mask_dir=str(masks_dir),
        image_size=(256, 256),  # Resize for efficiency
        num_classes=7,  # DeepGlobe has 7 classes
        batch_size=8
    )
    
    # Get all data pairs
    data_pairs = data_loader.get_image_mask_pairs()
    print(f"Found {len(data_pairs)} real satellite image-mask pairs")
    
    # Split data
    train_pairs, val_pairs, test_pairs = split_train_val_test(
        data_pairs,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )
    
    print(f"Data split:")
    print(f"  Training: {len(train_pairs)} samples")
    print(f"  Validation: {len(val_pairs)} samples")
    print(f"  Test: {len(test_pairs)} samples")
    
    # Create datasets
    train_dataset = data_loader.create_tf_dataset(
        [str(pair[0]) for pair in train_pairs],
        [str(pair[1]) for pair in train_pairs],
        augment=True
    )
    
    val_dataset = data_loader.create_tf_dataset(
        [str(pair[0]) for pair in val_pairs],
        [str(pair[1]) for pair in val_pairs],
        augment=False
    )
    
    # Fix data types
    def fix_dtypes(image, mask):
        image = tf.cast(image, tf.float32)
        mask = tf.cast(mask, tf.int32)
        return image, mask
    
    train_dataset = train_dataset.map(fix_dtypes)
    val_dataset = val_dataset.map(fix_dtypes)
    
    # Create model
    model = unet_model(
        input_shape=(256, 256, 3),
        num_classes=7,
        filters=32,
        dropout_rate=0.3
    )
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'models/unet_real_data_best.h5',
            monitor='val_loss',
            save_best_only=True
        )
    ]
    
    # Train
    print("\\n🚀 Starting training on real data...")
    print("⚠️  This will take 2-4 hours on CPU!")
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=100,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\\n✅ Training completed!")
    print("📁 Best model saved: models/unet_real_data_best.h5")
    
    return True

if __name__ == "__main__":
    train_on_real_data()
'''
    
    with open('train_real_data.py', 'w') as f:
        f.write(script_content)
    
    print("✅ Created train_real_data.py script")

if __name__ == "__main__":
    print("🌍 DeepGlobe Dataset Setup")
    print("=" * 40)
    print("Choose an option:")
    print("1. Download DeepGlobe dataset (~6GB)")
    print("2. Create training script for real data")
    print("3. Both")
    
    choice = input("\nEnter choice (1/2/3): ")
    
    if choice in ['1', '3']:
        download_deepglobe()
    
    if choice in ['2', '3']:
        create_real_data_training_script()
    
    print("\n🎯 Next steps:")
    print("1. Run: python train_improved.py (better regularization)")
    print("2. Or run: python train_real_data.py (real satellite data)")
    print("3. Compare results!")
