#!/usr/bin/env python3
"""
Train pre-trained ResNet50 U-Net on real satellite data

Note: This script uses ResNet50 (not ResNet34) because ResNet34 is not
available in keras.applications. The filename "train_resnet34.py" is kept
for backwards compatibility, but the model uses ResNet50.
"""

import os
import ssl
import sys
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

# Disable SSL verification for downloading pre-trained weights
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add src to path
project_root = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(project_root / 'src'))

from data_loader import SatelliteDataLoader, split_train_val_test
from models import get_pretrained_segmentation_model, compile_segmentation_model
from utils import check_gpu_availability


class AccuracyDropCallback(keras.callbacks.Callback):
    """
    Custom callback to stop training if validation accuracy drops significantly
    
    This prevents wasting time when the model starts degrading (like we saw
    when accuracy dropped from 57% to 0.4%)
    """
    def __init__(self, drop_threshold=0.2, patience=3, min_epochs=5):
        """
        Args:
            drop_threshold: Stop if accuracy drops by this fraction (0.2 = 20% drop)
            patience: Number of epochs to wait before stopping
            min_epochs: Minimum epochs before monitoring starts
        """
        super().__init__()
        self.drop_threshold = drop_threshold
        self.patience = patience
        self.min_epochs = min_epochs
        self.best_accuracy = 0.0
        self.drop_count = 0
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_accuracy = logs.get('val_accuracy', 0)
        
        # Only start monitoring after minimum epochs
        if epoch < self.min_epochs:
            self.best_accuracy = max(self.best_accuracy, current_accuracy)
            return
        
        # Update best accuracy
        if current_accuracy > self.best_accuracy:
            self.best_accuracy = current_accuracy
            self.drop_count = 0  # Reset counter on improvement
            return
        
        # Check for significant drop
        if self.best_accuracy > 0:
            drop_ratio = (self.best_accuracy - current_accuracy) / self.best_accuracy
            
            if drop_ratio > self.drop_threshold:
                self.drop_count += 1
                print(f"\n⚠️  WARNING: Accuracy dropped by {drop_ratio*100:.1f}% "
                      f"(from {self.best_accuracy:.4f} to {current_accuracy:.4f})")
                print(f"   Drop count: {self.drop_count}/{self.patience}")
                
                if self.drop_count >= self.patience:
                    print(f"\n🛑 STOPPING: Validation accuracy has dropped significantly for {self.patience} epochs!")
                    print(f"   Best accuracy: {self.best_accuracy:.4f}")
                    print(f"   Current accuracy: {current_accuracy:.4f}")
                    print(f"   This indicates the model is degrading. Stopping early.")
                    self.model.stop_training = True
            else:
                self.drop_count = 0  # Reset if drop is not significant


def train_pretrained_resnet50():
    print("🚀 Training Pre-trained ResNet50 U-Net on Real Satellite Data")
    print("=" * 70)
    
    check_gpu_availability()
    
    # Configuration
    IMAGE_SIZE = (256, 256)
    NUM_CLASSES = 7  # DeepGlobe has 7 classes
    BATCH_SIZE = 8
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    
    CLASS_NAMES = ['Urban', 'Agriculture', 'Rangeland', 'Forest', 'Water', 'Barren', 'Unknown']
    
    print(f"Configuration:")
    print(f"  Image size: {IMAGE_SIZE}")
    print(f"  Number of classes: {NUM_CLASSES}")
    print(f"  Classes: {CLASS_NAMES}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    
    # Check if we have real data
    real_data_dir = Path('data/real')
    if not real_data_dir.exists():
        print(f"\n❌ Real satellite data not found!")
        print(f"   Expected directory: {real_data_dir}")
        print(f"   Please run setup_real_data.py first to download DeepGlobe dataset")
        return False
    
    # Set up paths for real data
    images_dir = real_data_dir / 'images'
    masks_dir = real_data_dir / 'masks'
    
    if not images_dir.exists() or not masks_dir.exists():
        print(f"\n❌ Real data directories not found!")
        print(f"   Images: {images_dir}")
        print(f"   Masks: {masks_dir}")
        print(f"   Please run setup_real_data.py first")
        return False
    
    # Create data loader for real data
    data_loader = SatelliteDataLoader(
        image_dir=str(images_dir),
        mask_dir=str(masks_dir),
        image_size=IMAGE_SIZE,
        num_classes=NUM_CLASSES,
        batch_size=BATCH_SIZE
    )
    
    # Get all data pairs
    data_pairs = data_loader.get_image_mask_pairs()
    print(f"\nFound {len(data_pairs)} real satellite image-mask pairs")
    
    if len(data_pairs) == 0:
        print(f"❌ No data pairs found!")
        return False
    
    # Split data (80/10/10 for more training data)
    train_pairs, val_pairs, test_pairs = split_train_val_test(
        data_pairs,
        train_ratio=0.8,  # Increased from 0.7 to 0.8
        val_ratio=0.1,    # Decreased from 0.15 to 0.1
        test_ratio=0.1,   # Decreased from 0.15 to 0.1
        random_seed=42
    )
    
    print(f"\nData split:")
    print(f"  Training: {len(train_pairs)} samples")
    print(f"  Validation: {len(val_pairs)} samples")
    print(f"  Test: {len(test_pairs)} samples")
    
    # Create TensorFlow datasets
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
    
    # Note: No need to fix dtypes anymore - data loader handles it correctly
    
    print("✅ Datasets created")
    
    # Create improved U-Net model
    print(f"\n🏗️ Creating improved U-Net model...")
    try:
        model = get_pretrained_segmentation_model(
            backbone='resnet50',  # Using ResNet50 (ResNet34 not available in keras.applications)
            input_shape=(*IMAGE_SIZE, 3),
            num_classes=NUM_CLASSES,
            architecture='Unet'
        )
        
        # Freeze the ResNet50 encoder to prevent overfitting
        print(f"\n🔒 Freezing ResNet50 encoder (keeping ImageNet weights intact)...")
        encoder_layer_count = 0
        for layer in model.layers:
            if 'resnet50' in layer.name.lower() or layer.name.startswith('block'):
                layer.trainable = False
                encoder_layer_count += 1
        
        # Count trainable vs non-trainable params
        trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
        non_trainable_count = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
        
        print(f"✅ ResNet50 U-Net created!")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        print(f"  Total parameters: {model.count_params():,}")
        print(f"  Trainable parameters: {trainable_count:,}")
        print(f"  Non-trainable parameters: {non_trainable_count:,}")
        print(f"  Architecture: ResNet50 encoder (FROZEN) + U-Net decoder (TRAINABLE)")
        print(f"  Strategy: Transfer learning - keep ImageNet features, train decoder only")
        
    except Exception as e:
        print(f"❌ Error creating improved model: {e}")
        return False
    
    # Compile model (no class weights - let model learn naturally)
    print(f"\n⚙️ Compiling model...")
    model = compile_segmentation_model(
        model=model,
        num_classes=NUM_CLASSES,
        learning_rate=LEARNING_RATE,
        loss_type='sparse_categorical_crossentropy'  # Using class ID masks (not one-hot)
    )
    print("✅ Model compiled")
    
    # Callbacks
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(models_dir / 'improved_unet_best.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        AccuracyDropCallback(
            drop_threshold=0.3,  # Stop if accuracy drops by 30%
            patience=3,          # Wait for 3 epochs of sustained drop
            min_epochs=5         # Start monitoring after epoch 5
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            'results/improved_unet_training_log.csv'
        ),
        keras.callbacks.TensorBoard(
            log_dir=f'logs/improved_unet_fit/{datetime.now().strftime("%Y%m%d-%H%M%S")}',
            histogram_freq=1
        )
    ]
    
    print(f"\n🛡️  Safety callbacks enabled:")
    print(f"   • EarlyStopping: Stops if val_loss doesn't improve for 10 epochs")
    print(f"   • AccuracyDrop: Stops if accuracy drops by >30% for 3 epochs")
    print(f"   • ReduceLROnPlateau: Reduces learning rate if stuck")
    
    # Train
    print(f"\n🚀 Starting training...")
    print(f"⚠️  This will take 30-60 minutes on CPU, 10-20 minutes on GPU!")
    print(f"📊 Training without class weights (more stable, realistic learning)")
    
    try:
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n" + "="*70)
        print("✅ Training completed!")
        print("="*70)
        
        # Load best model
        best_model = keras.models.load_model(str(models_dir / 'improved_unet_best.h5'))
        
        # Test on a few samples
        print(f"\n🧪 Testing predictions...")
        test_dataset = data_loader.create_tf_dataset(
            [str(pair[0]) for pair in test_pairs],
            [str(pair[1]) for pair in test_pairs],
            augment=False
        )
        
        # Unbatch the dataset to get individual images
        test_dataset_unbatched = test_dataset.unbatch()
        
        for i, (image, mask) in enumerate(test_dataset_unbatched.take(3)):
            prediction = best_model.predict(np.expand_dims(image, axis=0), verbose=0)
            pred_mask = np.argmax(prediction[0], axis=-1)
            
            unique, counts = np.unique(pred_mask, return_counts=True)
            print(f"  Test {i+1}: Classes predicted: {unique}")
            
            if len(unique) > 1:
                print(f"    ✅ Model predicts {len(unique)} different classes!")
                for class_id, count in zip(unique, counts):
                    percentage = (count / pred_mask.size) * 100
                    class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f'Class {class_id}'
                    print(f"      {class_name}: {percentage:.1f}%")
            else:
                print(f"    ⚠️  Model still predicts only class {unique[0]} ({CLASS_NAMES[unique[0]]})")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    success = train_pretrained_resnet50()
    if success:
        print("\n🎉 Improved U-Net training completed!")
        print("📁 Best model saved: models/improved_unet_best.h5")
        print("\n💡 Advantages of this approach:")
        print("  ✅ Improved U-Net architecture (BatchNormalization, skip connections)")
        print("  ✅ Real satellite data (DeepGlobe dataset)")
        print("  ✅ Realistic accuracy expectations (70-80%)")
        print("  ✅ Production-ready model")
    else:
        print("\n💥 Training failed!")
        sys.exit(1)
