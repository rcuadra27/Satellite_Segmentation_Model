"""
Data loading utilities for satellite image segmentation
"""

import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Tuple, List, Optional
import cv2
from PIL import Image


class SatelliteDataLoader:
    """Load and preprocess satellite images and segmentation masks"""
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        image_size: Tuple[int, int] = (256, 256),
        num_classes: int = 5,
        batch_size: int = 16
    ):
        """
        Args:
            image_dir: Directory containing satellite images
            mask_dir: Directory containing segmentation masks
            image_size: Target image size (height, width)
            num_classes: Number of segmentation classes
            batch_size: Batch size for training
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        
    def load_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess a single image"""
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        return image
    
    def load_mask(self, mask_path: str) -> np.ndarray:
        """Load and preprocess a single mask"""
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        return mask
    
    def get_image_mask_pairs(self) -> List[Tuple[Path, Path]]:
        """Get pairs of images and their corresponding masks"""
        image_files = sorted(self.image_dir.glob("*.png")) + \
                      sorted(self.image_dir.glob("*.jpg")) + \
                      sorted(self.image_dir.glob("*.tif"))
        
        pairs = []
        for img_path in image_files:
            # Extract base name (remove extension)
            base_name = img_path.stem
            
            # Handle DeepGlobe naming: 100694_sat.jpg -> 100694_mask.png
            if '_sat' in base_name:
                base_name = base_name.replace('_sat', '')
            
            # Try to find matching mask
            # First try exact name match
            mask_path = self.mask_dir / img_path.name
            if not mask_path.exists():
                # Try DeepGlobe pattern: look for *_mask.png
                mask_candidates = list(self.mask_dir.glob(f'{base_name}_mask.png'))
                if mask_candidates:
                    mask_path = mask_candidates[0]
                else:
                    # Try with just base name
                    mask_path = self.mask_dir / f'{base_name}.png'
            
            if mask_path.exists():
                pairs.append((img_path, mask_path))
        
        return pairs
    
    def create_tf_dataset(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        augment: bool = False
    ) -> tf.data.Dataset:
        """Create a TensorFlow dataset from image and mask paths"""
        
        def rgb_to_class_id(mask_rgb):
            """
            Convert RGB mask to class IDs for DeepGlobe dataset
            
            DeepGlobe color encoding:
              0: Urban       - RGB(0, 255, 255) - Cyan
              1: Agriculture - RGB(255, 255, 0) - Yellow
              2: Rangeland   - RGB(255, 0, 255) - Magenta
              3: Forest      - RGB(0, 255, 0)   - Green
              4: Water       - RGB(0, 0, 255)   - Blue
              5: Barren      - RGB(255, 255, 255) - White
              6: Unknown     - RGB(0, 0, 0)     - Black
            """
            # Define color mappings
            class_colors = tf.constant([
                [0, 255, 255],    # 0: Urban (Cyan)
                [255, 255, 0],    # 1: Agriculture (Yellow)
                [255, 0, 255],    # 2: Rangeland (Magenta)
                [0, 255, 0],      # 3: Forest (Green)
                [0, 0, 255],      # 4: Water (Blue)
                [255, 255, 255],  # 5: Barren (White)
                [0, 0, 0],        # 6: Unknown (Black)
            ], dtype=tf.float32)
            
            # Reshape mask for broadcasting
            h, w, c = tf.shape(mask_rgb)[0], tf.shape(mask_rgb)[1], tf.shape(mask_rgb)[2]
            mask_flat = tf.reshape(mask_rgb, [-1, 3])  # (H*W, 3)
            
            # Compute distances to each class color
            # Using L1 distance for speed
            distances = tf.reduce_sum(tf.abs(tf.expand_dims(mask_flat, 1) - class_colors), axis=2)  # (H*W, 7)
            
            # Get the closest class
            class_ids = tf.argmin(distances, axis=1, output_type=tf.int32)  # (H*W,)
            
            # Reshape back to image dimensions
            class_mask = tf.reshape(class_ids, [h, w])  # (H, W)
            
            return class_mask
        
        def load_data(image_path, mask_path):
            """Load image and mask using TensorFlow"""
            # Load image
            image = tf.io.read_file(image_path)
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
            image = tf.image.resize(image, self.image_size)
            image = tf.cast(image, tf.float32) / 255.0
            
            # Load mask as RGB (DeepGlobe masks are RGB-encoded)
            mask = tf.io.read_file(mask_path)
            mask = tf.image.decode_image(mask, channels=3, expand_animations=False)  # Load as RGB!
            mask = tf.image.resize(mask, self.image_size, method='nearest')
            mask = tf.cast(mask, tf.float32)
            
            # Convert RGB mask to class IDs (0-6)
            mask_class_ids = rgb_to_class_id(mask)
            
            # Add channel dimension for compatibility: (H, W) -> (H, W, 1)
            mask_class_ids = tf.expand_dims(mask_class_ids, axis=-1)
            
            return image, mask_class_ids
        
        def augment_data(image, mask):
            """Apply data augmentation - only to images"""
            # Random flip left-right
            if tf.random.uniform(()) > 0.5:
                image = tf.image.flip_left_right(image)
            
            # Random flip up-down
            if tf.random.uniform(()) > 0.5:
                image = tf.image.flip_up_down(image)
            
            # Random brightness
            if tf.random.uniform(()) > 0.5:
                image = tf.image.adjust_brightness(image, 0.1)
            
            # Random contrast
            if tf.random.uniform(()) > 0.5:
                image = tf.image.adjust_contrast(image, 1.2)
            
            return image, mask
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
        dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
        
        if augment:
            dataset = dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset


def normalize_satellite_image(image: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize satellite image
    
    Args:
        image: Input image
        method: 'minmax' or 'zscore'
    
    Returns:
        Normalized image
    """
    if method == 'minmax':
        img_min = image.min()
        img_max = image.max()
        if img_max - img_min > 0:
            return (image - img_min) / (img_max - img_min)
        return image
    elif method == 'zscore':
        mean = image.mean()
        std = image.std()
        if std > 0:
            return (image - mean) / std
        return image - mean
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def split_train_val_test(
    data_pairs: List[Tuple],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[List, List, List]:
    """
    Split data into train, validation, and test sets
    
    Args:
        data_pairs: List of (image_path, mask_path) tuples
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        random_seed: Random seed for reproducibility
    
    Returns:
        train_pairs, val_pairs, test_pairs
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    np.random.seed(random_seed)
    indices = np.random.permutation(len(data_pairs))
    
    n_train = int(len(data_pairs) * train_ratio)
    n_val = int(len(data_pairs) * val_ratio)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    train_pairs = [data_pairs[i] for i in train_indices]
    val_pairs = [data_pairs[i] for i in val_indices]
    test_pairs = [data_pairs[i] for i in test_indices]
    
    return train_pairs, val_pairs, test_pairs

