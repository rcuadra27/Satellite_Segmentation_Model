"""
Utility functions for satellite segmentation project
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from typing import List, Tuple, Optional
import cv2


def visualize_prediction(
    image: np.ndarray,
    true_mask: Optional[np.ndarray] = None,
    pred_mask: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> None:
    """
    Visualize an image with its true and predicted segmentation masks
    
    Args:
        image: Input image (H, W, C)
        true_mask: Ground truth mask (H, W) or (H, W, num_classes)
        pred_mask: Predicted mask (H, W) or (H, W, num_classes)
        class_names: List of class names for legend
        figsize: Figure size
    """
    # Convert one-hot to class indices if needed
    if true_mask is not None and len(true_mask.shape) == 3:
        true_mask = np.argmax(true_mask, axis=-1)
    if pred_mask is not None and len(pred_mask.shape) == 3:
        pred_mask = np.argmax(pred_mask, axis=-1)
    
    # Determine number of subplots
    n_plots = 1 + (true_mask is not None) + (pred_mask is not None)
    
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    
    # Plot original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    plot_idx = 1
    
    # Plot true mask
    if true_mask is not None:
        im = axes[plot_idx].imshow(true_mask, cmap='tab20')
        axes[plot_idx].set_title('Ground Truth Mask')
        axes[plot_idx].axis('off')
        plot_idx += 1
    
    # Plot predicted mask
    if pred_mask is not None:
        axes[plot_idx].imshow(pred_mask, cmap='tab20')
        axes[plot_idx].set_title('Predicted Mask')
        axes[plot_idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    cmap: str = 'tab20'
) -> np.ndarray:
    """
    Create an overlay of the segmentation mask on the original image
    
    Args:
        image: Original image (H, W, 3)
        mask: Segmentation mask (H, W)
        alpha: Transparency of overlay (0-1)
        cmap: Colormap for mask
    
    Returns:
        Overlayed image
    """
    # Normalize image to [0, 1] if needed
    if image.max() > 1:
        image = image / 255.0
    
    # Create colored mask
    cmap_func = plt.get_cmap(cmap)
    num_classes = mask.max() + 1
    colored_mask = cmap_func(mask / num_classes)[:, :, :3]
    
    # Create overlay
    overlay = image * (1 - alpha) + colored_mask * alpha
    return overlay


def plot_training_history(
    history: tf.keras.callbacks.History,
    metrics: List[str] = ['loss', 'accuracy', 'mean_iou'],
    save_path: Optional[str] = None
) -> None:
    """
    Plot training history
    
    Args:
        history: Training history from model.fit()
        metrics: Metrics to plot
        save_path: Optional path to save figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 4))
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        if metric in history.history:
            axes[idx].plot(history.history[metric], label=f'Training {metric}')
            
            val_metric = f'val_{metric}'
            if val_metric in history.history:
                axes[idx].plot(history.history[val_metric], label=f'Validation {metric}')
            
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel(metric.replace('_', ' ').title())
            axes[idx].set_title(f'{metric.replace("_", " ").title()} over Epochs')
            axes[idx].legend()
            axes[idx].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history plot to {save_path}")
    
    plt.show()


def calculate_class_weights(masks: np.ndarray, num_classes: int) -> dict:
    """
    Calculate class weights for handling class imbalance
    
    Args:
        masks: Array of segmentation masks (N, H, W)
        num_classes: Number of classes
    
    Returns:
        Dictionary of class weights
    """
    # Flatten all masks
    flat_masks = masks.flatten()
    
    # Count pixels per class
    class_counts = np.bincount(flat_masks, minlength=num_classes)
    
    # Calculate weights (inverse frequency)
    total_pixels = len(flat_masks)
    class_weights = {}
    
    for i in range(num_classes):
        if class_counts[i] > 0:
            class_weights[i] = total_pixels / (num_classes * class_counts[i])
        else:
            class_weights[i] = 0.0
    
    return class_weights


def save_predictions(
    images: np.ndarray,
    predictions: np.ndarray,
    save_dir: str,
    prefix: str = 'pred'
) -> None:
    """
    Save predicted masks as images
    
    Args:
        images: Original images
        predictions: Predicted masks
        save_dir: Directory to save predictions
        prefix: Filename prefix
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, (img, pred) in enumerate(zip(images, predictions)):
        # Convert prediction to class indices if one-hot
        if len(pred.shape) == 3:
            pred = np.argmax(pred, axis=-1)
        
        # Save mask
        mask_path = save_dir / f'{prefix}_mask_{idx:04d}.png'
        cv2.imwrite(str(mask_path), pred.astype(np.uint8))
        
        # Save overlay
        overlay = visualize_overlay(img, pred)
        overlay_path = save_dir / f'{prefix}_overlay_{idx:04d}.png'
        overlay_rgb = (overlay * 255).astype(np.uint8)
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))


def calculate_iou(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    """
    Calculate Intersection over Union (IoU) for each class
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        num_classes: Number of classes
    
    Returns:
        Dictionary with IoU per class and mean IoU
    """
    iou_per_class = {}
    
    for class_id in range(num_classes):
        true_class = (y_true == class_id)
        pred_class = (y_pred == class_id)
        
        intersection = np.logical_and(true_class, pred_class).sum()
        union = np.logical_or(true_class, pred_class).sum()
        
        if union > 0:
            iou_per_class[class_id] = intersection / union
        else:
            iou_per_class[class_id] = float('nan')
    
    # Calculate mean IoU (ignoring NaN values)
    valid_ious = [iou for iou in iou_per_class.values() if not np.isnan(iou)]
    mean_iou = np.mean(valid_ious) if valid_ious else 0.0
    
    return {
        'per_class': iou_per_class,
        'mean': mean_iou
    }


def create_color_palette(num_classes: int) -> List[Tuple[int, int, int]]:
    """
    Create a color palette for visualization
    
    Args:
        num_classes: Number of classes
    
    Returns:
        List of RGB colors
    """
    cmap = plt.get_cmap('tab20')
    colors = []
    
    for i in range(num_classes):
        rgb = cmap(i / num_classes)[:3]
        rgb_int = tuple(int(c * 255) for c in rgb)
        colors.append(rgb_int)
    
    return colors


def check_gpu_availability() -> None:
    """Check if GPU is available for TensorFlow"""
    print("TensorFlow version:", tf.__version__)
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ GPU available: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"  - {gpu}")
    else:
        print("✗ No GPU detected. Training will use CPU.")
    
    print(f"✓ Built with CUDA: {tf.test.is_built_with_cuda()}")


def get_model_summary_string(model: tf.keras.Model) -> str:
    """
    Get model summary as a string
    
    Args:
        model: Keras model
    
    Returns:
        Model summary as string
    """
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    return '\n'.join(stringlist)

