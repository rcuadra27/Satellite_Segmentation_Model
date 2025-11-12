"""
Inference script for satellite image segmentation

Usage:
    python inference.py --model models/unet_best.h5 --input data/test_image.png
    python inference.py --model models/unet_best.h5 --input data/test_images/ --batch
"""

import argparse
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime


def load_and_preprocess_image(image_path, target_size):
    """Load and preprocess an image for prediction"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image.copy()
    
    # Resize to model input size
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    
    return original_image, image


def predict_segmentation(model, image):
    """Generate segmentation prediction"""
    prediction_probs = model.predict(image, verbose=0)
    prediction_mask = np.argmax(prediction_probs[0], axis=-1)
    return prediction_mask, prediction_probs[0]


def save_prediction(original_image, prediction_mask, output_path, colormap='tab10'):
    """Save prediction mask and overlay"""
    # Resize mask to original size
    mask_resized = cv2.resize(
        prediction_mask.astype(np.uint8),
        (original_image.shape[1], original_image.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save mask
    mask_output = output_path.parent / f"{output_path.stem}_mask.png"
    cv2.imwrite(str(mask_output), mask_resized)
    
    # Create and save overlay
    cmap = plt.get_cmap(colormap)
    num_classes = prediction_mask.max() + 1
    colored_mask = cmap(mask_resized / num_classes)[:, :, :3]
    overlay = (original_image / 255.0) * 0.6 + colored_mask * 0.4
    overlay_rgb = (overlay * 255).astype(np.uint8)
    
    overlay_output = output_path.parent / f"{output_path.stem}_overlay.png"
    cv2.imwrite(str(overlay_output), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
    
    return mask_output, overlay_output


def process_single_image(model_path, image_path, output_dir=None):
    """Process a single image"""
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(str(model_path))
    
    input_shape = model.input_shape[1:3]
    num_classes = model.output_shape[-1]
    
    print(f"Model info:")
    print(f"  Input size: {input_shape}")
    print(f"  Number of classes: {num_classes}")
    
    print(f"\nProcessing image: {image_path}")
    original_img, preprocessed_img = load_and_preprocess_image(image_path, input_shape)
    
    print("Generating prediction...")
    pred_mask, pred_probs = predict_segmentation(model, preprocessed_img)
    
    # Set output directory
    if output_dir is None:
        output_dir = Path('results') / f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    image_name = Path(image_path).stem
    output_path = output_dir / f"{image_name}_prediction.png"
    
    mask_path, overlay_path = save_prediction(original_img, pred_mask, output_path)
    
    print(f"\n✓ Prediction saved:")
    print(f"  Mask: {mask_path}")
    print(f"  Overlay: {overlay_path}")
    
    # Print class distribution
    unique, counts = np.unique(pred_mask, return_counts=True)
    total = pred_mask.size
    print(f"\nClass distribution:")
    for class_id, count in zip(unique, counts):
        percentage = (count / total) * 100
        print(f"  Class {class_id}: {percentage:.2f}%")
    
    return pred_mask, pred_probs


def process_batch(model_path, input_dir, output_dir=None):
    """Process multiple images from a directory"""
    input_dir = Path(input_dir)
    
    # Find all images
    image_files = (
        list(input_dir.glob('*.png')) +
        list(input_dir.glob('*.jpg')) +
        list(input_dir.glob('*.jpeg')) +
        list(input_dir.glob('*.tif'))
    )
    
    if len(image_files) == 0:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Load model once
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(str(model_path))
    
    input_shape = model.input_shape[1:3]
    num_classes = model.output_shape[-1]
    
    print(f"Model info:")
    print(f"  Input size: {input_shape}")
    print(f"  Number of classes: {num_classes}")
    
    # Set output directory
    if output_dir is None:
        output_dir = Path('results') / f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    print(f"\nProcessing {len(image_files)} images...")
    
    for i, image_path in enumerate(image_files):
        try:
            print(f"[{i+1}/{len(image_files)}] {image_path.name}...")
            
            original_img, preprocessed_img = load_and_preprocess_image(image_path, input_shape)
            pred_mask, pred_probs = predict_segmentation(model, preprocessed_img)
            
            output_path = output_dir / f"{image_path.stem}_prediction.png"
            save_prediction(original_img, pred_mask, output_path)
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print(f"\n✓ Batch processing complete!")
    print(f"  Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Satellite image segmentation inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python inference.py --model models/unet_best.h5 --input test.png
  
  # Batch processing
  python inference.py --model models/unet_best.h5 --input data/images/ --batch
  
  # Specify output directory
  python inference.py --model models/unet_best.h5 --input test.png --output results/my_predictions/
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to trained model (.h5 file)'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input image or directory'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory for predictions (default: results/predictions_<timestamp>/)'
    )
    
    parser.add_argument(
        '--batch', '-b',
        action='store_true',
        help='Process all images in input directory'
    )
    
    args = parser.parse_args()
    
    # Validate model path
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return
    
    # Validate input path
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path not found: {input_path}")
        return
    
    # Process
    if args.batch or input_path.is_dir():
        process_batch(model_path, input_path, args.output)
    else:
        process_single_image(model_path, input_path, args.output)


if __name__ == '__main__':
    main()

