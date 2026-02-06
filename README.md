# Satellite Image Semantic Segmentation

## Overview
This project trains (or fine-tunes) a deep learning model to perform semantic segmentation on satellite images. The goal is to classify each pixel in a satellite image into different categories (e.g., buildings, roads, vegetation, water, etc.).

## Project Structure
```
satellite-segmentation/
├── src/                          # Python source modules
│   ├── data_loader.py             # Data loading & preprocessing
│   ├── models.py                  # Model architectures
│   ├── utils.py                   # Visualization & metrics
│   └── inference.py               # CLI inference script
├── data/                         # Data directory
│   ├── raw/                       # Raw satellite images & masks
│   └── real/                      # DeepGlobe dataset (images & masks)
├── models/                       # Saved trained models (.h5)
├── results/                      # Training results, plots, predictions
├── logs/                         # TensorBoard logs
├── setup_real_data.py            #Setting up real data                         
├── train_resnet_50.py            #Training script           
├── requirements.txt             # Project dependencies
└── README.md                    # This file
```

## Setup Instructions

### 1. Create a Virtual Environment
```bash
cd /Users/rodrigocuadra/MSDS/Practicum/satellite-segmentation

# Using conda (recommended)
conda create -n satellite-seg python=3.10
conda activate satellite-seg

# OR using venv
python -m venv venv
source venv/bin/activate  # On macOS/Linux
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Google Earth Engine Authentication (for leafmap)
If using leafmap to download satellite images:
```bash
earthengine authenticate
```

## Technology Stack
- **Framework**: TensorFlow/Keras
- **Data Source**: Leafmap (Google Earth Engine, Sentinel, Landsat, etc.)
- **Pre-trained Models**: segmentation-models library (U-Net, FPN, DeepLabV3+, etc.)
- **Image Processing**: OpenCV, scikit-image

## Approaches

### Option 1: Fine-tune Pre-trained Model (Recommended)
Use the `segmentation-models` library with pre-trained backbones (ResNet, EfficientNet, etc.) on ImageNet, then fine-tune on satellite imagery.

**Advantages:**
- Faster training
- Better performance with limited data
- Well-tested architectures

### Option 2: Train from Scratch
Define a custom U-Net or similar architecture using TensorFlow/Keras.

**Advantages:**
- Full control over architecture
- Can optimize for specific use case
- Learning opportunity

## Getting Started

### Step 1: Data Exploration
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```
- Download satellite data (Kaggle, leafmap, or use sample data)
- Explore and visualize images
- Create train/validation/test splits
- Generate dataset configuration

### Step 2: Model Training
```bash
jupyter notebook notebooks/02_model_training.ipynb
```
- Load prepared data
- Build U-Net model (or pre-trained model)
- Train with callbacks (early stopping, checkpointing)
- Evaluate on test set
- Save trained model

### Step 3: Inference
Option A - Interactive (Notebook):
```bash
jupyter notebook notebooks/03_inference.ipynb
```

Option B - Command Line (Script):
```bash
# Single image
python src/inference.py --model models/unet_best.h5 --input test.png

# Batch processing
python src/inference.py --model models/unet_best.h5 --input data/images/ --batch
```

## Architecture Overview

- Library modules (`src/`): Reusable building blocks you import.
  - `src/data_loader.py`: Creates TensorFlow datasets (load, resize/normalize, augment, batch, split).
  - `src/models.py`: Model definitions (custom U-Net and improved U-Net builder; compile helpers).
  - `src/utils.py`: Visualization, metrics, plotting, GPU checks.
  - `src/inference.py`: Functions for loading images, predicting, and saving results (also usable as a CLI).

- Application scripts (project root): End-to-end tasks you run.
  - `setup_real_data.py`: Download/organize real datasets (e.g., DeepGlobe) into `data/real/`.
  - `train_resnet50.py`: Orchestrates training using the library (loads data → builds model → trains → saves best model).

Think: `src/` = toolbox; root scripts = recipes that use the tools to complete a job.

## How U-Net Is Defined (Encoder, Bottleneck, Decoder)

The custom U-Net is implemented in `src/models.py` as `unet_model(input_shape, num_classes, filters, dropout_rate)`.

- Encoder (downsampling path):
  - Repeated Conv2D → Conv2D → MaxPool blocks.
  - Each block doubles the filters (e.g., 64 → 128 → 256 → 512) to learn richer features while spatial size halves.
  - Dropout is applied after pooling for regularization.
  - Output feature maps of each block are saved for skip connections.

- Bottleneck (bridge):
  - The deepest part (highest filters, smallest spatial size).
  - Two Conv2D layers operate without further pooling; captures the most abstract, global features.

- Decoder (upsampling path):
  - Repeated Conv2DTranspose (upsample by 2) → Concatenate with corresponding encoder feature map (skip connection) → Conv2D → Conv2D.
  - Mirrors the encoder: halves filters progressively (e.g., 512 → 256 → 128 → 64) while spatial size doubles each step.
  - Skip connections restore spatial detail lost during downsampling.

- Output layer:
  - Final `Conv2D(num_classes, kernel=1, activation='softmax')` maps features to per-pixel class probabilities.

Putting it together: the model first compresses the image into abstract features (encoder), processes them at the smallest scale (bottleneck), then reconstructs a full-resolution segmentation map while fusing high-resolution detail from the encoder (decoder). This balance of “what” (encoder semantics) and “where” (decoder spatial detail via skips) is why U-Net works well for segmentation.

## Common Satellite Image Datasets
- **Sentinel-2**: 10m resolution, multispectral
- **Landsat**: 30m resolution
- **NAIP**: High-resolution aerial imagery (US only)
- **DeepGlobe**: Land cover classification dataset
- **SpaceNet**: Building footprint detection
- **LandCover.ai**: European land cover dataset

## Notes
- Use GPU for faster training (check with `nvidia-smi` or TensorFlow GPU detection)
- Start with small image patches (256x256 or 512x512) for faster iteration

## Resources
- [Leafmap Documentation](https://leafmap.org/)
- [Segmentation Models](https://github.com/qubvel/segmentation_models)
- [TensorFlow Image Segmentation Tutorial](https://www.tensorflow.org/tutorials/images/segmentation)

