"""
Model architectures for satellite image segmentation
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Tuple, Optional


def _patch_keras_utils():
    """
    Patch keras.utils to add generic_utils for compatibility with segmentation-models
    This is needed because segmentation-models depends on an older version of efficientnet
    which expects keras.utils.generic_utils
    """
    try:
        from tensorflow.keras.utils import generic_utils
        if not hasattr(tf.keras.utils, 'generic_utils'):
            tf.keras.utils.generic_utils = generic_utils
            print("✅ Patched keras.utils.generic_utils for segmentation-models compatibility")
    except (ImportError, AttributeError) as e:
        print(f"⚠️  Could not patch generic_utils: {e}")


def _try_import_segmentation_models():
    """
    Try to import segmentation_models, with compatibility fixes
    Returns the module if successful, None otherwise
    """
    try:
        _patch_keras_utils()
        import segmentation_models as sm
        print("✅ Successfully imported segmentation_models")
        return sm
    except Exception as e:
        print(f"⚠️  Failed to import segmentation_models: {e}")
        return None


# Try to import segmentation_models on module load
try:
    sm = _try_import_segmentation_models()
except:
    sm = None


def unet_model(
    input_shape: Tuple[int, int, int] = (256, 256, 3),
    num_classes: int = 5,
    filters: int = 64,
    dropout_rate: float = 0.2
) -> Model:
    """
    U-Net architecture for semantic segmentation
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of segmentation classes
        filters: Number of filters in first conv layer (doubles with each level)
        dropout_rate: Dropout rate for regularization
    
    Returns:
        Keras Model
    """
    inputs = keras.Input(shape=input_shape)
    
    # Encoder (Contracting Path)
    # Block 1
    c1 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    p1 = layers.Dropout(dropout_rate)(p1)
    
    # Block 2
    c2 = layers.Conv2D(filters*2, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(filters*2, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    p2 = layers.Dropout(dropout_rate)(p2)
    
    # Block 3
    c3 = layers.Conv2D(filters*4, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(filters*4, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    p3 = layers.Dropout(dropout_rate)(p3)
    
    # Block 4
    c4 = layers.Conv2D(filters*8, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(filters*8, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    p4 = layers.Dropout(dropout_rate)(p4)
    
    # Bottleneck
    c5 = layers.Conv2D(filters*16, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(filters*16, (3, 3), activation='relu', padding='same')(c5)
    
    # Decoder (Expansive Path)
    # Block 6
    u6 = layers.Conv2DTranspose(filters*8, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    u6 = layers.Dropout(dropout_rate)(u6)
    c6 = layers.Conv2D(filters*8, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(filters*8, (3, 3), activation='relu', padding='same')(c6)
    
    # Block 7
    u7 = layers.Conv2DTranspose(filters*4, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    u7 = layers.Dropout(dropout_rate)(u7)
    c7 = layers.Conv2D(filters*4, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(filters*4, (3, 3), activation='relu', padding='same')(c7)
    
    # Block 8
    u8 = layers.Conv2DTranspose(filters*2, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    u8 = layers.Dropout(dropout_rate)(u8)
    c8 = layers.Conv2D(filters*2, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(filters*2, (3, 3), activation='relu', padding='same')(c8)
    
    # Block 9
    u9 = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    u9 = layers.Dropout(dropout_rate)(u9)
    c9 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(c9)
    
    # Output layer
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs], name='U-Net')
    return model


def get_pretrained_segmentation_model(
    backbone: str = 'resnet50',
    input_shape: Tuple[int, int, int] = (256, 256, 3),
    num_classes: int = 7,
    architecture: str = 'Unet'
) -> Model:
    """
    Get a pre-trained U-Net model with ResNet50 backbone (ImageNet weights)
    Builds U-Net from scratch using pre-trained ResNet50 as encoder
    
    Args:
        backbone: Backbone architecture (resnet50 - ResNet34 not available in keras.applications)
        input_shape: Input image shape
        num_classes: Number of classes
        architecture: Model architecture (Unet)
    
    Returns:
        Keras Model with pre-trained ResNet50 encoder and U-Net decoder
    """
    import os
    import ssl
    
    # Disable SSL verification for downloading weights
    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    print(f"Creating {architecture} with {backbone} backbone (pre-trained on ImageNet)")
    
    try:
        # Get pre-trained ResNet50 encoder
        encoder = get_pretrained_resnet50_encoder(input_shape)
        
        if encoder is None:
            print("⚠️  Using fallback custom U-Net")
            return _fallback_unet_model(input_shape, num_classes)
        
        # Create input layer
        inputs = keras.Input(shape=input_shape)
        
        # Pass input through pre-trained encoder
        x = encoder(inputs)  # This produces 8x8x2048 features
        
        # Now build decoder (symmetric to encoder)
        # The encoder goes: 256x256 -> 128x128 -> 64x64 -> 32x32 -> 16x16 -> 8x8
        # The decoder goes: 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256
        
        # Decoder Block 1: 8x8 -> 16x16
        x = layers.Conv2DTranspose(1024, (2, 2), strides=(2, 2), padding='same')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Decoder Block 2: 16x16 -> 32x32
        x = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Decoder Block 3: 32x32 -> 64x64
        x = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Decoder Block 4: 64x64 -> 128x128
        x = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Decoder Block 5: 128x128 -> 256x256
        x = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Output layer
        outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(x)
        
        # Create final model
        model = Model(inputs=inputs, outputs=outputs, name='ResNet50_UNet')
        
        print(f"✅ Successfully created {architecture} with {backbone} backbone")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        print(f"  Total parameters: {model.count_params():,}")
        print(f"  Pre-trained backbone: ResNet50 (ImageNet)")
        print(f"  Encoder params: {encoder.count_params():,}")
        print(f"  Decoder params: {model.count_params() - encoder.count_params():,}")
        
        return model
        
    except Exception as e:
        print(f"⚠️  Error creating {architecture} model: {e}")
        print(f"  Falling back to custom U-Net")
        import traceback
        traceback.print_exc()
        return _fallback_unet_model(input_shape, num_classes)


def get_pretrained_resnet50_encoder(input_shape: Tuple[int, int, int] = (256, 256, 3)) -> Model:
    """
    Get a pre-trained ResNet50 encoder using TensorFlow's ResNet implementation
    with ImageNet weights
    
    Note: ResNet34 is not available in keras.applications, so we use ResNet50
    which has the same architecture family and is more powerful.
    
    Args:
        input_shape: Input image shape (height, width, channels)
    
    Returns:
        Pre-trained ResNet50 model that acts as an encoder
    """
    print("Loading pre-trained ResNet50 from TensorFlow (ImageNet weights)...")
    
    try:
        # ResNet50 is available in keras.applications (ResNet34 is not)
        base_model = keras.applications.ResNet50(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        print("✅ Successfully loaded pre-trained ResNet50 (ImageNet)")
        print(f"   Total params: {base_model.count_params():,}")
        
        # Don't freeze weights - allow fine-tuning
        base_model.trainable = True
        
        return base_model
        
    except Exception as e:
        print(f"⚠️  Could not load pre-trained ResNet: {e}")
        print("   Will use random initialization instead")
        return None


def _fallback_unet_model(
    input_shape: Tuple[int, int, int] = (256, 256, 3),
    num_classes: int = 7
) -> Model:
    """
    Fallback U-Net model if segmentation-models is not available
    This is the improved U-Net we built from scratch
    """
    print("Creating improved U-Net model (no pre-trained weights - fallback)")
    
    inputs = keras.Input(shape=input_shape)
    
    # Encoder with residual connections (ResNet-like)
    # Block 1
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    p1 = layers.Dropout(0.1)(p1)
    
    # Block 2
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    p2 = layers.Dropout(0.1)(p2)
    
    # Block 3
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    c3 = layers.BatchNormalization()(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    p3 = layers.Dropout(0.2)(p3)
    
    # Block 4
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    c4 = layers.BatchNormalization()(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    p4 = layers.Dropout(0.2)(p4)
    
    # Bottleneck
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Dropout(0.3)(c5)
    
    # Decoder with skip connections
    # Block 6
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    u6 = layers.Dropout(0.2)(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
    c6 = layers.BatchNormalization()(c6)
    
    # Block 7
    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    u7 = layers.Dropout(0.2)(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
    c7 = layers.BatchNormalization()(c7)
    
    # Block 8
    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    u8 = layers.Dropout(0.1)(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
    c8 = layers.BatchNormalization()(c8)
    
    # Block 9
    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    u9 = layers.Dropout(0.1)(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    c9 = layers.BatchNormalization()(c9)
    
    # Output layer
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c9)
    
    model = Model(inputs=inputs, outputs=outputs, name='Fallback_Unet')
    
    return model


def compile_segmentation_model(
    model: Model,
    num_classes: int,
    learning_rate: float = 1e-4,
    loss_type: str = 'categorical_crossentropy'
) -> Model:
    """
    Compile a segmentation model with appropriate loss and metrics
    
    Args:
        model: Keras model to compile
        num_classes: Number of classes
        learning_rate: Learning rate for optimizer
        loss_type: Loss function type
    
    Returns:
        Compiled model
    """
    # Define metrics
    metrics = [
        'accuracy',
    ]
    
    # Define optimizer
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Define loss
    if loss_type == 'categorical_crossentropy':
        loss = keras.losses.CategoricalCrossentropy()
    elif loss_type == 'sparse_categorical_crossentropy':
        loss = keras.losses.SparseCategoricalCrossentropy()
    elif loss_type == 'dice':
        loss = dice_loss
    elif loss_type == 'focal':
        loss = focal_loss
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Dice coefficient for segmentation evaluation
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice coefficient
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
    )


def dice_loss(y_true, y_pred):
    """Dice loss for training"""
    return 1 - dice_coefficient(y_true, y_pred)


def focal_loss(gamma=2., alpha=0.25):
    """
    Focal loss for handling class imbalance
    
    Args:
        gamma: Focusing parameter
        alpha: Balancing parameter
    
    Returns:
        Loss function
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.keras.backend.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.keras.backend.log(y_pred)
        weight = alpha * y_true * tf.keras.backend.pow((1 - y_pred), gamma)
        loss = weight * cross_entropy
        return tf.keras.backend.sum(loss, axis=-1)
    
    return focal_loss_fixed


class SegmentationMetrics(keras.callbacks.Callback):
    """Custom callback to track segmentation metrics during training"""
    
    def __init__(self, validation_data, num_classes):
        super().__init__()
        self.validation_data = validation_data
        self.num_classes = num_classes
        
    def on_epoch_end(self, epoch, logs=None):
        """Calculate and log metrics at end of each epoch"""
        logs = logs or {}
        
        # You can add custom metric calculations here
        # For now, we'll just use the built-in metrics
        pass

