"""Neural network architecture definitions for car sound classification.

Defines three architectures from model_architecture_spec.md:
  - M2: Compact 2-D CNN (mel-spectrogram input)
  - M5: 1-D CNN (MFCC input)
  - M6: Depthwise Separable CNN (mel-spectrogram input, 32-channel variant)

These builders return *uncompiled* models so the training script can
configure the optimizer and learning rate independently.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ─── M2: Compact 2-D CNN ──────────────────────────────────────────────────


def build_compact_cnn(
    input_shape: tuple = (40, 92, 1),
    num_classes: int = 6,
    dropout_rate: float = 0.3,
) -> keras.Model:
    """Build the compact 2-D CNN for mel-spectrogram classification (M2).

    Architecture (from spec §3.5):
        Conv2D(16, 3×3) → BN → ReLU → MaxPool(2,2)
        Conv2D(32, 3×3) → BN → ReLU → MaxPool(2,2)
        Conv2D(32, 3×3) → BN → ReLU → GlobalAvgPool2D
        Dropout → Dense(N, softmax)

    Args:
        input_shape: Input tensor shape, default (40, 92, 1).
        num_classes: Number of output classes.
        dropout_rate: Dropout probability before the classifier head.

    Returns:
        Uncompiled Keras model.
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        # Block 1
        layers.Conv2D(16, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPool2D((2, 2)),

        # Block 2
        layers.Conv2D(32, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPool2D((2, 2)),

        # Block 3
        layers.Conv2D(32, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.GlobalAveragePooling2D(),

        # Classifier
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model


# ─── M5: 1-D CNN ──────────────────────────────────────────────────────────


def build_1d_cnn(
    input_shape: tuple = (92, 13),
    num_classes: int = 6,
    dropout_rate: float = 0.3,
) -> keras.Model:
    """Build the 1-D CNN for MFCC classification (M5).

    Architecture (from spec §4.6):
        Conv1D(16, 5) → BN → ReLU → MaxPool(2)
        Conv1D(32, 3) → BN → ReLU → MaxPool(2)
        Conv1D(32, 3) → BN → ReLU → GlobalAvgPool1D
        Dropout → Dense(N, softmax)

    Args:
        input_shape: Input tensor shape, default (92, 13).
        num_classes: Number of output classes.
        dropout_rate: Dropout probability before the classifier head.

    Returns:
        Uncompiled Keras model.
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        # Block 1
        layers.Conv1D(16, 5, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPool1D(2),

        # Block 2
        layers.Conv1D(32, 3, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPool1D(2),

        # Block 3
        layers.Conv1D(32, 3, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.GlobalAveragePooling1D(),

        # Classifier
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model


# ─── M6: Depthwise Separable CNN (32-channel variant) ─────────────────────


def build_ds_cnn(
    input_shape: tuple = (40, 92, 1),
    num_classes: int = 6,
    channels: int = 32,
    num_ds_blocks: int = 4,
) -> keras.Model:
    """Build the DS-CNN for mel-spectrogram classification (M6, 32-ch).

    Architecture (from spec §5.7, revised for SRAM budget):
        Conv2D(channels, 4×10, stride=2) → BN → ReLU
        N × DS Block:
            DepthwiseConv2D(3×3) → BN → ReLU
            Conv2D(channels, 1×1) → BN → ReLU
        GlobalAvgPool2D → Dense(N, softmax)

    Args:
        input_shape: Input tensor shape, default (40, 92, 1).
        num_classes: Number of output classes.
        channels: Channel count throughout the network.
        num_ds_blocks: Number of depthwise separable blocks.

    Returns:
        Uncompiled Keras model (Functional API).
    """
    inputs = layers.Input(shape=input_shape)

    # Initial convolution
    x = layers.Conv2D(channels, (4, 10), strides=(2, 2), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # DS blocks
    for _ in range(num_ds_blocks):
        x = layers.DepthwiseConv2D((3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(channels, (1, 1))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    # Classifier
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)


# ─── Model config registry ────────────────────────────────────────────────

MODEL_CONFIGS = {
    "m2": {
        "builder": build_compact_cnn,
        "input_shape": (40, 92, 1),
        "feature_type": "mel",
        "model_dir": "m2_cnn2d_float32",
        "display_name": "2-D CNN (M2)",
    },
    "m5": {
        "builder": build_1d_cnn,
        "input_shape": (92, 13),
        "feature_type": "mfcc",
        "model_dir": "m5_cnn1d_int8_qat",
        "display_name": "1-D CNN (M5)",
    },
    "m6": {
        "builder": build_ds_cnn,
        "input_shape": (40, 92, 1),
        "feature_type": "mel",
        "model_dir": "m6_dscnn_int8_qat",
        "display_name": "DS-CNN (M6)",
    },
}


def get_model_config(model_name: str) -> dict:
    """Return configuration dict for a model architecture.

    Args:
        model_name: One of ``"m2"``, ``"m5"``, ``"m6"``.

    Returns:
        Dict with keys: builder, input_shape, feature_type, model_dir,
        display_name.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {list(MODEL_CONFIGS)}"
        )
    return MODEL_CONFIGS[model_name]


# ─── Sanity check ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Model architecture sanity check")
    print("=" * 60)

    for name, cfg in MODEL_CONFIGS.items():
        for tier, n_classes in [(1, 2), (2, 6), (3, 9)]:
            kwargs = {"input_shape": cfg["input_shape"], "num_classes": n_classes}
            model = cfg["builder"](**kwargs)
            n_params = model.count_params()
            print(f"  {cfg['display_name']:>15s}  Tier {tier} ({n_classes}-class):  "
                  f"{n_params:>7,d} params")
        print()

    print("All models built successfully.")
