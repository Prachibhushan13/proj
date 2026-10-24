from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Model, layers

from app.core.config import settings


def build_cnn_lstm_model(
    sequence_length: int = settings.sequence_length,
    image_size: int = settings.image_size,
    train_backbone: bool = True,
) -> Model:
    cached_weights = Path.home() / ".keras" / "models" / "efficientnetb0_notop.h5"
    base_cnn = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=None if cached_weights.is_file() else "imagenet",
        input_shape=(image_size, image_size, 3),
        pooling="avg",
    )
    if cached_weights.is_file():
        # Keras' downloader may reject an older but load-compatible cached
        # checkpoint when its filename hash differs. Load its HDF5 weights
        # directly; this is still the official ImageNet EfficientNetB0 base.
        base_cnn.load_weights(str(cached_weights))
    base_cnn.trainable = False
    if train_backbone:
        for layer in base_cnn.layers[-20:]:
            layer.trainable = True

    inputs = layers.Input(shape=(sequence_length, image_size, image_size, 3))
    # Face crops from preprocessing are RGB float images in [0, 1]. Keras
    # EfficientNet includes an internal 1/255 rescaling layer and expects
    # [0, 255], so restore that range inside the model. The exported model
    # therefore keeps the same [0, 1] API input contract as training.
    model_input = layers.TimeDistributed(layers.Rescaling(255.0))(inputs)
    encoded = layers.TimeDistributed(base_cnn)(model_input)
    encoded = layers.TimeDistributed(layers.Dense(256, activation="relu"))(encoded)
    encoded = layers.Dropout(0.3)(encoded)
    temporal = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(encoded)
    temporal = layers.Bidirectional(layers.LSTM(64))(temporal)
    dense = layers.Dense(128, activation="relu")(temporal)
    dense = layers.Dropout(0.4)(dense)
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32")(dense)

    model = Model(inputs=inputs, outputs=outputs, name="hybrid_cnn_lstm")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="roc_auc"),
        ],
    )
    return model
def get_forensic_backbone() -> Model:
    """
    Returns a pre-trained MobileNetV2 model for feature extraction.
    Used for deep forensic analysis when the primary CNN-LSTM artifact is missing.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(settings.image_size, settings.image_size, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    return base_model

def build_xception_model(image_size: int = 299) -> Model:
    """
    Builds the XceptionNet baseline deepfake classification model (FaceForensics++ approach).
    """
    base_model = tf.keras.applications.Xception(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
        pooling="avg",
    )
    base_model.trainable = False
    
    inputs = layers.Input(shape=(image_size, image_size, 3))
    x = base_model(inputs)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="xception_baseline")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model
