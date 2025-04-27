import numpy as np
import tensorflow as tf
from config.settings import MODEL_INPUT_SHAPE, MODEL_OUTPUT_CLASSES

def generate_synthetic_data(num_samples=1000):
    """Generate synthetic data for training and testing"""
    X = np.random.rand(num_samples, MODEL_INPUT_SHAPE[0]) * np.array([
        100, 500, 1.0, 1000, 5, 50, 0.9, 10, 3.3, 1.0
    ])
    y = np.random.randint(0, MODEL_OUTPUT_CLASSES, size=(num_samples,))
    return X.astype(np.float32), tf.keras.utils.to_categorical(y, num_classes=MODEL_OUTPUT_CLASSES)