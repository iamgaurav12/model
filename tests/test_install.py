import tensorflow as tf
import flwr as fl

def test_installation():
    print(f"TensorFlow {tf.__version__}")
    print(f"Flower {fl.__version__}")
    print("Num GPUs:", len(tf.config.list_physical_devices('GPU')))

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_shape=(10,))
    ])
    print("Model built successfully!")
    return True

if __name__ == "__main__":
    test_installation()