# Federated Learning
NUM_CLIENTS = 10
NUM_ROUNDS = 5
MIN_AVAILABLE_CLIENTS = 3
MIN_FIT_CLIENTS = 3

# Model Architecture
MODEL_INPUT_SHAPE = (10,)
MODEL_OUTPUT_CLASSES = 3
MODEL_SAVE_PATH = "models/fl_protonet.tflite"
QUANTIZATION_ENABLED = False  # Disable for debugging

# Device Parameters
DEVICE_POWER_W = 5.0
DEVICE_TYPES = {
    "esp32": {"power": 3.3, "ram": 512},
    "raspberry_pi": {"power": 5.0, "ram": 4096}
}

# Energy/Carbon
CARBON_INTENSITY = 420  # gCO2/kWh
CARBON_SAVINGS_TARGET = 0.56

# Windows-specific fixes
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'