import numpy as np
import flwr as fl
import time
from core.model import FLProtoNet
from utils.eco_metrics import calculate_energy, calculate_carbon
from config.settings import DEVICE_POWER_W, CARBON_INTENSITY

class IoTClient(fl.client.NumPyClient):
    def __init__(self, cid: str, X_train: np.ndarray, y_train: np.ndarray):
        self.cid = cid
        self.X_train = X_train
        self.y_train = y_train
        self.model = FLProtoNet().build_model()
        self.device_type = self._detect_device()
        
    def _detect_device(self):
        return "esp32" if np.random.random() > 0.3 else "raspberry_pi"
    
    def get_parameters(self, config):
        return self.model.get_weights()
    
    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        start_time = time.time()
        history = self.model.fit(
            self.X_train, 
            self.y_train,
            epochs=1,
            batch_size=32,
            verbose=0
        )
        energy = calculate_energy(DEVICE_POWER_W, time.time() - start_time)
        return (
            self.model.get_weights(),
            len(self.X_train),
            {
                "energy": energy,
                "carbon": calculate_carbon(energy, CARBON_INTENSITY),
                "accuracy": history.history['accuracy'][0],
                "device_type": self.device_type
            }
        )
    
    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.X_train, self.y_train, verbose=0)
        return loss, len(self.X_train), {"accuracy": accuracy}