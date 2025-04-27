import unittest
import numpy as np
from core.client import IoTClient
from utils.data_gen import generate_synthetic_data

class TestIoTClient(unittest.TestCase):
    def setUp(self):
        X, y = generate_synthetic_data(num_samples=100)
        self.client = IoTClient("test_client", X, y)
        
    def test_fit_returns_metrics(self):
        initial_weights = self.client.get_parameters({})
        new_weights, num_samples, metrics = self.client.fit(initial_weights, {})
        
        self.assertIsInstance(new_weights, list)
        self.assertEqual(num_samples, 100)
        self.assertIn("energy", metrics)
        self.assertIn("carbon", metrics)
        
    def test_evaluate_returns_accuracy(self):
        initial_weights = self.client.get_parameters({})
        loss, num_samples, metrics = self.client.evaluate(initial_weights, {})
        
        self.assertIsInstance(loss, float)
        self.assertEqual(num_samples, 100)
        self.assertIn("accuracy", metrics)

if __name__ == "__main__":
    unittest.main()