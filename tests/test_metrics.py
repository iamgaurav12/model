import unittest
from utils.eco_metrics import calculate_energy, calculate_carbon

class TestEcoMetrics(unittest.TestCase):
    def test_energy_calculation(self):
        energy = calculate_energy(5.0, 10.0)  # 5W for 10 seconds
        self.assertEqual(energy, 50.0)
        
    def test_carbon_calculation(self):
        carbon = calculate_carbon(3600000)  # 1 kWh at default intensity
        self.assertAlmostEqual(carbon, 420.0)  # 420gCO2/kWh
        
    def test_carbon_custom_intensity(self):
        carbon = calculate_carbon(3600000, carbon_intensity=200)  # 1 kWh at 200g/kWh
        self.assertEqual(carbon, 200.0)

if __name__ == "__main__":
    unittest.main()