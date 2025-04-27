from config.settings import CARBON_INTENSITY
from typing import List, Tuple

def calculate_energy(power_w: float, time_s: float) -> float:
    """Calculate energy consumption in Joules"""
    return power_w * time_s

def calculate_carbon(energy_j: float, carbon_intensity=CARBON_INTENSITY) -> float:
    """Calculate carbon emissions in grams CO2e"""
    # Convert Joules to kWh: 1 kWh = 3,600,000 J
    return (energy_j / 3_600_000) * carbon_intensity

def aggregate_sustainability_metrics(results) -> Tuple[float, float]:
    """Aggregate energy and carbon metrics from client results"""
    total_energy = sum(res.metrics.get("energy", 0.0) for res in results if res.metrics)
    total_carbon = sum(res.metrics.get("carbon", 0.0) for res in results if res.metrics)
    return total_energy, total_carbon