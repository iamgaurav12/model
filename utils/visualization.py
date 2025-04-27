import matplotlib.pyplot as plt
from typing import List
import os

def plot_training_metrics(energy_history: List[float], carbon_history: List[float], accuracy_history: List[float]):
    """Plot energy consumption, carbon emissions, and model accuracy over training rounds"""
    plt.figure(figsize=(15, 5))
    
    # Create directory for plots if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    # Energy plot
    plt.subplot(1, 3, 1)
    plt.plot(energy_history, 'b-o')
    plt.title("Energy Consumption per Round")
    plt.xlabel("Round")
    plt.ylabel("Energy (J)")
    
    # Carbon plot
    plt.subplot(1, 3, 2)
    plt.plot(carbon_history, 'r-o')
    plt.title("Carbon Emissions per Round")
    plt.xlabel("Round")
    plt.ylabel("Carbon (gCO2e)")
    
    # Accuracy plot
    plt.subplot(1, 3, 3)
    plt.plot(accuracy_history, 'g-o')
    plt.title("Model Accuracy per Round")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig("plots/training_metrics.png")
    plt.close()