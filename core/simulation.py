import flwr as fl
import numpy as np
from typing import Dict
from core.client import IoTClient
from core.server import EcoFedStrategy
from utils.data_gen import generate_synthetic_data
from utils.visualization import plot_training_metrics
from config.settings import NUM_CLIENTS, NUM_ROUNDS

def client_fn(cid: str) -> IoTClient:
    X, y = generate_synthetic_data(num_samples=100)
    return IoTClient(cid, X, y)

def run_simulation() -> Dict:
    strategy = EcoFedStrategy()
    
    # Start the simulation
    hist = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1},
        ray_init_args={"include_dashboard": False}
    )
    
    # Get final metrics
    metrics = strategy.evaluate_global_metrics()
    
    # Plot metrics
    if strategy.energy_history and strategy.carbon_history and strategy.accuracy_history:
        plot_training_metrics(
            strategy.energy_history,
            strategy.carbon_history,
            strategy.accuracy_history
        )
    
    return metrics