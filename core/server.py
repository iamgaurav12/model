from typing import Dict, List, Optional, Tuple
import flwr as fl
from flwr.common import Metrics, FitRes
from utils.eco_metrics import aggregate_sustainability_metrics
from config.settings import CARBON_SAVINGS_TARGET, MIN_AVAILABLE_CLIENTS, MIN_FIT_CLIENTS

class EcoFedStrategy(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__(
            min_available_clients=MIN_AVAILABLE_CLIENTS,
            min_fit_clients=MIN_FIT_CLIENTS
        )
        self.energy_history = []
        self.carbon_history = []
        self.accuracy_history = []
        self.best_carbon_savings = 0.0
        
    def aggregate_fit(self, server_round: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]], failures: List):
        # Aggregate weights using default FedAvg
        aggregated_weights, metrics = super().aggregate_fit(server_round, results, failures)
        
        # Aggregate sustainability metrics
        if results:
            fit_results = [r for _, r in results]
            round_energy, round_carbon = aggregate_sustainability_metrics(fit_results)
            self.energy_history.append(round_energy)
            self.carbon_history.append(round_carbon)
            
            # Calculate carbon savings
            if server_round > 1:
                carbon_savings = 1 - (round_carbon / self.carbon_history[0]) if self.carbon_history[0] > 0 else 0
                self.best_carbon_savings = max(self.best_carbon_savings, carbon_savings)
            
        return aggregated_weights, metrics
        
    def aggregate_evaluate(self, server_round: int, results: List, failures: List):
        # Aggregate accuracy metrics
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        
        if aggregated_metrics and "accuracy" in aggregated_metrics:
            self.accuracy_history.append(aggregated_metrics.get("accuracy", 0.0))
            
        return aggregated_loss, aggregated_metrics
        
    def evaluate_global_metrics(self) -> Dict[str, float]:
        """Calculate final sustainability metrics"""
        total_energy = sum(self.energy_history)
        total_carbon = sum(self.carbon_history)
        avg_accuracy = sum(self.accuracy_history) / len(self.accuracy_history) if self.accuracy_history else 0.0
        
        return {
            "total_energy_joules": total_energy,
            "total_carbon_grams": total_carbon,
            "average_accuracy": avg_accuracy,
            "carbon_savings": self.best_carbon_savings,
            "target_achieved": self.best_carbon_savings >= CARBON_SAVINGS_TARGET
        }