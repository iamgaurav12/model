import matplotlib.pyplot as plt

def plot_training_metrics(energy_history, carbon_history, accuracy_history):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(energy_history, 'b-o')
    plt.title("Energy Consumption per Round")
    
    plt.subplot(1, 3, 2)
    plt.plot(carbon_history, 'r-o')
    plt.title("Carbon Emissions per Round")
    
    plt.subplot(1, 3, 3)
    plt.plot(accuracy_history, 'g-o')
    plt.title("Model Accuracy per Round")
    
    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.close()