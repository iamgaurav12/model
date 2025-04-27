import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU if not needed

import json
import sys
import tensorflow as tf
from core.simulation import run_simulation

def main():
    print("Starting FedCom++ Simulation...")
    print(f"TensorFlow version: {tf.__version__}")
    
    # Create directories if they don't exist
    os.makedirs("models", exist_ok=True)
    
    try:
        results = run_simulation()
        
        print("\n=== Final Results ===")
        print(json.dumps(results, indent=2))
        
        with open("results.json", "w") as f:
            json.dump(results, f)
        
        print("Simulation complete. Results saved to results.json")
        print("Check plots/training_metrics.png for visualization")
        return 0
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())