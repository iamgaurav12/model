# FedCom++ Simulation

## Installation Guide

Follow these steps to set up and run the FedCom++ simulation:

### Prerequisites
- Python 3.8 or later
- Virtual environment tool (e.g., `venv`)

### Steps

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Set Up a Virtual Environment**
   On Windows:
   ```bash
   python -m venv fedcompp_env
   cd fedcompp_env\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   Run the installation test script:
   ```bash
   python tests/test_install.py
   ```

5. **Run the Simulation**
   Execute the main script:
   ```bash
   python main.py
   ```

### Output
- Results will be saved in `results.json`.
- Training metrics visualization will be saved in `plots/training_metrics.png`.

### Notes
- Ensure TensorFlow is installed with CPU support if GPU is not available.
- Modify `config/settings.py` to adjust simulation parameters as needed.
