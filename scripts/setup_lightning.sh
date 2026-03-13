#!/bin/bash
# Lightning.ai Studio Setup for ORBIT Validation
# Run this in a Lightning.ai Studio with GPU enabled
#
# Lightning.ai offers free A10G GPU credits for open-source projects.
# To use: create a Studio, open a terminal, and run:
#   curl -sL https://raw.githubusercontent.com/Rahillasne/Orbit/main/scripts/setup_lightning.sh | bash
set -euo pipefail

echo "=== ORBIT Lightning.ai Setup ==="

# Install dependencies
echo "[1/4] Installing dependencies..."
pip install -q orbit-robotics[full] lerobot torch torchvision matplotlib pandas

# Clone the repo for development mode
echo "[2/4] Cloning ORBIT repository..."
if [ ! -d "Orbit" ]; then
    git clone https://github.com/Rahillasne/Orbit.git
fi
cd Orbit
pip install -q -e ".[full]"

# Verify GPU
echo "[3/4] Verifying GPU..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'  GPU:    {torch.cuda.get_device_name(0)}')
    print(f'  Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('  WARNING: No GPU detected — running on CPU (will be slower)')
"

# Run the full validation suite
echo "[4/4] Running validation suite..."
python scripts/run_lerobot_validation.py --output results/

echo ""
echo "=== Done! Results saved to results/ ==="
ls -lh results/
