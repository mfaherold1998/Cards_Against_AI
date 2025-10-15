#!/bin/bash

VENV_NAME="venv_cpu"

echo "--- 1. Creating virtual enviroment ($VENV_NAME) ---"
python -m venv "$VENV_NAME"

echo "--- 2. Activate ---"
source "$VENV_NAME"/Scripts/activate  

echo "--- 3. Installing dependencies (requirements.txt) ---"
pip install -r requirements.txt

echo "--- 4. Installing PyTorch ---"
pip install torch torchvision

echo "--- 5. Verifying PyTorch ---"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

echo "--- Installation complete. ---"
