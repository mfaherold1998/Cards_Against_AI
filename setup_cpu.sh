#!/bin/bash

VENV_NAME="venv_cpu"

echo "--- 1. Creating virtual enviroment ($VENV_NAME) ---"
python -m venv "$VENV_NAME"

echo "--- 2. Activate ---"
source "$VENV_NAME"/Scripts/activate  

echo "--- 3. Installing dependencies (requirements.txt) ---"
pip install -r requirements.txt

echo "--- 4. Installing PyTorch with CUDA (GPU) ---"
pip install torch torchvision

echo "--- 5. Verifying PyTorch and CUDA ---"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo "--- Installation complete. ---"
