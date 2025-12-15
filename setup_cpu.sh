#!/bin/bash

VENV_NAME="venv_cpu"

# # 1. Creating Virtual Enviroment
echo "--- Creating virtual enviroment ($VENV_NAME) ---"
python -m venv "$VENV_NAME"

echo "--- Activate Virtual Enviroment---"
# For MAC: "$VENV_NAME"/bin/activate
source "$VENV_NAME"/Scripts/activate  

# 2. installing requirements.txt
echo "--- Installing dependencies (requirements.txt) ---"
pip install -r requirements.txt

echo "--- Installing PyTorch ---"
pip install torch torchvision

echo "--- Verifying PyTorch ---"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

echo "--- Installation complete. ---"

echo "--- Starting the process ---"

# 3. Running the models and capturing run_id
echo "--- Running ollama models ---"
RUN_ID=$(python 1_run_llm.py --config-file ./config/1_run_config.json)
if [ -z "$RUN_ID" ]; then
    echo "ERROR: No run id captured. Abort."
    exit 1
fi
echo "--- RUN_ID: $RUN_ID ---"

# 4. Setting run_id for all scripts

# Copying templates with the placeholder
CONFIG_2="./config/2_build_sentences_config.json"
CONFIG_3="./config/3_toxicity_config.json"
cp "./config/2_build_sentences_config.json.template" "$CONFIG_2"
cp "./config/3_toxicity_config.json.template" "$CONFIG_3"

# Replacing the run id in the configurations
sed -i "s/RUN_ID_PLACEHOLDER/$RUN_ID/g" ./config/2_build_sentences_config.json
sed -i "s/RUN_ID_PLACEHOLDER/$RUN_ID/g" ./config/3_toxicity_config.json
#For MAC:
#sed -i '' "s/RUN_ID_PLACEHOLDER/$RUN_ID/g" ./config/2_build_sentences_config.json
# sed -i '' "s/RUN_ID_PLACEHOLDER/$RUN_ID/g" ./config/3_toxicity_config.json

echo "--- Configuration files updated successfully. ---"

# 5. Running scripts
python 2_build_sentences.py --config-file ./config/2_build_sentences_config.json
python 3_toxicity_scores.py --config-file ./config/3_toxicity_config.json
python 4_analysis.py --config-file ./config/4_analisis_config.json

# 6. Running streamlit app

echo "--- Process complete. ---"
echo "--- END ---"