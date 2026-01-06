#!/bin/bash

# 1. Environment Preparation
# 'uv sync' ensures the virtual environment is up to date according to pyproject.toml and uv.lock
echo "--- Sincronizando dependencias con uv ---"
uv sync

echo "--- Checking PyTorch ---"
uv run python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

echo "--- Starting the process ---"

# 2. Running the models and capturing run_id
echo "--- Running ollama models ---"

RUN_ID=$(uv run 1_run_llm.py --config-file ./config/1_run_config.json)

if [ -z "$RUN_ID" ]; then
    echo "ERROR: No run id captured. Abort."
    exit 1
fi
echo "--- RUN_ID: $RUN_ID ---"

# 3. Setting run_id for all scripts
CONFIG_2="./config/2_build_sentences_config.json"
CONFIG_3="./config/3_toxicity_config.json"
cp "./config/2_build_sentences_config.json.template" "$CONFIG_2"
cp "./config/3_toxicity_config.json.template" "$CONFIG_3"

if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' "s/RUN_ID_PLACEHOLDER/$RUN_ID/g" "$CONFIG_2"
    sed -i '' "s/RUN_ID_PLACEHOLDER/$RUN_ID/g" "$CONFIG_3"
else
    sed -i "s/RUN_ID_PLACEHOLDER/$RUN_ID/g" "$CONFIG_2"
    sed -i "s/RUN_ID_PLACEHOLDER/$RUN_ID/g" "$CONFIG_3"
fi

echo "--- Configuration files updated successfully. ---"

# 4. Running scripts
uv run 2_build_sentences.py --config-file ./config/2_build_sentences_config.json
uv run 3_toxicity_scores.py --config-file ./config/3_toxicity_config.json
uv run 4_analysis.py --config-file ./config/4_analisis_config.json

# 5. Running streamlit app
echo "--- Starting Streamlit App ---"
uv run streamlit run 5_plots_app.py

echo "--- Process complete. ---"
echo "--- END ---"