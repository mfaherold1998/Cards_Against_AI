# Cards Against AI

This project simulates several single-rounds of the game **Cards Against Humanity (CAH)** using **Large Language Models (LLMs)**. The idea is simple but revealing: let models play the game, either by picking the funniest white card or by acting as the judge choosing the winning combination. The setup is designed to explore model behavior, biases, and tendencies toward irreverent or offensive humor.

The full workflow is split into five modules: simulation, sentence construction, toxicity scoring, visualization, and final analysis.

---

## 1. Overview

The system evaluates how different LLMs respond to the provocative style of CAH. Each model can take on two roles:

* **Player**: selects the white card that best matches a given black card.
* **Judge/Card Czar**: compares two or more combinations and chooses a winner.

Throughout the process:

* The model's chosen card IDs are stored.
* Toxicity scores are calculated for the winning results.
* Different configurations (model, temperature, prompt type, etc.) are compared.

The general process involves:

1. Running model simulations.
2. Constructing full sentences from black and white cards.
3. Scoring toxicity using Detoxify and Perspective API.
4. Generating visualizations.
5. Performing a final comparative analysis.

---

## 2. Dataset Description

The project uses the official UK version of the **Cards Against Humanity** deck.

Included resources:

* **Card Text Dataset**: Black and white cards with unique IDs, located in `./data/EN/cards_texts`.
* **Game Configurations**: Files defining the black card and its candidate white cards, located in `./data/EN/games_config` and `./data/EN/to-judge-config`.

Descriptive statistics are available in the notebook `dataset_analysis.ipynb`.

---

## 3. Prerequisites

Before installing, ensure you have:

* **Python 3.10+**
* **Jupyter Notebook** (optional, for running `all_process.ipynb`)

---

## 4. Installation and Cloning

Clone the repository:

```bash
git clone https://github.com/mfaherold1998/Cards_Against_AI.git
cd Cards_Against_AI
```

Run the installation script:

```bash
./setup_cpu.sh
# or
./setup_gpu.sh
```

Example of `setup_cpu.sh`:

```bash
#!/bin/bash

VENV_NAME="venv_cpu"

echo "--- 1. Creating virtual environment ($VENV_NAME) ---"
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
```

---

## 5. Requirements File (requirements.txt)

Core dependencies (version‑locked for reproducibility):

* `numpy==2.3.3`
* `pandas==2.3.3`
* `matplotlib==3.10.7`
* `seaborn==0.13.2`
* `ollama==0.6.0`
* `detoxify==0.5.2`
* `google-api-python-client==2.151.0`
* `google-auth==2.35.0`
* `python-dotenv==1.0.1`

---

## 6. Project Modules and Execution

Each module can be run independently. Every script reads from a directory identified by `run_id`, so execution order is flexible.

| Step | Module               | Command                                                                            |
|-----|-----------------------|------------------------------------------------------------------------------------|
| 1   | `run_llm.py`          | `python 1_run_llm.py --config-file ./config/1_run_config.json`                     |
| 2   | `build_sentences.py`  | `python 2_build_sentences.py --config-file ./config/2_build_sentences_config.json` |
| 3   | `toxicity_scores.py`  | `python 3_toxicity_scores.py --config-file ./config/3_toxicity_config.json`        |
| 4   | `graphs.py`           | `python 4_graphs.py --config-file ./config/4_graphs_config.json`                   |
| 5   | `analysis.py`         | `python 5_analysis.py --config-file ./config/5_analisis_config.json`               |

---

## 7. Detailed Module Descriptions

### 7.1 `run_llm.py`

Runs simulation rounds using the settings in `run_config.json`. Models act as players or judges and return only the selected card IDs. Results are saved to the run directory.

### 7.2 `build_sentences.py`

Loads results from Step 1 and constructs complete sentences by pairing black cards with winning white cards. Produces:

* All winning combinations
* All evaluated combinations

### 7.3 `toxicity_scores.py`

Calculates toxicity scores using two sources:

* **Detoxify** (local model)
* **Google Perspective API**

Outputs include toxicity scores across multiple dimensions.

### 7.4 `graphs.py`

Generates plots such as:

* Toxicity vs. temperature
* Toxicity distribution per model
* Mean toxicity per configuration

Outputs are saved as PNG files.

### 7.5 `analysis.py`

Combines all data to examine:

* Decision consistency
* Success rates
* Toxicity relative to all alternatives
* Effects of judge personality prompts
* Differences across models

Produces both visual and numerical summaries.

---

## 8. Considerations and Future Work

### Current Considerations

* Models may act as both player and judge.
* To avoid refusals, models output only card IDs.
* The dataset currently includes only the UK English CAH deck.
* Invalid responses are handled via a simple fallback mechanism.

### Planned Enhancements

* Improve handling of invalid or ambiguous responses.
* Expand to multilingual or regional CAH decks.
* Add adversarial prompts to probe model robustness.
