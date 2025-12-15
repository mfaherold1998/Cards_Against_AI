# Cards Against AI

This project simulates several single-rounds of the game **Cards Against Humanity (CAH)** using **Large Language Models (LLMs)**. The idea is simple: let models play the game, either by picking the funniest white card or by acting as the judge choosing the winning combination. The setup is designed to explore model behavior, biases, and tendencies toward irreverent or offensive humor.

The full workflow is split into five modules: simulation of the game, sentence construction, toxicity scoring, responses analysis, and visualization.

---

## 1. Overview

The system evaluates how different LLMs respond to the provocative style of CAH. Each model can take on two roles:

* **Player**: selects the white card that best matches a given black card.
* **Judge/Card Czar**: compares two or more combinations and chooses a winner.

Throughout the process:

* The model's chosen card IDs are stored.
* Toxicity measures are calculated for winning cards and for all card combinations among the overall options.
* Different configurations (model, temperature, prompt type, etc.) are compared.

The general process involves:

1. Running model simulations.
2. Constructing full sentences from black and white cards (versions are built only for the winning cards and others for all the cards in the hand).
3. Scoring toxicity using Detoxify and Perspective API.
4. Performing a comparative analysis between the different runs.
5. Generating visualizations.


Notes: Not all models are ready to play; some have stricter safety measures that prevent them from participating from the start.

---

## 2. Dataset Description

The project currently use just English version of the **Cards Against Humanity** deck.

Included resources:
* **Card Text Dataset**: Black and white cards with unique IDs files, located in `./data/EN/cards_texts`.
* **Game Configurations**: Files defining the black card and its candidate white cards, located in `./data/EN/games_config` and `./data/EN/to-judge-config`.

Descriptive statistics are available in the notebook `dataset_analysis.ipynb`.

---

## 3. Prerequisites

Before installing, ensure you have:

* **Python 3.12.0** or grather.
* **Ollama** installed locally.
* **Jupyter Notebook** (optional for running the notebooks provided.)
* **Perspective API Key** (is free for research.)

---

## 4. Installation and Cloning

```bash
git clone https://github.com/mfaherold1998/Cards_Against_AI.git
cd Cards_Against_AI
```

---

## 5. Dependencies (requirements.txt)

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
* `openpyxl==3.1.5`
* `streamlit==1.52.1`
* `plotly==6.5.0`

---

## 6. Project Modules and Execution

Run the script according to the device where you want to run it (remember to give execution permissions with chmod +x):

```bash
./setup_cpu.sh
# or
./setup_gpu.sh
```

`setup_cpu.sh` performs:

* Creates a virtual environment named `venv_cpu` and activates it automatically.
* Installs all the dependencies in requirements.txt
* Runs module 1 (the simulation of the game with LLMs) and obtain the generated run_id
* Propagates the run_id to the configuration files of the other modules and executes them.

Note: Each module can be run independently. Every script reads from a directory identified by `run_id`, so execution order is flexible.

| Step | Module               | Command                                                                            |
|-----|-----------------------|------------------------------------------------------------------------------------|
| 1   | `run_llm.py`          | `python 1_run_llm.py --config-file ./config/1_run_config.json`                     |
| 2   | `build_sentences.py`  | `python 2_build_sentences.py --config-file ./config/2_build_sentences_config.json` |
| 3   | `toxicity_scores.py`  | `python 3_toxicity_scores.py --config-file ./config/3_toxicity_config.json`        |
| 4   | `analysis.py`         | `python 4_analysis.py --config-file ./config/4_analisis_config.json`               |
| 5   | `plots_app.py`        | `streamlit run ./5_plots_app.py`                                                   |

---

## 7. Project structure description

## 8. Detailed Module Descriptions

The following variables can be found in the configuration files:
* **cards_dir** : "./data" (directory that contains the datasets in several languages)
* **results_dir** : "./results" (directory created to store the results of the runs and analyses)
* **languages** : ["EN"] (List of available languages, currently `EN`, `IT`, `ES`)
* **file_type** :"xlsx" (file extension, it can be `xlsx` or `csv`)
* **models** : ["gemma3:4b", "moondream"] (List of Ollama models to be used in the simulation)
* **temperatures** : [0.5, 0.8] (temperatures to test)
* **rounds** : 4 (rounds for each play, that is, a specific combination of cards)
* **dataset_mode** : "test" (Simulation type, can be `test` or `all`)
* **test_num_rows** : 4 (if the simulation is a `test`, the first 4 rows of the dataset are loaded.)
* **prompt_type**: "prompt_player" (can be `prompt_player` or `prompt_judge`)
* **character_description** : "A dark humor lover." (judge personality description)
* **detoxify_model** : "original" (detoxify model to be used; can be `original`, `unbiased`, or `multilingual`.)
* **device** : "cpu" (can be `cpu` or `gpu`)
* **batch_size** : 64 (batch block size for detoxify)
* **analysis_dir** : "./results/analysis_module"

A more detailed description of the entire process can be found in the notebook `all_process.ipynb`

### `run_llm.py`

Runs simulation rounds using the settings in `run_config.json`. Models act as players or judges and return only the selected card IDs. Results are saved to the run directory.

### `build_sentences.py`

Loads results from Step 1 and constructs complete sentences by pairing black cards with winning white cards. Schemes are used to ensure the integrity of the data loaded into each module.

Produces:
* All winning combinations
* All evaluated combinations

### `toxicity_scores.py`

Calculates toxicity scores using two sources:

* **Detoxify** (local model)
* **Google Perspective API**

Outputs include toxicity scores across multiple categories (toxicity, severe toxicity, obscene, etc).

### `analysis.py`

Combines all data to examine:

* Decision consistency
* Success rates
* Toxicity relative to all alternatives
* Effects of judge personality in models responses

Produces numerical summaries.

### `plots_app.py`

Create a graphical interface using the streamlit library to visualize the data obtained in the previous analyses. All graphs are interactive.

---

## 9. Considerations and Future Work

### Current Considerations

* Models may act as both player and judge.
* To avoid refusals, models are asked to output only card IDs.
* The dataset currently includes only the UK English CAH deck.
* Invalid responses are just ignored.

### Planned Enhancements

* Tests several datasets.
* Expand to multilingual or regional CAH decks.
* Add adversarial prompts to probe model robustness.
