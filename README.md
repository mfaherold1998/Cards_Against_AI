# Cards Against AI—Evaluating Toxicity and Bias in Language Models

```
python version = 3.12.0
```

This repository contains the code developed for a thesis exploring biases and toxicity in large language models (LLMs) by simulating automated rounds of *Cards Against Humanity*. The project runs independent single-round games, analyzes model outputs, and produces toxicity profiles and visualizations.

---

## 📦 Repository Structure

```
├── cards_dataset/              # Folder containing black and white card datasets (by language)
│   └── EN/                     # English version
│       ├── BLACK_cards.xlsx
│       ├── WHITE_cards.xlsx
│       ├── random_configurations_5.xlsx
│       ├── random_configurations_10.xlsx
│       ├── toxic_configurations_ID_5.xlsx
│       └── toxic_configurations_ID_10.xlsx
│
├── code_script.py              # Main script: runs rounds, measures toxicity, and generates plots
├── ollama_code.ipynb           # Illustrative notebook showing the experiment workflow
├── config.json                 # Configuration file defining runtime parameters
├── requirements.txt            # Python dependencies
├── setup_cpu.sh                # Installation script for CPU environments
├── setup_gpu.sh                # Installation script for GPU environments
└── README.md                   # This file
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/mfaherold1998/Cards_Against_AI.git
cd Cards_Against_AI
```

### 2. Create the environment and install dependencies

Choose the appropriate setup script depending on your hardware:

**CPU:**

```bash
./setup_cpu.sh
```

**GPU (CUDA):**

```bash
./setup_gpu.sh
```

This will create a virtual environment, install all dependencies from `requirements.txt`, and verify your PyTorch installation.

---

## 🚀 Running the Experiment

### 1. Configure parameters

Edit the `config.json` file to define the experiment parameters. Example:

```json
{
    "dataset": "test",
    "rounds": 10,
    "models": ["gemma3:4b"],
    "temperatures": [0.5, 0.8],
    "pick_more_than_2": false
}
```

**Fields:**

- `dataset`: can be `test` (small subset) or `all` (full dataset)
- `rounds`: number of repeated runs per configuration
- `models`: list of Ollama models to evaluate
- `temperatures`: sampling temperatures for the model
- `PICK_more_than_2`: whether to include black cards with more than two blanks

### 2. Run the main script

```bash
python code_script.py --config-file ./config.json
```

This will automatically:

1. Load the cards and game configurations.
2. Query the specified LLMs via **Ollama**.
3. Collect model responses and extract selected card IDs.
4. Reconstruct complete sentences.
5. Compute toxicity metrics using **Detoxify**.
6. Generate multiple `.png` visualizations (curves, distributions, comparisons, etc.).

Results are saved to:

```
./cards_dataset/EN/all_configurations_results.xlsx
```

---

## 📊 Outputs and Visualizations

The script produces the following plots automatically:

- **Toxicity vs Temperature:** mean toxicity by model.
- **Distribution per model:** violin plots showing full distributions.
- **Rates above threshold:** percentage of sentences with toxicity ≥ 0.8.
- **Black card triggers:** black cards that most increase toxicity.
- **Top plays:** most toxic black–white combinations.
- **Instability charts:** standard deviation of toxicity per play.
- **Category comparison:** average scores for insult, threat, identity attack, etc.
- **Language risk:** toxicity profiles per language (as more languages are added).

All figures are saved as `.png` in the root directory.

---

## 🧠 Main Dependencies

As listed in `requirements.txt`:

```txt
numpy==2.3.3
pandas==2.3.3
matplotlib==3.10.7
seaborn==0.13.2
ollama==0.6.0
detoxify==0.5.2
```

**PyTorch** is also required and installed automatically via the setup scripts.

---

## 📘 Illustrative Notebook

`ollama_code.ipynb` demonstrates the full process in a clear, annotated workflow. It is intended for explanation and visualization, not for full-scale batch execution.

---

## 🔬 Experimental Flow

Each model plays multiple *single-round* versions of *Cards Against Humanity*. Instead of generating open-ended text, the model is instructed to return only the **ID** of the chosen white card(s). This avoids generating explicit or harmful text and ensures the model does not refuse participation.

Afterward, complete sentences are reconstructed and analyzed using **Detoxify**, which computes toxicity and subcategory metrics (e.g., insult, threat, identity attack). These are aggregated into statistical profiles and visualized to highlight model behavior differences.
