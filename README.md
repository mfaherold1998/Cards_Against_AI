# Cards Against AI

A comprehensive research tool to simulate and analyze **Cards Against Humanity (CAH)** games played by Large Language Models (LLMs). This project explores model behavior, decision-making biases, and humor tendencies through toxicity scoring and multi-dimensional analysis.

## Table of Contents
- [1. Project Description](#1-project-description)
- [2. Data Description](#2-data-description)
- [3. Result Types](#3-result-types)
- [4. Graphics and Visualizations](#4-graphics-and-visualizations)
- [5. Technical Description](#5-technical-description)
- [6. Execution Guide](#6-execution-guide)
- [7. Installation Instructions](#7-installation-instructions)

---

## 1. Project Description

Cards Against AI is a simulation framework designed to evaluate how different LLMs interact with the irreverent and often provocative style of Cards Against Humanity.

### Main Objective
The primary goal is to assess how models handle offensive humor and whether their choices align with human-like preferences or specific toxicity profiles. By assigning roles like **Player** (choosing the best white card) or **Judge** (selecting a winner from combinations), we can quantify "model personality" and safety guardrails.

### General Workflow
The system follows a linear pipeline:

*   **Simulation:** Models play rounds under various temperatures and configurations.
*   **Reconstruction:** Resulting IDs are converted back into full sentences.
*   **Scoring:** Sentences are evaluated for toxicity using both local and cloud-based APIs.
*   **Analysis:** Comparative statistics are calculated between models and prompts.
*   **Visualization:** Interactive dashboards present the findings.

## 2. Data Description

The project manages three primary data entities:

*   **Black Cards:** Question or fill-in-the-blank cards that set the context for the round.
*   **White Cards:** Response cards containing potential answers. The system handles thousands of unique combinations.
*   **Games (Matches):** Configuration files that define specific matchups. There are two types of files based on their purpose:
    *   **Player Configs:** Sets designed for models to pick a single white card.
    *   **Judge Configs:** Sets designed for models to act as the "Card Czar" and evaluate combinations.

> **Note:** While their purpose differs, their internal structure remains consistent to ensure system compatibility.

The dataset currently includes regional versions for both **English (UK)** and **English (USA)**, located in `./data/EN/`.

For a detailed statistical breakdown and description of each file, please refer to the `all_datasets.ipynb` notebook in the `notebooks/` directory.

## 3. Result Types

The framework generates five specialized types of results to provide a 360-degree view of the simulation:

*   **Winning Combination Toxicity:** Individual toxicity scores for the specific combinations (Black Card + Model's chosen White Card) selected during the rounds.
*   **Alternative Option Profiles:** Toxicity scores for every possible combination available in a hand (Black Card paired with each candidate White Card). This allows the calculation of a "toxicity pattern" for each model compared to what it could have chosen.
*   **Success Rate Analysis:** Metrics for each White Card, comparing the frequency with which it was chosen against the number of times it was available as an option in the matches.
*   **Decision Inconsistency:** Analysis of variability across multiple rounds using identical card combinations. This measures how deterministic or erratic a model's "sense of humor" is under different parameters.
*   **Model Personality & Influence:** A global analysis of all runs to define the toxicity profile of each model. This includes a comparison based on the Judge's character description, verifying if models are susceptible to "persona" influence or if they maintain a consistent role regardless of the prompt's personality instructions.

## 4. Graphics and Visualizations

The project includes **16 interactive visualizations** built with Plotly. A detailed description of each chart is available in the `all_plots.ipynb` notebook located in the `notebooks/` directory. Additionally, a Streamlit application has been developed for enhanced visualization and interaction with the results, where comprehensive descriptions of each plot are also included.

These visualizations include:

*   **Toxicity Distribution:** Histograms comparing different models.
*   **Model Consistency:** Heatmaps showing how often a model picks the same card at different temperatures.
*   **Winning Patterns:** Bar charts showing the most frequently chosen cards.
*   **Toxicity vs. Success:** Scatter plots exploring if "funnier" (winning) cards are inherently more toxic.

## 5. Technical Description

### Used Technologies
*   **Language:** Python 3.12+
*   **Package Management:** `uv` (for ultra-fast environment resolution)
*   **Environment:** Jupyter Notebooks (for exploratory analysis)
*   **APIs:** Google Perspective API (cloud toxicity scoring)
*   **Local Models:** Detoxify (local toxicity analysis), Ollama (LLM orchestration)
*   **UI & Charts:** Streamlit & Plotly Express

### Project Structure

```text
.
├── 1_run_llm.py            # Step 1: LLM Simulation
├── 2_build_sentences.py    # Step 2: Text reconstruction
├── 3_toxicity_scores.py    # Step 3: Scoring with Detoxify/Perspective
├── 4_analysis.py           # Step 4: Statistical analysis
├── 5_plots_app.py          # Step 5: Streamlit Dashboard
├── config/                 # Module configuration files (.json)
├── data/                   
│   └── EN/                 # UK and USA card datasets and game configs
├── notebooks/              # Analysis, data stats, and visualization walkthroughs
├── logs/                   # Error tracing and execution monitoring
├── results/                # Organized by run_id (contains outputs of all modules)
├── setup_cpu.sh            # Automatic setup and execution for CPU
├── pyproject.toml          # Project dependencies for uv
└── .env                    # Environment variables (API Keys)
```
### Modules Workflow & Results

## 6. Execution Guide

### 6.1 Master Script
The project provides automation scripts that execute the entire pipeline, from simulation to the launching of the Streamlit dashboard. JSON configuration templates have been created for seamless automation.

> **Note:** Ensure the scripts have execution permissions.

```bash
# Grant execution permissions
chmod +x setup_cpu.sh

# Run the full pipeline (includes Streamlit app launch)
./setup_cpu.sh
```

### 6.2 Individual Module Commands
If you wish to run each module separately, simply replace the variables in the corresponding JSON configuration files manually.

| Module | Command |
| :--- | :--- |
| **Simulation** | `python 1_run_llm.py --config-file ./config/1_run_config.json` |
| **Building** | `python 2_build_sentences.py --config-file ./config/2_build_sentences_config.json` |
| **Scoring** | `python 3_toxicity_scores.py --config-file ./config/3_toxicity_config.json` |
| **Analysis** | `python 4_analysis.py --config-file ./config/4_analisis_config.json` |
| **Dashboard** | `streamlit run ./5_plots_app.py` |

## 7. Installation Instructions

Follow these steps to set up the project locally.

**1. Clone the Repository:**

```bash
git clone https://github.com/mfaherold1998/Cards_Against_AI.git
cd Cards_Against_AI
```

**2. Install `uv` on your system globally**

*   **macOS / Linux:**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

*   **Windows (PowerShell):**
    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

**3. Prerequisites (Ollama & API Keys):**

*   **Ollama:** Install Ollama and download the models you intend to use (e.g., `ollama pull gemma3:4b`).
*   **Perspective API:** Create a `.env` file in the root directory and add your Google Perspective API key:

    ```env
    PERSPECTIVE_API_KEY=your_api_key_here
    ```

**4. Run the master script:**
Run `./setup_cpu.sh`. It will be responsible for creating or synchronizing the virtual environment with `uv`. (Verify that you have execution permissions).