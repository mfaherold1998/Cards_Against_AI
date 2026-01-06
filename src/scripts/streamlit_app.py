import streamlit as st
import pandas as pd
from src.scripts.plotly_express_plots import *

# --- File upload section ---
def initialize_session_state():
    """
    Initializes the 'data_store' and a key for the file uploader reset.
    """
    if 'data_store' not in st.session_state:
        st.session_state['data_store'] = {}
    
    # We use this key to force the file_uploader to reset after usage
    if 'uploader_key' not in st.session_state:
        st.session_state['uploader_key'] = 0

    # --- NEW: Initialize a queue for messages to persist across reruns ---
    if 'toast_queue' not in st.session_state:
        st.session_state['toast_queue'] = []

def show_queued_toasts():
    """
    Checks if there are any messages in the queue and displays them.
    Should be called at the very beginning of the app loop.
    """
    if 'toast_queue' in st.session_state and st.session_state['toast_queue']:
        for msg, icon_type in st.session_state['toast_queue']:
            st.toast(msg, icon=icon_type)
        # Clear queue after showing
        st.session_state['toast_queue'] = []

def truncate_filename(filename, max_length=25):
    """
    Truncates a filename to a fixed length with ellipsis if too long.
    Example: 'very_long_filename.xlsx' -> 'very_long_fi...xlsx'
    """
    if len(filename) > max_length:
        ext_index = filename.rfind('.')
        extension = filename[ext_index:] if ext_index != -1 else ""
        # Keep start of name + "..." + extension
        return filename[:max_length - 5 - len(extension)] + "..." + extension
    return filename

def load_data(uploaded_file):
    """
    Attempts to read a CSV or Excel file and returns a DataFrame.
    """
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            # --- MODIFIED: Push to queue instead of direct toast ---
            st.session_state['toast_queue'].append((f"❌ Format not supported: {uploaded_file.name}", "⚠️"))
            return None
        return df
    except Exception as e:
        # --- MODIFIED: Push to queue instead of direct toast ---
        st.session_state['toast_queue'].append((f"❌ Error reading {uploaded_file.name}: {e}", "⚠️"))
        return None

def add_files_to_session(uploaded_files):
    """
    Processes files and queues messages.
    """
    files_added = False
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state['data_store']:
                df = load_data(uploaded_file)
                if df is not None:
                    st.session_state['data_store'][uploaded_file.name] = df
                    
                    # --- MODIFIED: Use truncated name for the message ---
                    short_name = truncate_filename(uploaded_file.name)
                    st.session_state['toast_queue'].append((f"Loaded: {short_name}", "✅"))
                    
                    files_added = True
    return files_added

def remove_files_from_session(filenames_to_remove):
    """
    Removes a list of filenames from the session state.
    """
    if filenames_to_remove:
        for filename in filenames_to_remove:
            if filename in st.session_state['data_store']:
                del st.session_state['data_store'][filename]
        
        # --- MODIFIED: Simplified message ---
        st.session_state['toast_queue'].append((f"Removed {len(filenames_to_remove)} file(s)", "🗑️"))
        
        st.rerun()

# --- Graphs section and its parameters ---
def get_document_type(filename):
    """
    Analyzes the filename to determine the document type.
    Returns a string identifier (e.g., 'type_1', 'type_2') or None.
    """
    filename = filename.lower()
    
    if "winners" in filename and 'success_rate_by_model'in filename:
        return "success_rate_by_model"
    elif "winners" in filename and 'inconsistencies'in filename:
        return "inconsistencies"
    elif "winners" in filename:
        return "winners"
    elif "combinations" in filename:
        return "combinations"
    elif 'character_description' in filename:
        return 'character_description'
    else:
        return "unknown"

def get_available_charts_for_type(doc_type):
    """
    Returns a dictionary {'Display Name': 'function_name_in_user_file'}
    based on the document type.
    """
    charts = {}
    
    if doc_type == "winners":
        charts = {
            "Toxicity vs Temperature": "plot_toxicity_vs_temperature",
            "Form of the distribution by model": "plot_distribution_by_model",
            "Temperature curve per model": "plot_toxicity_vs_temperature_shaded",
            "High tail by model":"plot_rates_above_threshold",
            "Top toxic black cards":"plot_black_card_triggers",
            "Top toxic plays":"plot_top_plays_heatmap_per_model",
            "Models Instability":"plot_instability",
            "Profile of attributes per model":"plot_category_comparison",
            "Language risk per model":"plot_language_risk_faceted",
            "Mean toxicity by configuration":"plot_config_toxicity_per_model",
            "Toxicity distribution by configuration":"plot_config_distribution",
            "High-toxicity rate by configuration":"plot_config_tail_rate"
        }
    elif doc_type == "combinations":
        charts = {
            "Toxicity Percetage of model choices": "plot_model_tox_percentage"
        }
    elif doc_type == "success_rate_by_model":
        charts = {
            "Success Rate of white cards": "plot_white_cards_success_rate_by_model"
        }
    elif doc_type == "inconsistencies":
        charts = {
            "Consistencies of model choices": "plot_inconsistencies_per_model"
        }
    elif doc_type == "character_description":
        charts = {
            "Models Toxicity comparison by Judges": "plot_jude_description_comparison"
        }
    return charts

def _get_params_col(df):

    valid_columns = [c for c in ATTRIBUTE_COLUMNS if c in df.columns]
    
    if not valid_columns:
        st.error("No toxicity columns were found in this file.")
        return {}
    
    st.subheader("⚙️ Parameters: Temp vs Toxicity")
    
    default_index = 0
    if 'toxicity' in ATTRIBUTE_COLUMNS:
        default_index = ATTRIBUTE_COLUMNS.index('toxicity')
        
    selected_col = st.selectbox(
        "Select attribute to plot:", 
        options=ATTRIBUTE_COLUMNS,
        index=default_index,
        help="Column that will be used to measure toxicity in relation to temperature."
    )
    
    return {'col': selected_col}

def _get_params_thr_col(df):
    
    st.subheader("⚙️ Parameters: Rates > Threshold")
    
    valid_columns = [c for c in ATTRIBUTE_COLUMNS if c in df.columns]
    
    if not valid_columns:
        st.error("No toxicity columns were found in this file.")
        return {} 

    default_index = 0
    if 'toxicity' in valid_columns:
        default_index = valid_columns.index('toxicity')
        
    selected_col = st.selectbox(
        "Select attribute to plot:", 
        options=valid_columns,
        index=default_index,
        key="sb_rates_col"
    )
    
    selected_thr = st.slider(
        "Threshold:",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Rates will be calculated for values ​​exceeding this number."
    )
    
    return {
        'col': selected_col, 
        'thr': selected_thr
    }

def _get_params_topk_col(df):
    
    st.subheader("⚙️ Parameters: Top Triggers")
    
    valid_columns = [c for c in ATTRIBUTE_COLUMNS if c in df.columns]
    
    if not valid_columns:
        st.error("No toxicity columns were found in this file.")
        return {} 

    default_index = 0
    if 'toxicity' in valid_columns:
        default_index = valid_columns.index('toxicity')
        
    selected_col = st.selectbox(
        "Select attribute to plot:", 
        options=valid_columns,
        index=default_index,
        key="sb_triggers_col_1"
    )
    
    selected_k = st.slider(
        "Top K:",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
        help="Define how many triggers with higher toxicity will be displayed."
    )
    
    return {
        'col': selected_col, 
        'top_k': selected_k
    }

def _get_params_min_n_col(df):
    
    st.subheader("⚙️ Parameters: Top Triggers")
    
    valid_columns = [c for c in ATTRIBUTE_COLUMNS if c in df.columns]
    
    if not valid_columns:
        st.error("No toxicity columns were found in this file.")
        return {} 

    default_index = 0
    if 'toxicity' in valid_columns:
        default_index = valid_columns.index('toxicity')
        
    selected_col = st.selectbox(
        "Select attribute to plot:", 
        options=valid_columns,
        index=default_index,
        key="sb_triggers_col_2"
    )
    
    selected_k = st.slider(
        "Min n:",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        help="Graph only the entries that have more than Min n rounds.."
    )
    
    return {
        'col': selected_col, 
        'min_n': selected_k
    }

def _get_params_config_col(df):
    
    st.subheader("⚙️ Parameters: Config Distribution")
    
    valid_columns = [c for c in ATTRIBUTE_COLUMNS if c in df.columns]
    
    if not valid_columns:
        st.error("No toxicity columns were found in this file.")
        return {} 

    col_idx = 0
    if 'toxicity' in valid_columns:
        col_idx = valid_columns.index('toxicity')
        
    selected_col = st.selectbox(
        "Select attribute to plot:", 
        options=valid_columns,
        index=col_idx,
        key="sb_conf_dist_col"
    )
    
    if 'config' not in df.columns:
        st.error("There is not column 'config'.")
        return {}

    unique_configs = sorted(df['config'].astype(str).unique().tolist())
    config_idx = 0
    default_target = 'toxic_games_9'
    
    if default_target in unique_configs:
        config_idx = unique_configs.index(default_target)
    
    selected_config = st.selectbox(
        "Select config to plot:",
        options=unique_configs,
        index=config_idx,
        help="Select one of the configurations present in the data."
    )
    
    return {
        'col': selected_col, 
        'target_config': selected_config
    }

def render_chart_controls(chart_function_name, df):
    """
    It receives the name of the selected function and delegates
    the creation of inputs to the corresponding handler.
    """
    params = {}
    
    if chart_function_name in ["plot_toxicity_vs_temperature", 
                               "plot_toxicity_vs_temperature_shaded",
                               "plot_distribution_by_model",
                               "plot_language_risk_faceted"]:
        params = _get_params_col(df)
    
    elif chart_function_name in ["plot_rates_above_threshold", 
                               "plot_config_tail_rate"]:
        params = _get_params_thr_col(df)

    elif chart_function_name in ["plot_black_card_triggers", 
                               "plot_top_plays_heatmap_per_model",
                               "plot_instability"]:
        params = _get_params_topk_col(df)

    elif chart_function_name in ["plot_config_toxicity_per_model"]:
        params = _get_params_min_n_col(df)

    elif chart_function_name in ["plot_config_distribution"]:
        params = _get_params_config_col(df)
        
    else:
        st.info("This chart does not have configurable parameters.")
        
    return params

# --- Graphs desciptions ---
CHART_DESCRIPTIONS = {
    
    # Plot 1
    "plot_toxicity_vs_temperature": 
    """
    **Description:** Generates a line chart with error bars. It shows the mean toxicity on the Y-axis versus the generation temperature on the X-axis, with a separate line for each model. The error bars represent the 95% confidence interval (using 1.96 x Standard Error of the Mean (SEM)).
    
    **How to interpret:** If the line rises steeply, it means increasing randomness (temperature) significantly increases the mean toxicity of the model's output. If the error bars are large, it indicates higher variability in toxicity scores at that temperature.
    """,
    
    #Plot 2
    "plot_toxicity_vs_temperature_shaded": 
    """
    **Description:** Generates separate line charts (one per model). Each chart shows the mean toxicity versus temperature, similar to the first function, but with the 95% confidence interval shaded instead of using error bars.

    **How to interpret**: Represents the individual trend of toxicity vs. temperature for each model, with a visual emphasis on the uncertainty area.
    """,
    
    # Plot 3
    "plot_distribution_by_model": 
    """
    **Description:** Generates a violin plot. It shows the distribution of the toxicity score for each model. The shape of the violin represents the probability density of the toxicity score.

    **How to interpret**: The wider parts of the "violin" indicate a higher density of observations (more generated content) having that specific toxicity score. A violin plot that is wider near the high end of the Y-axis (e.g., closer to 1.0) suggests that the model frequently produces content with high toxicity.
    """,

    # Plot 4
    "plot_rates_above_threshold": 
    """
    **Description:** Percentage of generated content whose toxicity score is greater than or equal to a specified threshold (thr, default 0.8), plotted against the temperature for each model.

    **How to interpret**: A high percentage (Y-axis) means the model frequently generates highly toxic content (above the threshold) at that temperature. This plot represents the frequency or rate at which each model produces high-toxicity responses ("tail of the distribution") as a function of temperature.
    """,

    # Plot 5
    "plot_black_card_triggers": 
    """
    **Description:** Generates a heatmap. It shows the mean toxicity grouped by the black cards (black_id) that triggered the most toxic responses (the top_k with the highest mean toxicity) and the model. 

    **How to interpret**: Brighter colors (yellow in the default viridis colormap) indicate a higher mean toxicity score for that specific black card. This heatmap identifies which prompts (Y-axis) are most likely to trigger high toxicity responses from models. A column that is consistently brightly colored across many black cards suggests that model is generally more susceptible to toxic prompting.
    """,

    # Plot 6
    "plot_top_plays_heatmap_per_model": 
    """
    **Description:** Generates a heatmap. It shows the mean toxicity for the specific combinations of black card + winning response (play_key) that were most toxic (the top_k) versus the model.

    **How to interpret**: Brighter colors represent a higher mean toxicity for that specific play combination. Unlike the black card trigger plot, this shows the specific combinations of prompt and completion that are the most toxic. It helps pinpoint highly toxic generated phrases.
    """,

    # Plot 7
    "plot_instability": 
    """
    **Graph: Mean Toxicity vs. Instability**

    **Description:** Shows the mean toxicity on the X-axis versus the standard deviation of toxicity (instability) on the Y-axis. The point size reflects the number of observations (n).

    **How to interpret**: High X-Value (High Mean Toxicity): The play is frequently toxic. High Y-Value (High Instability): The toxicity score for that play varies widely across different generation attempts.  Plays in the top-right quadrant (High Mean, High STD) are the most concerning: they are often toxic and unpredictable.

    **Graph: Toxic stable/unstable plays**

    **Description:** Identifies plays that are consistently toxic (high mean, low STD), those that are consistently non-toxic (low mean, low STD), and the unstable/risky plays (high STD), where the model is sometimes toxic and sometimes not.

    **How to interpret**: Highlights the most unstable interactions.
    """,

    # Plot 8
    "plot_category_comparison": 
    """
    **Description:** Generates a grouped bar chart. It shows the average score for all toxicity attribute categories (e.g., SEVERE_TOXICITY, INSULT, PROFANITY, etc.) grouped by model.

    **How to interpret**: The risk profile of each model, showing whether a model is more prone to generating obscene content, insults, threats, etc., compared to other models.
    """,

    # Plot 9
    "plot_language_risk_faceted": 
    """
    **Description:** Generates separate bar charts (one per language LANG). Each chart shows the toxicity metrics (mean, p50, p90, p95 percentiles) for all models in that language.

    **How to interpret**: The mean and 50th percentile (median) show the typical toxicity score.  The 90th and 95th percentiles are measures of the distribution's tail (the high-risk values). A high P95 indicates that 5% of the content generated by that model/language combination is extremely toxic. If Model A has a low mean in Language X but a high P95, it means while generally safe, it occasionally produces extreme toxicity in that language. If Model B has high scores across all percentiles, it is fundamentally more toxic in that language.
    """,

    # Plot 10
    "plot_config_toxicity_per_model": 
    """
    **Description:** Generates a heatmap. It shows the mean toxicity grouped by configuration (racism, random, etc) and model.

    **How to interpret**: How different configurations influence the toxicity of each model's responses.
    """,

    # Plot 11
    "plot_config_distribution": 
    """
    **Description:** Generates a violin plot for the toxicity distribution by configuration.

    **How to interpret**: 
    * **Toxicity distribution by configuration (split by model)**: This allows for a granular comparison. You can see not only if a configuration is risky overall but also which model performs the worst (or best) under that specific configuration.
    """,

    # Plot 12
    "plot_config_tail_rate": 
    """
    **Description:** Generates a grouped bar chart. It shows the percentage of responses above the toxicity threshold grouped by configuration and model.

    **How to interpret**: The height represents the high-toxicity rate for a given configuration/model combination. This is a direct measure of risk based on configuration. Higher bars indicate configurations that are highly likely to produce critically toxic outputs.
    """,

    # Plot 13
    "plot_white_cards_success_rate_by_model": 
    """
    **Description:** A Horizontal Bar Chart displaying the Observed Success Rate (Victories / Appearances) for the top n white_id cards across the entire dataset. The y-axis represents the the specific card and the x-axis represents the calculated Success_Rate, ranging from 0 to 1. The bars are sorted in descending order by the Success Rate. The number of times each card appears in the play column, multiplied by the number of rounds played for each play, represents the `Appearances` . Each `Victory` represents the card's victory in a round.

    **How to interpret**: High Success Rate (Bars extending far to the right): A card with a high Success Rate (closer to $1.0$) is one that was chosen as the winner a large percentage of the times it appeared as an option.
    """,

    # Plot 14
    "plot_inconsistencies_per_model": 
    """
    **Description:** A Grouped Vertical Bar Chart illustrating the Majority Election Rate (MER) for different Language Models (LLMs) across various unique game configurations. The X-axis represents the Configuration_Key, which is a concatenated identifier of the game parameters (config, lang, temperature, black_id). Each unique key represents a specific game setup. The Y-axis represents the MER, ranging from 0 to 1. The bars are grouped and colored by the LLM Model, allowing for direct comparison of consistency between models. 

    **How to Interpret**: High MER (Bar close to 1.0): Indicates high consistency. For that specific game setup, the model repeatedly chose the same winning card in a high percentage of the rounds. This suggests the model's judgment was stable. Low MER (Bar close to 0.0): Indicates low consistency or high variability. The model frequently switched its choice of the winning card across the rounds for that single configuration. If Model A has a much higher bar than Model B for the same configuration, Model A is generally more robust and consistent in its choice for that specific game.
    """,

    # Plot 15
    "plot_model_tox_percentage": 
    """
    **Description:** A Normalized Stacked Vertical Bar Chart showing the distribution of the winning card's toxicity pattern across different Language Models (LLMs). The height of each bar segment represents the percentage of rounds where the winning card fell into one of three categories relative to the other available cards: 
    - Most Toxic: The winning card had the highest toxicity score among the choices. (Colored Red for warning)
    - Least Toxic: The winning card had the lowest toxicity score among the choices. (Colored Green for preference against toxicity)
    - Intermediate: The winning card's toxicity score was between the highest and lowest available scores. (Colored Gold)

    **How to interpret**: The chart immediately reveals a model's bias towards toxicity in its choices. 
    - A model with a large red segment (e.g., Model B at 40%) has a strong tendency to select the most toxic card available in a round. This indicates a potential alignment or lack of sensitivity to toxic content. 
    - A model with a large green segment (e.g., Model C at 65%) shows a preference for choosing the least toxic card available. This suggests the model may be biased towards safer or non-offensive content, potentially acting as a "detoxifier." 
    - The "Intermediate" segment size shows how often the model is selecting cards that are neither the absolute best nor absolute worst in terms of toxicity, suggesting a more nuanced choice based on non-toxicity factors (e.g., humor or relevance).
    """,

    # Plot 16
    "plot_jude_description_comparison": 
    """
    **Description:** This chart employs a Faceted Bar Plot to illustrate how different Judge Descriptions (character_description) influence the average toxicity scores (mean_toxicity and mean_severe_toxicity) of two distinct language Models. The visualization breaks down the results into separate subplots (facets), where each subplot represents one unique judge description. Within each facet, the models are directly compared across the two measured toxicity metrics.

    **How to Interpret**: Examine each subplot (facet) individually. The facet title indicates the Judge Description currently being analyzed (e.g., "Agressive Critic"). This comparison shows which model (represented by color) yielded higher average toxicity scores when exposed to that specific judge persona.
    """,
    
}

def get_chart_description(chart_function_name):
    
    return CHART_DESCRIPTIONS.get(chart_function_name, None)