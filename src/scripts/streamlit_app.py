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