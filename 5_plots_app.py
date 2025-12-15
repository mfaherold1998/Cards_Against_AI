import streamlit as st
import pandas as pd
from src.scripts.plotly_express_plots import *

# Constants
CONFIG = "config"
LANG = "lang"
MODEL = "model"
TEMPERATURE = "temperature"
BLACKID = "black_id"
PLAY = "play"
WINNERS = "winners"
SENTENCE = "sentence"
TOXICITY = "toxicity"
SEVERE_TOXICITY = "severe_toxicity"
OBSCENE = "obscene"
THREAT = "threat" 
INSULT = "insult"
IDENTITY_ATTACK = "identity_attack"
SEXUALLY_EXPLICIT = "sexually_explicit"
PROFANITY = "profanity"

ATTRIBUTE_COLUMNS = [TOXICITY, SEVERE_TOXICITY, OBSCENE, THREAT, INSULT, IDENTITY_ATTACK, SEXUALLY_EXPLICIT, PROFANITY]

# Data Loading and Caching Functions
@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    """Load the data from an Excel file with caching."""
    
    if uploaded_file is not None:
        st.info("Loading and processing data. This will only happen once. The file will be saved in cache")
        
        try:
            df = pd.read_excel(uploaded_file)
            
            df[TEMPERATURE] = pd.to_numeric(df[TEMPERATURE], errors='coerce')
            df[TOXICITY] = pd.to_numeric(df[TOXICITY], errors='coerce')
            for attr in ATTRIBUTE_COLUMNS:
                df[attr] = pd.to_numeric(df[attr], errors='coerce')
                        
            return df
        
        except Exception as e:
            st.error(f"Error reading the Excel file. Please ensure the format is correct. Error: {e}")
            return pd.DataFrame()
    
    return pd.DataFrame()

# Plots imported from plotly_express_plots
def plot_toxicity_vs_temperature_px(df: pd.DataFrame):
    
    st.subheader(f"📈 Mean Toxicity vs Temperature with bar errors")

    return plot_toxicity_vs_temperature(df)

def plot_distribution_by_model_px(df: pd.DataFrame):
    
    st.subheader(f"📊Toxicity distribution (violin plot)")
    
    return plot_distribution_by_model(df)

def plot_temperature_curve_per_model_px(df: pd.DataFrame):
    
    st.subheader(f"📊Mean Toxicity vs Temperature with shaded area")
    
    return plot_toxicity_vs_temperature_shaded(df)

def plot_rates_above_threshold_px(df: pd.DataFrame, thr:float = 0.5, column:str = 'toxicity'):
    
    st.subheader(f"📊Percentage of LLM responses that obtained a toxicity score higher than a threshold")
    
    return plot_rates_above_threshold(df, thr, column)

def plot_black_card_triggers_px(df: pd.DataFrame, top_k:int =10, column:str = 'toxicity'):
    
    st.subheader(f"📊Top cards that have triggered the most toxicity in each model")
    
    return plot_black_card_triggers(df, top_k, column)


# Streamlit app logic
def main():
    
    st.set_page_config(layout="wide", page_title="Visualization of LLM response graphs (CAH)")
    
    st.title("Interactive Project Visualization Application")
    st.markdown("Upload the Excel file and select the chart to view.")
    
    # --- File Upload Interface ---
    uploaded_file = st.file_uploader(
        "Upload the excel file (.xlsx)", 
        type=["xlsx", "xls"],
        help="Verify that the file contains all the necessary columns before uploading it."
    )

    df = load_data(uploaded_file)
    
    if uploaded_file is None:
        st.info("Waiting for a file to be load.")
        return
    if df.empty:
        st.error("The DataFrame is empty. Please check the Excel file and the upload code.")
        return

    # Displays uploaded data for verification (optional)
    if st.checkbox("Show loaded data"):
        st.dataframe(df.head())

    # ------------------------------

    # --- Dynamic Chart Selector ---
    
    st.sidebar.header("Graphics Options")
    
    # Dictionary that maps names to Plotly functions (ADD PLOTS HERE!!)
    chart_options = {
        "Mean Toxicity vs Temperature with bar errors": plot_toxicity_vs_temperature_px,
        "Toxicity Distribution per model": plot_distribution_by_model_px,
        "Mean Toxicity vs Temperature with shaded area": plot_temperature_curve_per_model_px,
        "High Tail of Toxicity": plot_rates_above_threshold_px,
        "Top 10 toxic black cards": plot_black_card_triggers_px
    }

    selected_chart_name = st.sidebar.selectbox(
        "1. Select the plot:",
        list(chart_options.keys())
    )

    st.sidebar.markdown("---")

    # ------------------------------

    # --- Top 10 black cards seccion ---
    
    DEFAULT_TOP = 10
    DEFAULT_TOP_COL = 'toxicity'
    top = DEFAULT_TOP
    top_column = DEFAULT_TOP_COL

    if selected_chart_name == "Top 10 toxic black cards":

        if 'high_tail_reset_key' not in st.session_state:
            st.session_state['high_tail_reset_key'] = 0

        reset_key = st.session_state['high_tail_reset_key']

        top = st.sidebar.slider(
            "2. Select the top cards",
            min_value=5,
            max_value=30,
            value=10,
            step=5,
            key=f"thr_slider_{reset_key}",
            help="Set the number of toxic cards to display at the top."
        )

        top_column = st.sidebar.selectbox(
            "3. Select Score Column",
            options=ATTRIBUTE_COLUMNS, 
            index=ATTRIBUTE_COLUMNS.index('toxicity'),
            key=f"score_col_select_{reset_key}",
            help="Selects the column used to calculate the mean and the rate above threshold."
        )

        if st.sidebar.button("Reset Parameters", key="reset_button"):
            st.session_state['high_tail_reset_key'] += 1
            st.rerun()

    #-----------------------------------

    # --- High Tail plot section ---
    
    DEFAULT_THR = 0.5
    DEFAULT_THR_COL = 'toxicity'
    thr = DEFAULT_THR
    thr_column = DEFAULT_THR_COL
    
    if selected_chart_name == "High Tail of Toxicity":

        if 'high_tail_reset_key' not in st.session_state:
            st.session_state['high_tail_reset_key'] = 0

        reset_key = st.session_state['high_tail_reset_key']

        thr = st.sidebar.slider(
            "2. Select Toxicity Threshold (thr)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key=f"thr_slider_{reset_key}",
            help="Sets the threshold for the '% >= thr' calculation in the plot."
        )

        thr_column = st.sidebar.selectbox(
            "3. Select Score Column",
            options=ATTRIBUTE_COLUMNS, 
            index=ATTRIBUTE_COLUMNS.index('toxicity'),
            key=f"score_col_select_{reset_key}",
            help="Selects the column used to calculate the mean and the rate above threshold."
        )

        if st.sidebar.button("Reset Parameters", key="reset_button"):
            st.session_state['high_tail_reset_key'] += 1
            st.rerun()
    
    # ---------------------------------
    
    # --- Get the corresponding Plotly function ---
    selected_chart_func = chart_options[selected_chart_name]
    
    if selected_chart_name == "High Tail of Toxicity":
        fig = selected_chart_func(df, thr=thr, column=thr_column)
    elif selected_chart_name == "Top 10 toxic black cards":
        fig = selected_chart_func(df, top, top_column) 
    else:
        fig = selected_chart_func(df) 
    
    st.plotly_chart(fig, use_container_width=True)
    

if __name__ == "__main__":
    main()