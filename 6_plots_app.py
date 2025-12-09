import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from io import BytesIO
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
    
    st.sidebar.header("Graphics Options")
    
    # --- Dynamic Chart Selector ---
    
    # Dictionary that maps names to Plotly functions (ADD PLOTS HERE!!)
    chart_options = {
        "Mean Toxicity vs Temperature with bar errors": plot_toxicity_vs_temperature_px,
        "Toxicity Distribution per model": plot_distribution_by_model_px,
        "Mean Toxicity vs Temperature with shaded area": plot_temperature_curve_per_model_px
    }

    selected_chart_name = st.sidebar.selectbox(
        "1. Select the plot:",
        list(chart_options.keys())
    )
    
    # Get the corresponding Plotly function
    selected_chart_func = chart_options[selected_chart_name]
    fig = selected_chart_func(df) 
    st.plotly_chart(fig, use_container_width=True)
    

if __name__ == "__main__":
    main()