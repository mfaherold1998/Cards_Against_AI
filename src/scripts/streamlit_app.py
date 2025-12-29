import streamlit as st
import pandas as pd


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

# --- NEW: Function to display queued messages ---
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

# --- NEW: Helper function to truncate long filenames for cleaner UI ---
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
