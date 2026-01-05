import streamlit as st
from src.scripts import streamlit_app as sa
try:
    from src.scripts import plotly_express_plots as user_plots
except ImportError:
    user_plots = None

# Page configuration
st.set_page_config(page_title="Toxicity Plots of LLMs", layout="wide")

# --- MODIFIED: CSS Injection ---
def inject_custom_css():
    st.markdown("""
        <style>
        /* 1. HIDE EVERYTHING UNDER THE UPLOADER */
        [data-testid='stFileUploaderUploadedFiles'] {
            display: none;
        }

        /* 2. REPOSITION TOASTS TO BOTTOM-RIGHT */
        [data-testid="stToastContainer"] {
            top: unset;
            bottom: 20px;
            right: 20px;
            pointer-events: none; 
            width: auto;
        }

        /* 3. STYLE & ANIMATION FOR TOASTS */
        [data-testid="stToast"] {
            pointer-events: auto !important;
            background-color: #333;
            padding: 10px;
            border-radius: 8px;
            width: 320px; 
            animation: slideUpFade 0.5s cubic-bezier(0.23, 1, 0.32, 1) forwards;
        }
        
        [data-testid="stToast"] button {
            pointer-events: auto !important;
        }

        @keyframes slideUpFade {
            0% {
                opacity: 0;
                transform: translateY(100px);
            }
            100% {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* --- NEW SECTION: CUSTOMIZE DROPDOWNS (SELECTBOX & MULTISELECT) --- */
        
        /* A. The input box itself (closed state) */
        div[data-baseweb="select"] > div {
            border: 1px solid #666 !important; /* Visible grey border */
            background-color: #0E1117; /* Dark background matching app */
        }

        /* B. The Dropdown Menu (The list that opens up) */
        div[data-baseweb="popover"] {
            border: 1px solid #999 !important; /* Lighter border to make it POP */
            background-color: #151515 !important; /* Slightly different dark bg */
            box-shadow: 0px 4px 15px rgba(0,0,0,0.5); /* Shadow for depth */
        }

        /* C. The individual options inside the list */
        li[role="option"] {
            border-bottom: 1px solid #333; /* Subtle separator line */
        }
        
        /* D. Highlight color when hovering an option */
        li[role="option"]:hover {
            background-color: #333 !important;
        }

        </style>
    """, unsafe_allow_html=True)

def main():
    st.title("📊 Interactive Graphics Generator")

    # --- NEW: Inject CSS at the start of the app ---
    inject_custom_css()
    
    # 1. Initialize State
    sa.initialize_session_state()

    # --- NEW: Process and show any pending messages from the previous run ---
    # This ensures the UI draws first, and then the toasts fade in smoothly.
    sa.show_queued_toasts()

    # --- SIDEBAR: Upload & Selection ---
    with st.sidebar:

        st.header("📂 Data Management")

        # Calculamos los archivos disponibles antes para usar el contador en el título
        available_files = list(st.session_state['data_store'].keys())

        with st.expander(f"🗂️ Loaded Files ({len(available_files)})", expanded=False):

            if not available_files:
                st.info("No files loaded.")
                selected_file = None # Aseguramos que la variable exista
            else:
                # 1. File select section
                # index=None hace que no haya nada seleccionado al inicio
                selected_file = st.selectbox(
                    "Select Active File to Visualize:", 
                    options=available_files,
                    index=None, 
                    placeholder="Choose a file..." 
                )
                
                # 2. Remove Button Logic
                # El botón siempre se muestra si hay archivos, pero se desactiva si no hay selección
                btn_disabled = (selected_file is None)
                
                btn_label = f"🗑️ Remove"

                if st.button(btn_label, disabled=btn_disabled, key="btn_remove_single"):
                    # IMPORTANTE: Pasamos selected_file dentro de una lista [] 
                    # porque tu función remove_files_from_session espera iterar sobre una lista.
                    sa.remove_files_from_session([selected_file])

        st.markdown("---")

        # --- GRAPHICS MENU (LOGIC INTEGRATION) ---
        selected_chart_function = None
        chart_params = {}

        if selected_file:
            st.header("📈 Visualization Settings")
            
            # 1. Identify Document Type
            doc_type = sa.get_document_type(selected_file)
            
            # 2. Get Available Charts for this Type
            available_charts_dict = sa.get_available_charts_for_type(doc_type)
            
            if not available_charts_dict:
                st.warning(f"No charts defined for document type: '{doc_type}'")
            else:
                # 3. Chart Selection
                selected_chart_name = st.selectbox(
                    "Select Plot",
                    options=list(available_charts_dict.keys()),
                    index=None,
                    placeholder="Choose a chart type..."
                )
                
                # 4. Render Dynamic Controls (Thresholds, Columns, etc.)
                if selected_chart_name:
                    
                    selected_chart_function = available_charts_dict[selected_chart_name]
                    
                    # Pass the dataframe to the logic to populate selectboxes with real columns
                    df = st.session_state['data_store'][selected_file]
                    chart_params = sa.render_chart_controls(selected_chart_function, df)
        
        else:
            st.info("Select a file above to see available charts.")
        

    # --- MAIN AREA ---

    # A. File Uploader
    uploaded_files = st.file_uploader(
        "Upload new documents", 
        type=['csv', 'xlsx', 'xls'], 
        accept_multiple_files=True,
        key=st.session_state['uploader_key']
    )

    # Logic to process and clear the uploader
    if uploaded_files:
        sa.add_files_to_session(uploaded_files)
        # Increment key to reset the uploader widget visually
        st.session_state['uploader_key'] += 1
        # Force a rerun so the uploader clears immediately
        st.rerun()

    # 2. Preview Area
    if selected_file:

        st.markdown(f"### 📄 Active File: `{selected_file}`")

        # --- MODIFIED: Small UI tweak for consistency ---
        if st.checkbox("Show data"): # Optional: value=True if you want it open by default      
        
            # Retrieve the dataframe
            df = st.session_state['data_store'][selected_file]
            
            # Simple preview as requested
            st.dataframe(df.head(), use_container_width=True)
            st.caption(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")

        st.markdown("---")

        # 2. Render the Chart
        if selected_chart_function and user_plots:
            try:
                # Dynamically get the function from your file
                if hasattr(user_plots, selected_chart_function):
                    plot_func = getattr(user_plots, selected_chart_function)
                    
                    # Call the function passing df + the dynamic params dictionary
                    df = st.session_state['data_store'][selected_file]
                    fig = plot_func(df, **chart_params)
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("The function returned no figure.")
                else:
                    st.error(f"Function `{selected_chart_function}` not found in `plotly_express_plots.py`")
            except Exception as e:
                st.error(f"Error generating chart: {e}")
                st.exception(e) # This helps debugging exactly which parameter failed
        elif selected_file and not selected_chart_function:
            st.info("👈 Select a chart from the sidebar to visualize.")
        
    else:
        st.info("👈 Please upload and select a file from the sidebar to start.")

if __name__ == "__main__":
    main()