"""
Streamlit entry point for Nova Score application.
This file is required for Streamlit Cloud deployment.
"""
import streamlit as st
from pathlib import Path
import sys
import os

# Add the project root and models directory to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "models"))

# Ensure models directory exists
models_dir = project_root / "models"
if not models_dir.exists():
    models_dir.mkdir(parents=True, exist_ok=True)
    st.warning(f"Created models directory at: {models_dir}")

# Debug info
if 'debug_info' not in st.session_state:
    st.session_state.debug_info = {
        'python_path': sys.path,
        'cwd': os.getcwd(),
        'files': [str(f) for f in models_dir.glob('*')] if models_dir.exists() else []
    }

# Import all page modules
from app.score import main as home_page
from app.pages.about import main as about_page
from app.pages.partner import main as partner_page
from app.pages.fairness import main as fairness_page
from app.pages.result import main as result_page
from app.pages.ops import main as ops_page

# Set page config
st.set_page_config(
    page_title="Nova Score - Fair Credit for Grab Partners",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Navigation
st.sidebar.title("Navigation")

# Define all pages including the result page
pages = {
    'Home': 'home',
    'Partner Scoring': 'partner',
    'Fairness Analysis': 'fairness',
    'Operations Console': 'ops',
    'About': 'about',
    'Results': 'result'  # Added result page to navigation
}

# Only show main pages in the sidebar, not the result page
main_pages = {k: v for k, v in pages.items() if v != 'result'}

# Only show navigation if we're not on the result page
if st.session_state.get('page') != 'result':
    selection = st.sidebar.radio("Go to", list(main_pages.keys()))
    st.session_state.page = main_pages[selection]

# Display the selected page
if st.session_state.page == 'home':
    home_page()
elif st.session_state.page == 'about':
    about_page()
elif st.session_state.page == 'partner':
    partner_page()
elif st.session_state.page == 'fairness':
    fairness_page()
elif st.session_state.page == 'ops':
    ops_page()
elif st.session_state.page == 'result':
    # Show a back button when on the result page
    if st.sidebar.button("← Back to Form"):
        st.session_state.page = 'partner'
        st.experimental_rerun()
    result_page()
