"""
Main application entry point for Nova Score.
"""
import streamlit as st
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Nova Score - Fair Credit for Grab Partners",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton>button {
        width: 100%;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def home_page():
    """Render the home page with quick navigation and clear CTAs."""
    # Set page config for better mobile experience
    st.markdown("""
    <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero {
            text-align: center;
            margin-bottom: 3rem;
        }
        .hero h1 {
            font-size: 2.75rem;
            margin-bottom: 1rem;
            font-weight: 700;
            line-height: 1.2;
        }
        .cta-button {
            margin: 1rem 0;
        }
        .secondary-ctas {
            display: flex;
            gap: 1rem;
            justify-content: center;
            margin: 2rem 0;
        }
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: #f8f9fa;
            padding: 1rem;
            text-align: center;
            font-size: 0.85rem;
            color: #6c757d;
            border-top: 1px solid #dee2e6;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Hero Section
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    st.markdown("# Nova Score for Grab Partners")
    st.markdown("### Fair, explainable credit")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Main CTA
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("Check Eligibility", type="primary", use_container_width=True, 
                    help="Start your credit assessment"):
            st.session_state.page = "partner"
    
    # Secondary CTAs
    st.markdown('<div class="secondary-ctas">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Fairness Dashboard", use_container_width=True, 
                    help="View fairness metrics and analysis"):
                st.session_state.page = "fairness"
    with col2:
        if st.button("🖥️ ControlDesk", use_container_width=True,
                    help="Access the partner management control desk"):
            st.session_state.page = "ops"
    
    # Footer
    st.markdown('<div class="footer">Scores shown are from a demo model with synthetic data.</div>', 
                unsafe_allow_html=True)

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Page routing
if st.session_state.page == 'home':
    home_page()
elif st.session_state.page == 'partner':
    st.switch_page("pages/partner.py")
elif st.session_state.page == 'result':
    st.switch_page("pages/result.py")
elif st.session_state.page == 'fairness':
    st.switch_page("pages/fairness.py")
elif st.session_state.page == 'ops':
    st.switch_page("pages/ops.py")
elif st.session_state.page == 'about':
    st.switch_page("pages/about.py")
