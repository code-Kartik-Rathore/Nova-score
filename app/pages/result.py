"""
Result Page - Shows the score and recommendations with engaging animations
"""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import sys
import os
import joblib
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from utils
from utils.animations import animate_score_counter, add_confetti, add_success_message
from utils.model_loader import load_xgboost_model, ensure_model_files

# Load the model and metadata
@st.cache_resource
def get_model():
    """Load and cache the XGBoost model."""
    # Ensure model files exist
    ensure_model_files()
    
    # Load the model and metadata
    model, metadata = load_xgboost_model()
    if model is None or metadata is None:
        st.error("Failed to load the scoring model. Please check the model files.")
        st.stop()
    return model, metadata

# Page config
st.set_page_config(page_title="Your Nova Score", page_icon="📊")

def preprocess_input(data, metadata):
    """Preprocess input data for model prediction."""
    try:
        # Define expected features in the exact order the model expects them
        expected_features = [
            'months_on_platform', 'weekly_trips', 'cancel_rate', 
            'on_time_rate', 'avg_rating', 'earnings_volatility', 'tenure_days',
            'region_East', 'region_North', 'region_South', 'region_West'
        ]
        
        # Initialize with one row of zeros
        input_data = pd.DataFrame(0.0, index=[0], columns=expected_features)
        
        # Map form inputs to model features with type conversion
        form_data = {
            'weekly_trips': float(data.get('weekly_trips', 0)),
            'cancel_rate': float(data.get('cancel_rate', 0)) / 100.0,  # Convert percentage to decimal
            'on_time_rate': float(data.get('on_time_rate', 0)) / 100.0,  # Convert percentage to decimal
            'avg_rating': float(data.get('avg_rating', 0)),
            'earnings_volatility': float(data.get('earn_cv', 0)),
            'months_on_platform': float(data.get('tenure_months', 0)) * 30,  # Convert months to days
            'tenure_days': float(data.get('tenure_months', 0)) * 30,  # Same as months_on_platform
            'region': str(data.get('region', '')).lower()
        }
        
        # Set direct mappings
        direct_mappings = [
            'weekly_trips', 'cancel_rate', 'on_time_rate', 
            'avg_rating', 'earnings_volatility', 'months_on_platform', 'tenure_days'
        ]
        
        for feature in direct_mappings:
            if feature in input_data.columns:
                input_data[feature] = form_data.get(feature, 0.0)
        
        # Handle region one-hot encoding
        region = form_data.get('region', '').lower()
        region_columns = [f'region_{r.capitalize()}' for r in ['east', 'north', 'south', 'west']]
        
        # Set the selected region to 1.0 if it exists
        region_col = f'region_{region.capitalize()}'
        if region_col in input_data.columns:
            input_data[region_col] = 1.0
        
        # Ensure all values are float and handle any potential NaN values
        input_data = input_data.astype(float).fillna(0.0)
        
        # Ensure we only return the columns the model expects, in the right order
        return input_data[expected_features]
        
    except Exception as e:
        st.error(f"Error preparing input data: {str(e)}")
        return None

def calculate_score(data):
    """Calculate score using the trained XGBoost model."""
    try:
        # Load the model and metadata
        model, metadata = get_model()
        
        # Preprocess input data
        input_data = preprocess_input(data, metadata)
        if input_data is None:
            return 0  # Return 0 if input data is invalid
        
        # Make prediction
        if hasattr(model, 'predict_proba'):
            # For classifiers that support probability
            score = model.predict_proba(input_data)[:, 1][0] * 100  # Convert to 0-100 scale
        else:
            # For regressors or models without predict_proba
            score = model.predict(input_data)[0] * 100  # Convert to 0-100 scale
        
        # Ensure score is within 0-100 range
        return max(0, min(100, score))
        
    except Exception as e:
        st.error(f"Error calculating score: {str(e)}")
        # Fallback to mock score if model prediction fails
        return calculate_mock_score(data)

def calculate_mock_score(data):
    """Calculate a mock score based on input data (fallback)."""
    try:
        score = 70  # Base score
        
        # Adjust based on factors
        on_time_rate = float(data.get('on_time_rate', 0)) / 100.0
        cancel_rate = float(data.get('cancel_rate', 0)) / 100.0
        avg_rating = float(data.get('avg_rating', 3.0))
        
        score += (on_time_rate - 0.85) * 100  # 85% is baseline
        score -= cancel_rate * 100  # Reduce for cancellations
        score += (avg_rating - 3.0) * 10  # 3.0 is baseline
        score += min(float(data.get('tenure_months', 0)) / 2, 10)  # Cap tenure bonus
        
        # Cap score between 0-100
        return max(0, min(100, score))
    except:
        return 50  # Default score if something goes wrong

def get_score_band(score):
    """Determine the score band."""
    if score >= 80:
        return "Pre-approved", "✅ Pre-approved", "success"
    elif score >= 60:
        return "Review", "⚠️ Review", "warning"
    else:
        return "Decline", "❌ Decline", "error"

def create_gauge_figure(score):
    """Create a gauge chart for the score."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Nova Score", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "#FF5252"},
                {'range': [40, 70], 'color': "#FFC107"},
                {'range': [70, 100], 'color': "#4CAF50"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=50, r=50, t=100, b=10)
    )
    return fig

def main():
    st.title("Your Nova Score")
    
    # Initialize session state if not exists
    if 'form_submitted' not in st.session_state:
        st.session_state.form_submitted = True  # Set to True to allow demo data to show
    
    # Check if we have form data or need to use demo data
    if not hasattr(st.session_state, 'form_data') or not st.session_state.form_submitted:
        st.warning("No form data found. Using demo data.")
        form_data = {
            'on_time_rate': 92,  # 92%
            'cancel_rate': 8,    # 8%
            'avg_rating': 4.7,
            'tenure_months': 12,
            'weekly_trips': 35,
            'earn_cv': 0.25,     # 25% earnings volatility
            'region': 'east'     # Default region
        }
    else:
        form_data = st.session_state.form_data
    
    # Calculate the score
    with st.spinner("Calculating your score..."):
        try:
            score = calculate_score(form_data)
            band, band_display, band_style = get_score_band(score)
        except Exception as e:
            st.error(f"Error calculating score: {str(e)}")
            score = 50  # Fallback score
            band, band_display, band_style = get_score_band(score)
    
    # Show score and band with animation
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Animate the gauge
        gauge_placeholder = st.empty()
        with gauge_placeholder:
            st.plotly_chart(create_gauge_figure(0), use_container_width=True)
        
        # Animate the score counter
        score_placeholder = st.empty()
        animate_score_counter(score_placeholder, score, duration=1.5)
        
        # Show the final gauge with the actual score
        gauge_placeholder.plotly_chart(create_gauge_figure(score), use_container_width=True)
        
        # Add confetti for high scores
        if score >= 80:
            add_confetti()
            add_success_message("🎉 Congratulations! You've been pre-approved!", delay=2)
    
    with col2:
        st.markdown(f"## {int(score)}/100")
        st.markdown(f"### {band_display}")
        
        # Reason chips
        st.markdown("#### Key Factors")
        reasons = []
        
        if st.session_state.get('cancel_rate', 0) < 0.1:
            reasons.append("✓ Low cancel rate")
        else:
            reasons.append("✗ High cancel rate")
            
        if st.session_state.get('on_time_rate', 0) > 0.9:
            reasons.append("✓ Excellent on-time rate")
        
        if st.session_state.get('avg_rating', 0) >= 4.5:
            reasons.append("✓ High customer ratings")
        
        for reason in reasons:
            st.markdown(f"- {reason}")
    
    # What you did right / What to improve
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✅ What You Did Right")
        st.markdown("""
        - Consistently high ratings from customers
        - Reliable on-time performance
        - Good track record with cancellations
        """)
    
    with col2:
        st.markdown("### 📈 What To Improve")
        st.markdown("""
        - Reduce cancellation rate below 5%
        - Maintain on-time rate above 95%
        - Increase weekly trips for better consistency
        """)
    
    # Action buttons based on score band
    st.markdown("---")
    
    if band == "Pre-approved":
        if st.button("🎉 View Mock Offer", type="primary"):
            st.session_state.show_offer = True
        
        if st.session_state.get('show_offer'):
            with st.expander("🎯 Your Pre-Approved Offer", expanded=True):
                st.markdown("""
                #### 💰 Credit Line: S$5,000
                - **APR:** 12.9%
                - **Term:** 12 months
                - **Monthly Payment:** S$445.00
                - **Disbursement:** Same day
                
                *This is a mock offer for demonstration purposes.*
                """)
                
                if st.button("✓ Accept Offer"):
                    st.success("Offer accepted! Your credit line will be available shortly.")
    
    elif band == "Review":
        if st.button("📄 Request Documents", type="primary"):
            st.session_state.show_docs = True
        
        if st.session_state.get('show_docs'):
            with st.expander("📑 Required Documents", expanded=True):
                st.markdown("""
                Please upload the following documents for verification:
                
                - [ ] Bank statements (last 3 months)
                - [ ] NRIC/Passport copy
                - [ ] Proof of address
                - [ ] Business registration (for merchants)
                
                *Our team will review your application within 2 business days.*
                """)
    
    else:  # Decline
        if st.button("📝 Build Your Score Plan", type="primary"):
            st.session_state.show_plan = True
        
        if st.session_state.get('show_plan'):
            with st.expander("📈 30-Day Score Improvement Plan", expanded=True):
                st.markdown("""
                ### Your Action Plan
                
                #### 1. Reduce Cancellations (Target: <5%)
                - [ ] Accept orders you can fulfill
                - [ ] Plan your schedule better
                
                #### 2. Improve On-Time Performance (Target: >95%)
                - [ ] Allow extra time for deliveries
                - [ ] Check traffic before starting trips
                
                #### 3. Increase Activity
                - [ ] Complete 30+ trips this week
                - [ ] Maintain high ratings (4.5+)
                
                *Check back in 30 days for a score review.*
                """)
    
    # Footer navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("← Back to Home"):
            st.session_state.page = "home"
            st.rerun()
    
    with col2:
        if st.button("Try Another Scenario"):
            st.session_state.page = "partner"
            st.rerun()

# For Streamlit page integration
def result_page():
    return main()

if __name__ == "__main__":
    main()
