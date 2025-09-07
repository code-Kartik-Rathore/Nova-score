"""
Partner Scoring Page with Interactive Form and Animations
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
from pathlib import Path
import json
import time
from datetime import datetime

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model_loader with relative path
try:
    from utils.model_loader import load_xgboost_model, ensure_model_files
except ImportError as e:
    st.error(f"Error importing model_loader: {e}")
    st.error(f"Python path: {sys.path}")
    
    # Fallback model loading
    def load_xgboost_model():
        st.error("Model loading not available. Please check the model files.")
        return None, None
    
    def ensure_model_files():
        return ["Model files not found"]

# Fallback for animations
try:
    from utils.animations import add_loading_animation, add_success_message, pulse_element
except ImportError:
    def add_loading_animation():
        """Dummy function if animations module is not available"""
        pass
    
    def add_success_message(message, icon="✅"):
        """Dummy function for success message"""
        st.success(f"{icon} {message}")
    
    def pulse_element():
        """Dummy function for pulse animation"""
        return ""

# Import model_loader with relative path
try:
    from utils.model_loader import load_xgboost_model, ensure_model_files
except ImportError:
    st.error("Failed to import model_loader. Please ensure the file exists in the utils directory.")

# Page config
st.set_page_config(page_title="Partner Scoring - Nova Score", page_icon="📝")

# Example data for different personas
EXAMPLE_DRIVER_A = {
    "persona": "driver",
    "weekly_trips": 45,
    "hours_online": 50,
    "cancel_rate": 0.08,
    "on_time_rate": 0.92,
    "avg_rating": 4.7,
    "earn_cv": 0.15,
    "tenure_months": 18
}

EXAMPLE_DRIVER_B = {
    "persona": "driver",
    "weekly_trips": 30,
    "hours_online": 35,
    "cancel_rate": 0.12,
    "on_time_rate": 0.85,
    "avg_rating": 4.2,
    "earn_cv": 0.25,
    "tenure_months": 6
}

EXAMPLE_MERCHANT = {
    "persona": "merchant",
    "weekly_trips": 120,
    "hours_online": 168,
    "cancel_rate": 0.05,
    "on_time_rate": 0.95,
    "avg_rating": 4.5,
    "earn_cv": 0.10,
    "gmv_weekly": 5000,
    "refund_rate": 0.03,
    "tenure_months": 24
}

def load_example(example):
    """Load example data into the form."""
    for key, value in example.items():
        if key in st.session_state:
            st.session_state[key] = value

def load_model():
    """Load the XGBoost model and its metadata using the model_loader utility."""
    try:
        # Check for missing model files
        missing_files = ensure_model_files()
        if missing_files:
            st.warning(f"Missing model files: {', '.join(missing_files)}")
        
        # Load the model
        model, metadata = load_xgboost_model()
        
        # Store debug info
        if model is not None and metadata is not None:
            st.session_state.debug_info = {
                'model_loaded': True,
                'model_type': str(type(model)),
                'model_features': metadata.get('features', [])
            }
        
        return model, metadata
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def preprocess_input(data, metadata):
    """Preprocess input data for model prediction."""
    try:
        print("\n=== Raw input data ===")
        print(data)
        
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
        
        print("\n=== Processed form data ===")
        for k, v in form_data.items():
            print(f"{k}: {v} (type: {type(v).__name__})")
        
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
        
        # Set all regions to 0 first
        for col in region_columns:
            if col in input_data.columns:
                input_data[col] = 0.0
        
        # Set the selected region to 1.0 if it exists
        region_col = f'region_{region.capitalize()}'
        if region_col in input_data.columns:
            input_data[region_col] = 1.0
        
        # Ensure all values are float and handle any potential NaN values
        input_data = input_data.astype(float).fillna(0.0)
        
        # Ensure we only return the columns the model expects, in the right order
        input_data = input_data[expected_features]
        
        print("\n=== Final preprocessed data ===")
        print("Shape:", input_data.shape)
        print("Columns:", input_data.columns.tolist())
        print("Values:")
        for col in input_data.columns:
            print(f"  {col}: {input_data[col].iloc[0]}")
            
        return input_data
            
        # Log the prepared input data for debugging
        print("\n=== Final preprocessed data ===")
        print("DataFrame shape:", input_data.shape)
        print("DataFrame columns:", input_data.columns.tolist())
        print("DataFrame values:")
        print(input_data.head())
        
        # Check if DataFrame is empty
        if input_data.empty:
            print("WARNING: Preprocessed DataFrame is empty!")
        else:
            print("First row values:", input_data.iloc[0].to_dict())
            
        return input_data
    except Exception as e:
        st.error(f"Error preprocessing input: {str(e)}")
        return None

def predict_score(data):
    """
    Predict score using the loaded model.
    Handles different model types (XGBoost, scikit-learn, etc.)
    """
    try:
        # Load model and metadata
        model, metadata = load_model()
        if model is None or metadata is None:
            st.error("Failed to load model or metadata")
            return None
            
        # Preprocess input data
        input_data = preprocess_input(data, metadata)
        if input_data is None or input_data.empty:
            st.error("No valid input data for prediction")
            return None
            
        # Debug info
        print("\n=== Model Prediction Debug ===")
        print(f"Model type: {type(model).__name__}")
        print("Input data shape:", input_data.shape)
        print("Input columns:", input_data.columns.tolist())
        
        # Handle different model types
        model_type = metadata.get('model_type', '').lower()
        
        # Prepare input features
        if hasattr(model, 'feature_names_in_'):
            # Ensure correct feature order and add missing features
            missing_cols = set(model.feature_names_in_) - set(input_data.columns)
            for col in missing_cols:
                input_data[col] = 0.0  # Add missing columns with default value 0
            input_data = input_data[model.feature_names_in_]
        
        # Convert to numpy array if needed
        if hasattr(input_data, 'values'):
            input_array = input_data.values
        else:
            input_array = input_data
            
        # Ensure 2D array
        if len(input_array.shape) == 1:
            input_array = input_array.reshape(1, -1)
        
        print("Final input shape:", input_array.shape)
        
        # Make prediction based on model type
        if 'xgb' in model_type.lower() or 'xgboost' in model_type.lower():
            import xgboost as xgb
            # Handle XGBoost Classifier (scikit-learn API)
            if hasattr(model, 'predict_proba'):
                try:
                    # Get probability of positive class (assuming binary classification)
                    prediction = model.predict_proba(input_array)[:, 1]
                    print("Used predict_proba for XGBoost Classifier")
                except Exception as e:
                    print(f"Error with predict_proba: {e}, falling back to predict")
                    prediction = model.predict(input_array)
            # Handle XGBoost Booster
            elif hasattr(model, 'get_booster'):
                try:
                    dmatrix = xgb.DMatrix(input_array, feature_names=input_data.columns.tolist())
                    prediction = model.predict(dmatrix)
                    print("Used Booster predict")
                except Exception as e:
                    print(f"Error with Booster predict: {e}")
                    return None
            else:
                # Fallback to predict if neither predict_proba nor get_booster is available
                prediction = model.predict(input_array)
                print("Used default predict")
        
        # Handle scikit-learn models
        elif hasattr(model, 'predict_proba'):
            prediction = model.predict_proba(input_array)[:, 1]  # Get probability of positive class
        elif hasattr(model, 'predict'):
            prediction = model.predict(input_array)
        else:
            st.error("Model does not support prediction")
            return None
        
        print("Raw prediction:", prediction)
        
        # Handle different prediction formats
        if isinstance(prediction, (list, np.ndarray)):
            if len(prediction.shape) > 1 and prediction.shape[1] > 1:  # Multi-class probabilities
                score = int(round(prediction[0][1] * 600 + 300))  # Assuming binary classification
            else:  # Single column of probabilities or regression values
                score = int(round(prediction[0] * 600 + 300))
        else:  # Single value
            score = int(round(prediction * 600 + 300))
        
        # Ensure score is within bounds
        score = max(300, min(900, score))
        print(f"Final score: {score}")
        
        return score
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        import traceback
        print(f"Prediction error: {traceback.format_exc()}")
        return None

def calculate_preview_score(data):
    """Calculate a preview score based on form inputs with dynamic updates."""
    if not data:
        return 500  # Default score
    
    try:
        # First try to use the model for prediction
        model, metadata = load_model()
        if model is not None and metadata is not None:
            # Preprocess input data
            preprocessed_data = preprocess_input(data, metadata)
            if preprocessed_data is not None:
                # Get prediction
                if hasattr(model, 'predict_proba'):
                    # For binary classification models
                    prediction = model.predict_proba(preprocessed_data)[:, 1]
                    score = int(round(prediction[0] * 600 + 300))  # Scale to 300-900
                elif hasattr(model, 'predict'):
                    # For regression models
                    prediction = model.predict(preprocessed_data)
                    score = int(round(prediction[0] * 600 + 300))  # Scale to 300-900
                else:
                    raise Exception("Model does not have predict or predict_proba method")
                
                # Ensure score is within bounds
                return max(300, min(900, score))
        
        # Fallback to simple calculation if model prediction fails
        score = 500  # Base score
        
        weights = {
            "weekly_trips": 2.0,      # More impact from weekly trips
            "hours_online": 1.5,      # More impact from hours online
            "cancel_rate": -8.0,      # More negative impact from cancellations
            "on_time_rate": 5.0,      # More positive impact from on-time rate
            "avg_rating": 60,         # More impact from rating
            "earn_cv": -150,          # More impact from earning volatility
            "tenure_months": 0.5,     # Slight impact from tenure
        }
        
        # Additional weights for merchants
        if data.get("persona") == "merchant":
            weights.update({
                "gmv_weekly": 0.0005,  # Impact from GMV
                "refund_rate": -5.0,    # Negative impact from refunds
            })
        
        # Calculate weighted score
        for key, weight in weights.items():
            value = data.get(key, 0) or 0  # Handle None values
            if key == "avg_rating" and value is not None:
                score += (value - 1) * weight  # Scale 1-5 to 0-4
            elif key == "earn_cv" and value is not None and value > 0:
                score += (1 / (value + 0.01)) * abs(weight)  # Invert for CV (lower is better)
            elif value is not None:
                score += float(value) * weight
        
        # Apply non-linear scaling for better distribution
        score = 1 / (1 + 0.001 * (900 - score)) * 600 + 300
        
        # Ensure score is within bounds and round to nearest 10
        return int(round(max(300, min(900, score)) / 10) * 10)
        
    except Exception as e:
        st.error(f"Error calculating score: {str(e)}")
        return 500  # Return default score on error

def calculate_score(data):
    """
    Calculate the Nova Score using the trained XGBoost model.
    Falls back to a simple heuristic if model prediction fails.
    """
    try:
        # Try to use the XGBoost model first
        model, metadata = load_model()
        if model is not None and metadata is not None:
            # Preprocess input data
            input_data = preprocess_input(data, metadata)
            if input_data is not None:
                # Make prediction
                if hasattr(model, 'predict_proba'):
                    # For classifiers that support probability
                    score = model.predict_proba(input_data)[:, 1][0] * 1000  # Scale to 0-1000
                else:
                    # For regressors or models without predict_proba
                    score = model.predict(input_data)[0] * 1000  # Scale to 0-1000
                
                # Ensure score is within bounds and round to nearest 10
                score = max(300, min(900, score))
                return int(round(score / 10) * 10)
    
    except Exception as e:
        st.warning(f"Model prediction failed: {str(e)}. Falling back to heuristic scoring.")
    
    # Fallback to heuristic scoring if model prediction fails
    try:
        # Base score
        score = 500
        
        # Default weights for all user types
        weights = {
            'weekly_trips': 0.2,
            'cancel_rate': -0.5,
            'on_time_rate': 0.3,
            'avg_rating': 20,
            'tenure_months': 0.15,
            'earn_cv': -0.4
        }
        
        # Apply weights to each factor
        for factor, weight in weights.items():
            if factor in data and data[factor] is not None:
                value = data[factor]
                # Convert percentages to decimal if needed
                if factor in ['cancel_rate', 'on_time_rate']:
                    value = value / 100.0 if value > 1 else value
                
                # Apply weight with non-linear scaling for some factors
                if factor == 'avg_rating':
                    score += (value ** 2) * (weight / 4)  # Non-linear scaling for ratings
                elif factor == 'cancel_rate':
                    score += (value * weight * 15)  # Higher penalty for negative factors
                else:
                    score += value * weight * 10  # Standard scaling
        
        # Ensure score is within bounds and round to nearest 10
        score = max(300, min(900, score))
        return int(round(score / 10) * 10)
        
    except Exception as e:
        st.error(f"Error in score calculation: {str(e)}")
        return 500  # Default score if all else fails
        return 500  # Return default score on error

def update_score_display(score):
    """Update the score display with appropriate styling."""
    # Determine score category
    if score >= 700:
        color = "#4CAF50"  # Green
        category = "Excellent"
    elif score >= 550:
        color = "#8BC34A"  # Light Green
        category = "Good"
    elif score >= 400:
        color = "#FFC107"  # Amber
        category = "Average"
    else:
        color = "#F44336"  # Red
        category = "Needs Improvement"
    
    # Display score with styling
    st.markdown(f"""
    <div style="
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
        border-left: 5px solid {color};
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    ">
        <div style="font-size: 1rem; color: #6c757d; margin-bottom: 0.5rem;">Your Nova Score</div>
        <div style="font-size: 3rem; font-weight: bold; color: {color}; line-height: 1;">
            {score}
        </div>
        <div style="
            display: inline-block;
            background: {color}22;
            color: {color};
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.9rem;
            margin-top: 0.5rem;
            font-weight: 500;
        ">{category}</div>
        <div style="font-size: 0.9rem; color: #6c757d; margin-top: 1rem;">
            Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        </div>
    </div>
    """, unsafe_allow_html=True)

def validate_form():
    """Validate form inputs and return any errors."""
    errors = []
    
    required_fields = [
        ("weekly_trips", "Weekly Trips"),
        ("hours_online", "Hours Online"),
        ("cancel_rate", "Cancel Rate"),
        ("on_time_rate", "On-time Rate"),
        ("avg_rating", "Average Rating"),
        ("tenure_months", "Tenure"),
        ("earn_cv", "Earnings Volatility")
    ]
    
    for field, label in required_fields:
        if field not in st.session_state or st.session_state.get(field) is None:
            errors.append(f"{label} is required")
    
    if "weekly_trips" in st.session_state and st.session_state.weekly_trips < 0:
        errors.append("Weekly trips cannot be negative")
        
    if "hours_online" in st.session_state and (st.session_state.hours_online < 0 or st.session_state.hours_online > 168):
        errors.append("Hours online must be between 0 and 168")
        
    if "cancel_rate" in st.session_state and (st.session_state.cancel_rate < 0 or st.session_state.cancel_rate > 100):
        errors.append("Cancel rate must be between 0% and 100%")
        
    if "on_time_rate" in st.session_state and (st.session_state.on_time_rate < 0 or st.session_state.on_time_rate > 100):
        errors.append("On-time rate must be between 0% and 100%")
        
    if "avg_rating" in st.session_state and (st.session_state.avg_rating < 1 or st.session_state.avg_rating > 5):
        errors.append("Average rating must be between 1 and 5")
        
    if "tenure_months" in st.session_state and st.session_state.tenure_months < 0:
        errors.append("Tenure cannot be negative")
        
    if "earn_cv" in st.session_state and st.session_state.earn_cv < 0:
        errors.append("Earnings volatility cannot be negative")
    
    return errors

def submit_form():
    """Handle form submission with validation and score calculation."""
    errors = validate_form()
    
    if errors:
        for error in errors:
            st.error(error)
        st.session_state.page = "partner"
    else:
        # Collect all form data
        form_data = {
            'weekly_trips': st.session_state.get('weekly_trips', 0),
            'hours_online': st.session_state.get('hours_online', 0),
            'cancel_rate': st.session_state.get('cancel_rate', 0),
            'on_time_rate': st.session_state.get('on_time_rate', 0),
            'avg_rating': st.session_state.get('avg_rating', 0),
            'tenure_months': st.session_state.get('tenure_months', 0),
            'earn_cv': st.session_state.get('earn_cv', 0),
            'region': st.session_state.get('region', 'east'),
            'persona': st.session_state.get('persona', 'driver')
        }
        
        # Calculate score
        score = calculate_score(form_data)
        
        # Store all data in session state for the result page
        st.session_state.form_data = form_data
        st.session_state.current_score = score
        st.session_state.score_calculated = True
        st.session_state.page = "result"
        
        # Force a rerun to update the page
        st.experimental_rerun()
    
    st.rerun()

def main():
    st.title("📝 Partner Scoring")
    
    if 'form_data' not in st.session_state:
        st.session_state.form_data = {}
    
    persona = st.radio(
        "Partner Type", 
        ["Driver", "Merchant"], 
        index=1 if st.session_state.get('persona') == 'Merchant' else 0,
        horizontal=True,
        key='persona_radio',
        on_change=lambda: st.session_state.update({
            'persona': st.session_state.persona_radio.lower()
        })
    )
    
    st.session_state.persona = persona.lower()
    
    st.markdown("### Quick Start Examples")
    st.markdown("<div style='margin-bottom: 1rem;'>Try one of these examples or fill in your own data:</div>", 
                unsafe_allow_html=True)
    
    examples = [
        {"title": "🚀 High Performer", "description": "Top-rated driver with excellent metrics", "data": EXAMPLE_DRIVER_A},
        {"title": "📊 Average Performer", "description": "Typical driver with room for improvement", "data": EXAMPLE_DRIVER_B},
        {"title": "🏪 Merchant", "description": "Sample merchant profile with business metrics", "data": EXAMPLE_MERCHANT}
    ]
    
    cols = st.columns(3)
    for idx, example in enumerate(examples):
        with cols[idx]:
            if st.button(
                f"**{example['title']}**\n\n{example['description']}",
                key=f"example_{idx}",
                use_container_width=True,
                help=f"Load {example['title']} example"
            ):
                with st.spinner("Loading example data..."):
                    load_example(example['data'])
                    st.rerun()
    
    with st.form("scoring_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            weekly_trips = st.number_input(
                "Weekly Trips", 
                min_value=0, 
                key="weekly_trips",
                help="Number of trips completed per week",
                step=1
            )
            
            hours_online = st.number_input(
                "Hours Online", 
                min_value=0, 
                max_value=168,
                key="hours_online",
                help="Total hours active on the platform per week (max 168)",
                step=1
            )
            
            st.write("Cancel Rate")
            cancel_rate = st.slider(
                "", 
                min_value=0.0, 
                max_value=100.0, 
                value=st.session_state.get("cancel_rate", 0.0),
                step=0.1, 
                format="%.1f%%", 
                key="cancel_rate",
                help="Percentage of trips cancelled by the partner",
                label_visibility="collapsed"
            )
            
            st.write("On-time Rate")
            on_time_rate = st.slider(
                "",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state.get("on_time_rate", 95.0),
                step=0.1,
                format="%.1f%%",
                key="on_time_rate",
                help="Percentage of on-time deliveries",
                label_visibility="collapsed"
            )
        
        with col2:
            st.write("Average Rating")
            avg_rating = st.slider(
                "", 
                min_value=1.0, 
                max_value=5.0, 
                value=st.session_state.get("avg_rating", 4.0),
                step=0.1, 
                key="avg_rating",
                help="Average rating from customers (1-5 stars)",
                label_visibility="collapsed"
            )
            
            st.write("Earnings Volatility (CV)")
            earn_cv = st.slider(
                "",
                min_value=0.0, 
                max_value=1.0, 
                value=st.session_state.get("earn_cv", 0.3),
                step=0.01,
                key="earn_cv", 
                help="Coefficient of variation in weekly earnings (lower is better)",
                label_visibility="collapsed"
            )
            
            if persona == "Merchant":
                st.write("Weekly GMV (S$)")
                gmv_weekly = st.number_input(
                    "", 
                    min_value=0, 
                    key="gmv_weekly",
                    help="Gross merchandise value per week",
                    step=100,
                    label_visibility="collapsed"
                )
                
                st.write("Refund Rate")
                refund_rate = st.slider(
                    "",
                    min_value=0.0, 
                    max_value=100.0, 
                    value=st.session_state.get("refund_rate", 5.0),
                    step=0.1,
                    format="%.1f%%", 
                    key="refund_rate",
                    help="Percentage of orders refunded",
                    label_visibility="collapsed"
                )
            
            st.write("Tenure (months)")
            tenure_months = st.number_input(
                "", 
                min_value=0, 
                max_value=1200,
                key="tenure_months",
                help="Months active on the platform (max 100 years)",
                step=1,
                label_visibility="collapsed"
            )
        
        current_data = {
            "persona": st.session_state.get("persona", "driver"),
            "weekly_trips": st.session_state.get("weekly_trips", 0),
            "hours_online": st.session_state.get("hours_online", 0),
            "cancel_rate": st.session_state.get("cancel_rate", 0),
            "on_time_rate": st.session_state.get("on_time_rate", 0),
            "avg_rating": st.session_state.get("avg_rating", 0),
            "earn_cv": st.session_state.get("earn_cv", 0.3),
            "tenure_months": st.session_state.get("tenure_months", 0),
            "gmv_weekly": st.session_state.get("gmv_weekly", 0),
            "refund_rate": st.session_state.get("refund_rate", 0)
        }
        
        preview_score = calculate_preview_score(current_data)
        score_color = "#4CAF50"
        if preview_score < 500:
            score_color = "#F44336"
        elif preview_score < 700:
            score_color = "#FFC107"
            
        st.markdown(
            f"""
            <div style="
                background: #f8f9fa;
                border-radius: 10px;
                padding: 1rem;
                margin: 1rem 0;
                text-align: center;
                border-left: 5px solid {score_color};
            ">
                <div style="font-size: 0.9rem; color: #6c757d;">Estimated Nova Score</div>
                <div style="font-size: 2.5rem; font-weight: bold; color: {score_color};">
                    {preview_score}
                </div>
                <div style="font-size: 0.8rem; color: #6c757d;">
                    Based on current inputs • Updates in real-time
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            back_btn = st.form_submit_button(
                "← Back to Home",
                type="secondary",
                use_container_width=True
            )
            
        with col2:
            submit_btn = st.form_submit_button(
                "🚀 Get Full Nova Score", 
                type="primary",
                use_container_width=True,
                help="Submit for a detailed score analysis"
            )
            
        if submit_btn:
            errors = validate_form()
            if not errors:
                with st.spinner("🔍 Analyzing reliability metrics..."):
                    time.sleep(0.5)
                with st.spinner("📊 Evaluating quality indicators..."):
                    time.sleep(0.5)
                with st.spinner("💰 Calculating earnings stability..."):
                    time.sleep(0.5)
                with st.spinner("🤖 Running model prediction..."):
                    # Calculate final score using the model
                    final_score = predict_score(current_data)
                    if final_score is not None:
                        st.session_state.final_score = final_score
                        st.session_state.score_calculated = True
                    time.sleep(0.5)
                
                if 'final_score' in st.session_state:
                    submit_form()
                else:
                    st.error("Failed to calculate score. Please try again.")
        
        if back_btn:
            st.session_state.page = "home"
            st.rerun()
    
    st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
