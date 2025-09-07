


"""
Partner Scoring Page with Interactive Form and Animations
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import json
from app.utils.animations import add_loading_animation, add_success_message, pulse_element
import time
from datetime import datetime

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
    """Load the XGBoost model and its metadata."""
    try:
        # Use absolute path to the models directory
        base_dir = Path("/Users/kartikrathore/Documents/Grab-nova-score/nova-score")
        model_path = base_dir / "models" / "xgboost.joblib"
        meta_path = base_dir / "models" / "xgboost_meta.json"
        
        # Debug: Print paths to verify
        print(f"Looking for model at: {model_path}")
        print(f"Looking for metadata at: {meta_path}")
        print(f"Model exists: {model_path.exists()}")
        print(f"Metadata exists: {meta_path.exists()}")
        
        if not model_path.exists() or not meta_path.exists():
            st.error(f"Model files not found. Please ensure the model files exist at:\n{model_path}\n{meta_path}")
            return None, None
            
        model = joblib.load(model_path)
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
            
        return model, metadata
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None

def preprocess_input(data, metadata):
    """Preprocess input data for model prediction."""
    try:
        print("\n=== Raw input data ===")
        print(data)
        
        # Get feature names from feature_importance if available, otherwise use default
        feature_names = list(metadata.get('feature_importance', {}).keys())
        print("\nFeature names from metadata:", feature_names)
        
        # If no feature_importance, use the default expected features
        if not feature_names:
            feature_names = [
                'months_on_platform', 'weekly_trips', 'cancel_rate', 
                'on_time_rate', 'avg_rating', 'earnings_volatility',
                'region_East', 'region_North', 'region_South', 'region_West'
            ]
            print("Using default feature names:", feature_names)
        
        # Initialize with one row of zeros
        input_data = pd.DataFrame(0.0, index=[0], columns=feature_names)
        
        # Map form inputs to model features
        feature_mapping = {
            'weekly_trips': 'weekly_trips',
            'cancel_rate': 'cancel_rate',
            'on_time_rate': 'on_time_rate',
            'avg_rating': 'avg_rating',
            'earn_cv': 'earnings_volatility',
            'tenure_months': 'months_on_platform',
            'region': 'region'  # Will be one-hot encoded
        }
        print("\nFeature mapping:", feature_mapping)
        
        print("\n=== Processing form data ===")
        print("Available form keys:", list(data.keys()))
        
        # Map and set values from form data
        for form_key, model_key in feature_mapping.items():
            print(f"\nProcessing form key: {form_key} -> model key: {model_key}")
            if form_key in data and data[form_key] is not None:
                print(f"Found value for {form_key}: {data[form_key]}")
                if model_key in input_data.columns:
                    try:
                        input_data.at[0, model_key] = float(data[form_key])
                        print(f"Set {model_key} = {input_data.at[0, model_key]}")
                    except (ValueError, TypeError) as e:
                        print(f"Error converting {form_key} to float: {e}")
            
            # Handle one-hot encoding for region
            if form_key == 'region' and 'region' in data and data['region'] is not None:
                region = str(data['region']).lower()
                print(f"Processing region: {region}")
                for col in input_data.columns:
                    if col.startswith('region_'):
                        region_value = 1.0 if col.endswith(region.capitalize()) else 0.0
                        input_data.at[0, col] = region_value
                        print(f"Set {col} = {region_value}")
        
        # Ensure all values are numeric
        for col in input_data.columns:
            input_data[col] = pd.to_numeric(input_data[col], errors='coerce').fillna(0.0)
            
        print("\n=== Final input data ===")
        print("DataFrame shape:", input_data.shape)
        print("DataFrame columns:", input_data.columns.tolist())
        print("DataFrame values:")
        print(input_data.head())
        
        # Ensure all numeric columns are float
        for col in input_data.select_dtypes(include=['int64']).columns:
            input_data[col] = input_data[col].astype(float)
        
        # Convert all columns to float to ensure consistency
        for col in input_data.columns:
            input_data[col] = input_data[col].astype(float)
            
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
    """Predict score using the XGBoost model."""
    try:
        model_dict, metadata = load_model()
        if model_dict is None or metadata is None:
            return None
            
        # Preprocess input data
        input_data = preprocess_input(data, metadata)
        if input_data is None or input_data.empty:
            st.error("No valid input data for prediction")
            return None
            
        # Debug: Print input data shape and sample
        print("\nInput data shape:", input_data.shape)
        print("Input data columns:", input_data.columns.tolist())
        print("Input data values:", input_data.values)
        
        # The model might be a dictionary with 'model' key or the model itself
        model = model_dict.get('model', model_dict) if isinstance(model_dict, dict) else model_dict
        
        # Ensure input is 2D array with correct feature order
        if hasattr(model, 'feature_names_in_'):
            # Reorder columns to match model's expected feature order
            missing_cols = set(model.feature_names_in_) - set(input_data.columns)
            if missing_cols:
                for col in missing_cols:
                    input_data[col] = 0.0  # Add missing columns with default value 0
            input_data = input_data[model.feature_names_in_]
        
        # Convert to numpy array if it's a DataFrame
        if hasattr(input_data, 'values'):
            input_array = input_data.values
        else:
            input_array = input_data
            
        # Ensure 2D array
        if len(input_array.shape) == 1:
            input_array = input_array.reshape(1, -1)
            
        print("\nFinal input array shape:", input_array.shape)
        
        # Make prediction - handle both pipeline and raw model
        if hasattr(model, 'predict_proba'):
            # For models with predict_proba
            proba = model.predict_proba(input_array)
            print("Prediction probabilities:", proba)
            score = int(round(proba[0][1] * 600 + 300))  # Scale to 300-900 range
        elif hasattr(model, 'predict'):
            # For models with predict
            prediction = model.predict(input_array)
            print("Raw prediction:", prediction)
            score = int(round(prediction[0] * 600 + 300))  # Scale to 300-900 range
        else:
            st.error("Model does not have predict or predict_proba method")
            return None
        
        # Ensure score is within bounds
        score = max(300, min(900, score))
        print("Final score:", score)
        
        return score
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def calculate_preview_score(data):
    """Calculate a preview score based on form inputs."""
    if not data:
        return 0
    
    # Use the model for prediction if available, otherwise fall back to simple calculation
    score = predict_score(data)
    
    if score is None:
        # Fallback to simple calculation if model prediction fails
        score = 500  # Base score
        
        weights = {
            "weekly_trips": 0.2,
            "hours_online": 0.1,
            "cancel_rate": -0.5,
            "on_time_rate": 0.3,
            "avg_rating": 50,
            "earn_cv": -100,
            "tenure_months": 0.1,
        }
        
        if data.get("persona") == "merchant":
            weights.update({
                "gmv_weekly": 0.0001,
                "refund_rate": -0.3,
            })
        
        for key, weight in weights.items():
            value = data.get(key, 0)
            if value is not None:
                if key == "avg_rating":
                    score += (value - 1) * weight
                elif key == "earn_cv" and value > 0:
                    score += (1 / (value + 0.1)) * abs(weight)
                else:
                    score += value * weight
        
        if "weekly_trips" in data and data["weekly_trips"] > 0:
            trips = data["weekly_trips"]
            score += 100 * (1 - 1 / (1 + trips/50))
        
        score = max(300, min(900, score))
    
    return int(round(score / 10) * 10)

def calculate_score(data):
    """Calculate the Nova Score based on input parameters."""
    # Base score
    score = 500
    
    # Weights for different factors
    weights = {
        'weekly_trips': 0.2,
        'hours_online': 0.1,
        'cancel_rate': -0.5,
        'on_time_rate': 0.3,
        'avg_rating': 20,
        'tenure_months': 0.15,
        'earn_cv': -0.4
    }
    
    # Apply weights to each factor
    for factor, weight in weights.items():
        if factor in data:
            value = data[factor]
            if factor == 'cancel_rate' or factor == 'on_time_rate':
                value = value / 100.0  # Convert percentage to decimal
            score += value * weight * 10  # Scale the impact
    
    # Apply non-linear scaling for certain factors
    if 'avg_rating' in data:
        score += (data['avg_rating'] ** 2) * 5
    
    if 'weekly_trips' in data and data['weekly_trips'] > 0:
        score += 100 * (1 - 1 / (1 + data['weekly_trips']/50))
    
    # Ensure score is within bounds
    score = max(300, min(900, score))
    
    return int(round(score))

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
        # Collect form data
        form_data = {
            'weekly_trips': st.session_state.get('weekly_trips', 0),
            'hours_online': st.session_state.get('hours_online', 0),
            'cancel_rate': st.session_state.get('cancel_rate', 0),
            'on_time_rate': st.session_state.get('on_time_rate', 0),
            'avg_rating': st.session_state.get('avg_rating', 0),
            'tenure_months': st.session_state.get('tenure_months', 0),
            'earn_cv': st.session_state.get('earn_cv', 0)
        }
        
        # Calculate score
        score = calculate_score(form_data)
        
        # Store in session state
        st.session_state.current_score = score
        st.session_state.score_calculated = True
        st.session_state.page = "result"
    
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
