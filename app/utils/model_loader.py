"""
Model loading utility for Nova Score application.
Handles loading models in both development and production environments.
"""
import os
import sys
import joblib
import json
import requests
from pathlib import Path
import streamlit as st

# Base directory for the project
BASE_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / "models"

# Model file URLs (replace with your actual model file URLs if needed)
MODEL_URLS = {
    "xgboost.joblib": "https://github.com/code-Kartik-Rathore/Nova-score/raw/main/models/xgboost.joblib",
    "xgboost_meta.json": "https://github.com/code-Kartik-Rathore/Nova-score/raw/main/models/xgboost_meta.json",
    "xgboost_shap_values.npy": "https://github.com/code-Kartik-Rathore/Nova-score/raw/main/models/xgboost_shap_values.npy",
    "logistic_regression.joblib": "https://github.com/code-Kartik-Rathore/Nova-score/raw/main/models/logistic_regression.joblib",
    "logistic_regression_meta.json": "https://github.com/code-Kartik-Rathore/Nova-score/raw/main/models/logistic_regression_meta.json"
}

def ensure_models_directory():
    """Ensure the models directory exists."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR

def download_file(url, destination):
    """Download a file from a URL to the specified destination."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        st.error(f"Error downloading {url}: {str(e)}")
        return False

def ensure_model_files():
    """Ensure all required model files exist, download if missing."""
    ensure_models_directory()
    missing_files = []
    
    for filename, url in MODEL_URLS.items():
        file_path = MODELS_DIR / filename
        if not file_path.exists():
            st.warning(f"Model file not found: {filename}")
            if st.button(f"Download {filename}"):
                with st.spinner(f"Downloading {filename}..."):
                    if download_file(url, file_path):
                        st.success(f"Successfully downloaded {filename}")
                        st.experimental_rerun()
                    else:
                        st.error(f"Failed to download {filename}")
                missing_files.append(filename)
            else:
                missing_files.append(filename)
    
    return missing_files

def get_model_path(filename):
    """Get the path to a model file, downloading it if necessary."""
    file_path = MODELS_DIR / filename
    
    # If file doesn't exist, try to download it
    if not file_path.exists() and filename in MODEL_URLS:
        with st.spinner(f"Downloading missing model file: {filename}..."):
            if download_file(MODEL_URLS[filename], file_path):
                st.success(f"Successfully downloaded {filename}")
            else:
                st.error(f"Failed to download {filename}")
    
    return file_path

def load_xgboost_model():
    """
    Load the XGBoost model and its metadata.
    Handles different model formats including dictionary-wrapped models.
    """
    try:
        model_path = get_model_path("xgboost.joblib")
        meta_path = get_model_path("xgboost_meta.json")
        
        if not model_path.exists() or not meta_path.exists():
            missing = []
            if not model_path.exists():
                missing.append("xgboost.joblib")
            if not meta_path.exists():
                missing.append("xgboost_meta.json")
            st.error(f"Missing required model files: {', '.join(missing)}")
            ensure_model_files()
            return None, None
        
        with st.spinner("Loading XGBoost model..."):
            # Load the model
            loaded = joblib.load(model_path)
            
            # Handle different model storage formats
            if isinstance(loaded, dict):
                if 'model' in loaded:
                    model = loaded['model']
                    feature_names = loaded.get('feature_names', [])
                else:
                    st.error("Invalid model format: dictionary missing 'model' key")
                    return None, None
            else:
                model = loaded
                feature_names = []
            
            # Load metadata
            try:
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
            except Exception as e:
                st.error(f"Error loading metadata: {str(e)}")
                metadata = {}
            
            # Update metadata with model info
            model_type = type(model).__name__
            metadata.update({
                'model_type': model_type,
                'feature_names': feature_names or getattr(model, 'feature_names_in_', []),
                'model_class': model_type
            })
            
            # Debug info
            print(f"Loaded model type: {model_type}")
            print(f"Available methods: {[m for m in dir(model) if not m.startswith('_')]}")
            
            return model, metadata
            
    except Exception as e:
        st.error(f"Error loading XGBoost model: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None
        return None, None

def load_logistic_regression_model():
    """Load the Logistic Regression model and its metadata."""
    try:
        model_path = get_model_path("logistic_regression.joblib")
        meta_path = get_model_path("logistic_regression_meta.json")
        
        if not model_path.exists() or not meta_path.exists():
            missing = []
            if not model_path.exists():
                missing.append(f"logistic_regression.joblib")
            if not meta_path.exists():
                missing.append(f"logistic_regression_meta.json")
            st.error(f"Missing required model files: {', '.join(missing)}")
            ensure_model_files()
            return None, None
        
        with st.spinner("Loading Logistic Regression model..."):
            model = joblib.load(model_path)
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            return model, metadata
            
    except Exception as e:
        st.error(f"Error loading Logistic Regression model: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None
