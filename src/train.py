"""
Model training script for Nova Score project.
Trains and evaluates credit scoring models (Logistic Regression and XGBoost).
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import shap

# Ensure models directory exists
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

def load_data(data_path="data/partners.csv"):
    """Load and preprocess the dataset."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Convert date to datetime and calculate tenure in days
    if 'join_date' in df.columns:
        df['join_date'] = pd.to_datetime(df['join_date'])
        df['tenure_days'] = (datetime.now() - df['join_date']).dt.days
    
    # Define features and target
    feature_cols = [
        'months_on_platform', 'weekly_trips', 'cancel_rate', 
        'on_time_rate', 'avg_rating', 'earnings_volatility'
    ]
    
    # One-hot encode categorical variables
    categorical_cols = ['region']
    if 'region' in df.columns:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        # Update feature columns to include one-hot encoded columns
        feature_cols = [col for col in df.columns 
                       if col not in ['partner_id', 'join_date', 'repay_on_time'] 
                       and not col.startswith('logit') 
                       and not col.startswith('prob_')]
    
    X = df[feature_cols]
    y = df['repay_on_time']
    
    return X, y, feature_cols

def train_models(X_train, y_train, feature_names, seed=42):
    """Train and return multiple models."""
    print("\nTraining models...")
    models = {}
    
    # Standardize features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 1. Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(
        random_state=seed, 
        max_iter=1000,
        class_weight='balanced'
    )
    lr.fit(X_train_scaled, y_train)
    models['logistic_regression'] = {
        'model': lr,
        'scaler': scaler,
        'feature_names': feature_names
    }
    
    # 2. XGBoost with monotonicity constraints
    print("Training XGBoost with monotonicity constraints...")
    # Define monotonicity constraints (1 = increasing, -1 = decreasing, 0 = no constraint)
    monotone_constraints = []
    for feature in feature_names:
        if feature == 'weekly_trips':
            monotone_constraints.append(1)  # More trips should increase score
        else:
            monotone_constraints.append(0)  # No constraint for other features
    
    xgb = XGBClassifier(
        random_state=seed,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=sum(y_train == 0) / sum(y_train == 1),  # Handle class imbalance
        use_label_encoder=False,
        eval_metric='logloss',
        monotone_constraints=tuple(monotone_constraints)  # Add monotonicity constraints
    )
    xgb.fit(X_train, y_train)
    models['xgboost'] = {
        'model': xgb,
        'feature_names': feature_names
    }
    
    return models

def evaluate_models(models, X_test, y_test):
    """Evaluate models and return metrics."""
    print("\nEvaluating models...")
    results = {}
    
    for name, model_data in models.items():
        print(f"\n--- {name.upper()} ---")
        
        if name == 'logistic_regression':
            X_test_processed = model_data['scaler'].transform(X_test)
        else:
            X_test_processed = X_test
        
        model = model_data['model']
        
        # Make predictions
        y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        auroc = roc_auc_score(y_test, y_pred_proba)
        brier = brier_score_loss(y_test, y_pred_proba)
        
        # Store results
        results[name] = {
            'auroc': auroc,
            'brier': brier,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Print metrics
        print(f"AUROC: {auroc:.4f}")
        print(f"Brier Score: {brier:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    
    return results

def save_models(models, results):
    """Save trained models and metadata."""
    print("\nSaving models and metadata...")
    
    for name, model_data in models.items():
        # Save model
        model_path = MODEL_DIR / f"{name}.joblib"
        joblib.dump(model_data, model_path)
        
        # Save metadata
        metadata = {
            'model_name': name,
            'training_date': datetime.now().isoformat(),
            'metrics': results.get(name, {}),
            'feature_importance': {}
        }
        
        # Add feature importance if available
        if name == 'xgboost':
            model = model_data['model']
            feature_importances = model.feature_importances_
            metadata['feature_importance'] = dict(zip(
                model_data['feature_names'], 
                feature_importances.tolist()
            ))
        
        # Save metadata
        meta_path = MODEL_DIR / f"{name}_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved {name} model to {model_path}")
        print(f"Saved {name} metadata to {meta_path}")

def generate_shap_analysis(models, X_train, X_test, feature_names):
    """Generate and save SHAP analysis for models."""
    print("\nGenerating SHAP analysis...")
    
    for name, model_data in models.items():
        if name != 'xgboost':  # Skip non-tree models for SHAP
            continue
            
        print(f"Generating SHAP values for {name}...")
        model = model_data['model']
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_test)
        
        # Save SHAP values
        shap_values_path = MODEL_DIR / f"{name}_shap_values.npy"
        np.save(shap_values_path, shap_values)
        
        # Save SHAP explainer
        explainer_path = MODEL_DIR / f"{name}_explainer.joblib"
        joblib.dump(explainer, explainer_path)
        
        print(f"Saved SHAP values to {shap_values_path}")
        print(f"Saved SHAP explainer to {explainer_path}")

def main():
    """Main function to train and evaluate models."""
    # Load and prepare data
    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train models
    models = train_models(X_train, y_train, feature_names)
    
    # Evaluate models
    results = evaluate_models(models, X_test, y_test)
    
    # Generate SHAP analysis
    generate_shap_analysis(models, X_train, X_test, feature_names)
    
    # Save models and metadata
    save_models(models, results)
    
    print("\nTraining and evaluation complete!")

if __name__ == "__main__":
    main()
