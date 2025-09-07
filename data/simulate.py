"""
Data simulation for Nova Score project.
Generates synthetic partner data with realistic patterns for model development.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from pathlib import Path

# Ensure the data directory exists
Path("data").mkdir(exist_ok=True)

def generate_synthetic_partners(n=10000, seed=42):
    """Generate synthetic partner data with realistic patterns."""
    np.random.seed(seed)
    random.seed(seed)
    
    # Basic partner information
    partner_ids = [f'P{100000 + i}' for i in range(n)]
    join_dates = [datetime.now() - timedelta(days=np.random.randint(30, 365*3)) for _ in range(n)]
    
    # Generate features with realistic relationships
    data = {
        'partner_id': partner_ids,
        'join_date': join_dates,
        'months_on_platform': np.random.randint(1, 37, size=n),  # 1-36 months
        'weekly_trips': np.random.poisson(lam=25, size=n) + 5,  # 5-50 trips/week
        'cancel_rate': np.clip(np.random.beta(2, 20, size=n), 0, 0.3),  # 0-30% cancel rate
        'on_time_rate': np.clip(np.random.beta(15, 2, size=n), 0.7, 1.0),  # 70-100% on-time
        'avg_rating': np.clip(np.random.normal(4.5, 0.3, size=n), 3.0, 5.0),  # 3.0-5.0 rating
        'earnings_volatility': np.clip(np.random.gamma(shape=2, scale=0.1, size=n), 0.05, 0.5),  # 5-50% CV in earnings
        'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], size=n),  # Geographic region
    }
    
    # Create a DataFrame
    df = pd.DataFrame(data)
    
    # Calculate log-odds of default with realistic relationships
    df['logit'] = (
        -2.0  # Base log-odds
        + 0.05 * df['months_on_platform']  # More stable with tenure
        - 0.1 * np.log1p(df['weekly_trips'])  # More active = lower risk
        + 2.0 * df['cancel_rate']  # Higher cancel rate = higher risk
        - 1.5 * (df['on_time_rate'] - 0.85)  # Higher on-time = lower risk
        - 0.5 * (df['avg_rating'] - 4.0)  # Higher rating = lower risk
        + 1.2 * (df['earnings_volatility'] / 0.2)  # More volatile earnings = higher risk
        + np.random.normal(0, 0.5, size=n)  # Random noise
    )
    
    # Add some regional bias (for fairness testing)
    region_effects = {
        'North': 0.1, 'South': -0.1, 'East': 0.2, 'West': -0.05, 'Central': 0.0
    }
    df['logit'] += df['region'].map(region_effects)
    
    # Convert log-odds to probability
    df['prob_default'] = 1 / (1 + np.exp(-df['logit']))
    
    # Generate binary outcome (repay_on_time = 1 if no default)
    df['repay_on_time'] = np.random.binomial(1, 1 - df['prob_default'])
    
    # Drop intermediate columns
    df = df.drop(columns=['logit', 'prob_default'])
    
    return df

def save_synthetic_data(df, filename='partners.csv'):
    """Save the synthetic dataset to a CSV file."""
    filepath = Path("data") / filename
    df.to_csv(filepath, index=False)
    print(f"Saved synthetic data to {filepath}")

def generate_synthetic_data(n=10000, filename='partners.csv'):
    """Generate and save synthetic partner data."""
    print("Generating synthetic partner data...")
    df = generate_synthetic_partners(n=n)
    save_synthetic_data(df, filename)
    
    # Print dataset summary
    print("\nDataset summary:")
    print(f"Number of partners: {len(df)}")
    print(f"Repayment rate: {df['repay_on_time'].mean():.1%}")
    print("\nFeature distributions:")
    print(df.describe().round(2).T)
    
    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
