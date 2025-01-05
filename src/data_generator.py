import pandas as pd
import numpy as np
from typing import Tuple
from config import FEATURE_DATA_PATH, LABELS_DATA_PATH, TRADITIONAL_FEATURES, ALTERNATIVE_FEATURES

def generate_synthetic_data(n_samples: int = 1000) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate synthetic data for training the credit risk model.
    """
    np.random.seed(42)
    
    # Generate traditional features
    data = {
        'credit_score': np.random.normal(650, 100, n_samples).clip(300, 850),
        'income': np.random.normal(60000, 20000, n_samples).clip(20000, 150000),
        'debt_to_income': np.random.normal(0.3, 0.1, n_samples).clip(0, 0.8),
        'employment_length': np.random.normal(5, 3, n_samples).clip(0, 20),
        'num_credit_lines': np.random.normal(3, 2, n_samples).clip(0, 10),
        
        # Alternative data features
        'utility_payment_regularity': np.random.beta(8, 2, n_samples),
        'utility_bill_amount_consistency': np.random.beta(7, 3, n_samples),
        'social_media_sentiment': np.random.normal(0.6, 0.2, n_samples).clip(0, 1),
        'social_media_activity_score': np.random.beta(5, 5, n_samples),
        'spending_regularity': np.random.beta(6, 4, n_samples),
        'essential_spending_ratio': np.random.beta(7, 3, n_samples)
    }
    
    # Create DataFrame
    X = pd.DataFrame(data)
    
    # Generate target variable (default probability)
    default_prob = (
        -0.4 * (X['credit_score'] - 300) / 550 +  # Normalize credit score
        -0.2 * (X['income'] - 20000) / 130000 +   # Normalize income
        0.3 * X['debt_to_income'] +
        -0.1 * X['utility_payment_regularity'] +
        -0.1 * X['spending_regularity'] +
        0.1 * (1 - X['essential_spending_ratio'])
    )
    
    # Convert to binary target (1 = default, 0 = no default)
    y = (default_prob + np.random.normal(0, 0.1, n_samples) > 0).astype(int)
    
    return X, y

def save_data(n_samples: int = 1000):
    """Generate and save synthetic data."""
    X, y = generate_synthetic_data(n_samples)
    
    # Save to CSV files
    X.to_csv(FEATURE_DATA_PATH, index=False)
    pd.Series(y).to_csv(LABELS_DATA_PATH, index=False, header=True, name='default')
    
    print(f"Data saved to {FEATURE_DATA_PATH} and {LABELS_DATA_PATH}")
    print("\nFeature statistics:")
    print(X.describe())
    print("\nDefault rate:", y.mean())

if __name__ == "__main__":
    save_data()