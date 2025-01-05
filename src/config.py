import os
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Model parameters
MODEL_CONFIG = {
    'n_estimators': 100,
    'random_state': 42,
    'test_size': 0.2
}

# Feature configuration
TRADITIONAL_FEATURES = [
    'credit_score',
    'income',
    'debt_to_income',
    'employment_length',
    'num_credit_lines'
]

ALTERNATIVE_FEATURES = [
    'utility_payment_regularity',
    'utility_bill_amount_consistency',
    'social_media_sentiment',
    'social_media_activity_score',
    'spending_regularity',
    'essential_spending_ratio'
]

# File paths
FEATURE_DATA_PATH = DATA_DIR / "credit_risk_features.csv"
LABELS_DATA_PATH = DATA_DIR / "credit_risk_labels.csv"
MODEL_PATH = MODELS_DIR / "credit_risk_model.pkl"   