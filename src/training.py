import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from credit_risk_assessor import CreditRiskAssessor
from config import MODEL_PATH, FEATURE_DATA_PATH, LABELS_DATA_PATH, MODEL_CONFIG
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model():
    """Train and save the credit risk model."""
    try:
        # Load data
        logger.info("Loading data...")
        X = pd.read_csv(FEATURE_DATA_PATH)
        y = pd.read_csv(LABELS_DATA_PATH)['default']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=MODEL_CONFIG['test_size'],
            random_state=MODEL_CONFIG['random_state']
        )
        
        # Initialize and train model
        logger.info("Training model...")
        assessor = CreditRiskAssessor()
        assessor.train(X_train, y_train)
        
        # Evaluate model
        logger.info("Evaluating model...")
        test_predictions = []
        test_probabilities = []
        
        for _, row in X_test.iterrows():
            data = row.to_dict()
            data.update({
                'utility_payments': [{'days_late': 0}],
                'utility_amounts': [100],
                'social_posts': [""],
                'social_activity': {
                    'posts': 50,
                    'professional_connections': 500,
                    'profile_completeness': 0.9
                },
                'transactions': [{'amount': 1000}],
                'expenses': {
                    'essential': 2000,
                    'total': 3000
                }
            })
            
            risk_score, _ = assessor.predict_risk(data)
            test_probabilities.append(risk_score)
            test_predictions.append(1 if risk_score > 0.5 else 0)
        
        # Save model
        logger.info(f"Saving model to {MODEL_PATH}")
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(assessor, f)
        
        logger.info("Model training complete!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()