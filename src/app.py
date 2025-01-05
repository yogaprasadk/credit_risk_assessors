import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pickle
from pathlib import Path
from config import MODEL_PATH

def load_model():
    """Load the trained model."""
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

def main():
    st.set_page_config(page_title="Credit Risk Assessment", layout="wide")
    
    st.title("Real-Time Credit Risk Assessment")
    st.subheader("Democratizing Credit Access Through Alternative Data")
    
    # Load model
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # Sidebar inputs
    st.sidebar.header("Applicant Information")
    
    # Traditional Data
    st.sidebar.subheader("Traditional Data")
    credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
    income = st.sidebar.number_input("Annual Income", 20000, 200000, 60000)
    dti = st.sidebar.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3)
    employment_length = st.sidebar.slider("Employment Length (years)", 0, 20, 5)
    credit_lines = st.sidebar.slider("Number of Credit Lines", 0, 10, 3)
    
    # Alternative Data
    st.sidebar.subheader("Alternative Data")
    utility_score = st.sidebar.slider("Utility Payment Score", 0.0, 1.0, 0.8)
    social_score = st.sidebar.slider("Social Media Activity Score", 0.0, 1.0, 0.7)
    spending_score = st.sidebar.slider("Spending Pattern Score", 0.0, 1.0, 0.75)
    
    # Prepare input data
    input_data = {
        'credit_score': credit_score,
        'income': income,
        'debt_to_income': dti,
        'employment_length': employment_length,
        'num_credit_lines': credit_lines,
        'utility_payments': [{'days_late': 0}],
        'utility_amounts': [100],
        'social_posts': [""],
        'social_activity': {
            'posts': 50,
            'professional_connections': 500,
            'profile_completeness': social_score
        },
        'transactions': [{'amount': 1000}],
        'expenses': {
            'essential': 2000,
            'total': 3000
        }
    }
    
    # Make prediction
    risk_score, explanation = model.predict_risk(input_data)
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Risk Assessment")
        
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score * 100,
            title={'text': "Risk Score"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ]
            }
        ))
        st.plotly_chart(fig)
        
        # Decision
        if risk_score < 0.3:
            st.success("Recommendation: Approve")
        elif risk_score < 0.7:
            st.warning("Recommendation: Review Required")
        else:
            st.error("Recommendation: Deny")
    
    with col2:
        st.header("Factor Analysis")
        
        # Create feature importance chart
        importance_df = pd.DataFrame({
            'Factor': [k for k, v in explanation['top_factors']],
            'Importance': [abs(v) for k, v in explanation['top_factors']]
        })
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Factor',
            orientation='h',
            title='Top Contributing Factors'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()