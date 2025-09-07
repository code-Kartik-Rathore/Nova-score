"""
About Page - Model and system information
"""
import streamlit as st

# Page config
st.set_page_config(page_title="About - Nova Score", page_icon="ℹ️")

def main():
    st.title("ℹ️ About Nova Score")
    
    st.markdown("""
    ## Model Information
    
    **Nova Score v0.1** is a credit scoring system designed specifically for Grab partners.
    It helps evaluate partner reliability and creditworthiness using machine learning.
    
    ### Technical Details
    
    - **Model Type**: XGBoost Classifier
    - **Training Data**: Synthetic partner data (simulated)
    - **Version**: 0.1.0
    - **Last Updated**: August 2025
    
    ### Performance Metrics
    
    | Metric | Value |
    |--------|-------|
    | **AUROC** | 0.79 |
    | **Accuracy** | 0.82 |
    | **Precision** | 0.78 |
    | **Recall** | 0.75 |
    | **F1-Score** | 0.76 |
    
    ## Fairness & Ethics
    
    We're committed to building fair and transparent AI systems:
    
    - **Bias Monitoring**: Regular fairness audits across protected attributes
    - **Explainability**: SHAP values for model interpretability
    - **Human Oversight**: All critical decisions are reviewed by our operations team
    
    ## Data Privacy
    
    - No personally identifiable information (PII) is used in model features
    - All data is anonymized and aggregated for analysis
    - Partners can request their data at any time
    
    ## Contact
    
    For questions or feedback, please contact:
    
    - **Email**: nova-support@grab.com
    - **Phone**: +65 XXXX XXXX
    
    ---
    
    *This is a demo application. All data shown is synthetic and for demonstration purposes only.*
    """)
    
    # Navigation
    if st.button("← Back to Home"):
        if 'page' in st.session_state:
            st.session_state.page = "home"
            st.rerun()

if __name__ == "__main__":
    main()
