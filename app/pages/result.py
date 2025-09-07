"""
Result Page - Shows the score and recommendations with engaging animations
"""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from utils.animations import animate_score_counter, add_confetti, add_success_message

# Page config
st.set_page_config(page_title="Your Nova Score", page_icon="📊")

def calculate_score(data):
    """Calculate a mock score based on input data."""
    # This is a simplified scoring function for demo purposes
    score = 70  # Base score
    
    # Adjust based on factors
    score += (data.get('on_time_rate', 0) - 0.85) * 100  # 85% is baseline
    score -= data.get('cancel_rate', 0) * 100  # Reduce for cancellations
    score += (data.get('avg_rating', 3.0) - 3.0) * 10  # 3.0 is baseline
    score += min(data.get('tenure_months', 0) / 2, 10)  # Cap tenure bonus
    
    # Cap score between 0-100
    return max(0, min(100, score))

def get_score_band(score):
    """Determine the score band."""
    if score >= 80:
        return "Pre-approved", "✅ Pre-approved", "success"
    elif score >= 60:
        return "Review", "⚠️ Review", "warning"
    else:
        return "Decline", "❌ Decline", "error"

def create_gauge_figure(score):
    """Create a gauge chart for the score."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Nova Score", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "#FF5252"},
                {'range': [40, 70], 'color': "#FFC107"},
                {'range': [70, 100], 'color': "#4CAF50"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=50, r=50, t=100, b=10)
    )
    return fig

def main():
    st.title("Your Nova Score")
    
    # Calculate score based on session state or use demo data
    if not hasattr(st.session_state, 'on_time_rate'):
        st.warning("No score data found. Using demo data.")
        st.session_state.update({
            'on_time_rate': 0.92,
            'cancel_rate': 0.08,
            'avg_rating': 4.7,
            'tenure_months': 12
        })
    
    score = calculate_score(st.session_state)
    band, band_display, band_style = get_score_band(score)
    
    # Show score and band with animation
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Animate the gauge
        gauge_placeholder = st.empty()
        with gauge_placeholder:
            st.plotly_chart(create_gauge_figure(0), use_container_width=True)
        
        # Animate the score counter
        score_placeholder = st.empty()
        animate_score_counter(score_placeholder, score, duration=1.5)
        
        # Show the final gauge with the actual score
        gauge_placeholder.plotly_chart(create_gauge_figure(score), use_container_width=True)
        
        # Add confetti for high scores
        if score >= 80:
            add_confetti()
            add_success_message("🎉 Congratulations! You've been pre-approved!", delay=2)
    
    with col2:
        st.markdown(f"## {int(score)}/100")
        st.markdown(f"### {band_display}")
        
        # Reason chips
        st.markdown("#### Key Factors")
        reasons = []
        
        if st.session_state.get('cancel_rate', 0) < 0.1:
            reasons.append("✓ Low cancel rate")
        else:
            reasons.append("✗ High cancel rate")
            
        if st.session_state.get('on_time_rate', 0) > 0.9:
            reasons.append("✓ Excellent on-time rate")
        
        if st.session_state.get('avg_rating', 0) >= 4.5:
            reasons.append("✓ High customer ratings")
        
        for reason in reasons:
            st.markdown(f"- {reason}")
    
    # What you did right / What to improve
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✅ What You Did Right")
        st.markdown("""
        - Consistently high ratings from customers
        - Reliable on-time performance
        - Good track record with cancellations
        """)
    
    with col2:
        st.markdown("### 📈 What To Improve")
        st.markdown("""
        - Reduce cancellation rate below 5%
        - Maintain on-time rate above 95%
        - Increase weekly trips for better consistency
        """)
    
    # Action buttons based on score band
    st.markdown("---")
    
    if band == "Pre-approved":
        if st.button("🎉 View Mock Offer", type="primary"):
            st.session_state.show_offer = True
        
        if st.session_state.get('show_offer'):
            with st.expander("🎯 Your Pre-Approved Offer", expanded=True):
                st.markdown("""
                #### 💰 Credit Line: S$5,000
                - **APR:** 12.9%
                - **Term:** 12 months
                - **Monthly Payment:** S$445.00
                - **Disbursement:** Same day
                
                *This is a mock offer for demonstration purposes.*
                """)
                
                if st.button("✓ Accept Offer"):
                    st.success("Offer accepted! Your credit line will be available shortly.")
    
    elif band == "Review":
        if st.button("📄 Request Documents", type="primary"):
            st.session_state.show_docs = True
        
        if st.session_state.get('show_docs'):
            with st.expander("📑 Required Documents", expanded=True):
                st.markdown("""
                Please upload the following documents for verification:
                
                - [ ] Bank statements (last 3 months)
                - [ ] NRIC/Passport copy
                - [ ] Proof of address
                - [ ] Business registration (for merchants)
                
                *Our team will review your application within 2 business days.*
                """)
    
    else:  # Decline
        if st.button("📝 Build Your Score Plan", type="primary"):
            st.session_state.show_plan = True
        
        if st.session_state.get('show_plan'):
            with st.expander("📈 30-Day Score Improvement Plan", expanded=True):
                st.markdown("""
                ### Your Action Plan
                
                #### 1. Reduce Cancellations (Target: <5%)
                - [ ] Accept orders you can fulfill
                - [ ] Plan your schedule better
                
                #### 2. Improve On-Time Performance (Target: >95%)
                - [ ] Allow extra time for deliveries
                - [ ] Check traffic before starting trips
                
                #### 3. Increase Activity
                - [ ] Complete 30+ trips this week
                - [ ] Maintain high ratings (4.5+)
                
                *Check back in 30 days for a score review.*
                """)
    
    # Footer navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("← Back to Home"):
            st.session_state.page = "home"
            st.rerun()
    
    with col2:
        if st.button("Try Another Scenario"):
            st.session_state.page = "partner"
            st.rerun()

if __name__ == "__main__":
    main()
