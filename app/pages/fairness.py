"""
Fairness Dashboard - Interactive visualization of model fairness metrics
with smooth animations and what-if analysis
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from app.utils.animations import add_loading_animation
import time

# Page config
st.set_page_config(page_title="Fairness Dashboard - Nova Score", page_icon="📊")

def load_fairness_data():
    """Generate mock fairness data."""
    # Mock data for demonstration
    regions = ["North", "South", "East", "West", "Central"]
    tenure_buckets = ["0-3", "4-12", "13-24", "25-36", "37+"]
    
    # Generate random approval rates with some controlled variation
    np.random.seed(42)
    region_approval = {
        'Region': regions,
        'Approval Rate': np.clip(0.65 + np.random.normal(0, 0.05, len(regions)), 0.5, 0.8)
    }
    
    tenure_approval = {
        'Tenure (months)': tenure_buckets,
        'Approval Rate': np.clip([0.4, 0.6, 0.7, 0.75, 0.8] + np.random.normal(0, 0.03, 5), 0.3, 0.85)
    }
    
    return {
        'eo_gap': 0.032,  # Equal Opportunity gap
        'dp_gap': 0.046,  # Demographic Parity gap
        'threshold': 0.60,
        'region_data': pd.DataFrame(region_approval),
        'tenure_data': pd.DataFrame(tenure_approval)
    }

def main():
    st.title("📊 Fairness Dashboard")
    
    # Load data
    data = load_fairness_data()
    
    # Fairness metrics with animated cards
    st.markdown("## 🎯 Fairness Metrics")
    
    # Animate the metrics in sequence
    metrics_container = st.container()
    
    with metrics_container:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            with st.spinner("Analyzing equal opportunity..."):
                time.sleep(0.5)
                eo_color = "green" if data['eo_gap'] <= 0.05 else "orange"
                st.metric(
                    "Equal Opportunity Gap", 
                    f"{data['eo_gap']*100:.1f} pp",
                    delta_color="off",
                    help="Difference in true positive rates between groups"
                )
                st.markdown(
                    f"<span style='color:{eo_color}'>"
                    f"{'✓ Within target' if data['eo_gap'] <= 0.05 else '⚠️ Monitor closely'}"
                    "</span>",
                    unsafe_allow_html=True
                )
        
        with col2:
            with st.spinner("Checking demographic parity..."):
                time.sleep(0.8)
                dp_color = "green" if data['dp_gap'] <= 0.05 else "orange"
                st.metric(
                    "Demographic Parity Gap", 
                    f"{data['dp_gap']*100:.1f} pp",
                    delta_color="off",
                    help="Difference in approval rates between groups"
                )
                st.markdown(
                    f"<span style='color:{dp_color}'>"
                    f"{'✓ Within target' if data['dp_gap'] <= 0.05 else '⚠️ Monitor closely'}"
                    "</span>",
                    unsafe_allow_html=True
                )
        
        with col3:
            with st.spinner("Loading threshold..."):
                time.sleep(1.1)
                st.metric("Threshold", f"{data['threshold']*100:.2f}%", "Score cutoff")
    
    # Add a subtle animation to the metrics container
    st.markdown("""
    <style>
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .metric-card {
            animation: fadeInUp 0.6s ease-out;
            transition: all 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-3px);
        }
    </style>
    """, unsafe_allow_html=True)
    
    # What-if analysis with interactive threshold
    st.markdown("## 🎚️ What-if Analysis")
    
    # Add a container for the threshold slider with animation
    with st.container():
        st.markdown("### Adjust Approval Threshold")
        
        # Add a visual indicator of the threshold range
        col1, col2, col3 = st.columns([1, 6, 1])
        
        with col1:
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            st.markdown("<small>0%</small>", unsafe_allow_html=True)
        
        with col2:
            # Threshold slider with callback
            new_threshold = st.slider(
                "",
                min_value=0,
                max_value=100,
                value=int(data['threshold']*100),
                step=1,
                format="%d%%",
                help="See how changing the approval threshold affects fairness metrics",
                label_visibility="collapsed"
            )
            
            # Add a visual indicator of the threshold range
            st.markdown("""
            <div style="display: flex; justify-content: space-between; margin-top: -15px;">
                <small>Low Risk</small>
                <small>Medium Risk</small>
                <small>High Risk</small>
            </div>
            <div style="
                height: 10px;
                background: linear-gradient(90deg, #4CAF50, #FFC107, #F44336);
                border-radius: 5px;
                margin: 5px 0 20px 0;
                position: relative;
            ">
                <div style="
                    position: absolute;
                    top: -15px;
                    left: calc({}% - 1px);
                    width: 2px;
                    height: 30px;
                    background: #000;
                    z-index: 10;
                "></div>
            </div>
            """.format(new_threshold), unsafe_allow_html=True)
        
        with col3:
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            st.markdown("<small>100%</small>", unsafe_allow_html=True)
    
    # Show impact of threshold change
    if new_threshold != int(data['threshold']*100):
        with st.spinner("Recalculating metrics..."):
            time.sleep(0.5)
            
            # Calculate impact
            approval_change = abs(new_threshold - int(data['threshold']*100))
            impact_direction = "increase" if new_threshold > int(data['threshold']*100) else "decrease"
            
            # Show impact metrics
            st.markdown(f"#### Impact of {approval_change}% {impact_direction} in threshold:")
            
            # Create columns for impact metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Approval Rate Change",
                    f"{approval_change}%",
                    f"{impact_direction.capitalize()}",
                    delta_color=("inverse" if impact_direction == "increase" else "normal")
                )
            
            with col2:
                st.metric(
                    "Affected Partners",
                    f"{approval_change * 25:,}",
                    "Estimated"
                )
            
            with col3:
                st.metric(
                    "Fairness Impact",
                    "Minimal" if approval_change < 5 else "Moderate" if approval_change < 15 else "Significant",
                    "On model bias"
                )
            
            # Add a reset button
            if st.button("↩️ Reset to original threshold", type="secondary"):
                new_threshold = int(data['threshold']*100)
                st.rerun()
    
    # Approval rates by region with interactive chart
    st.markdown("## 📊 Approval Rates by Region")
    
    # Generate mock data with animation
    with st.spinner("Loading regional data..."):
        time.sleep(0.5)
        
        regions = ["Singapore", "Malaysia", "Indonesia", "Thailand", "Philippines", "Vietnam"]
        approval_rates = [78, 82, 75, 80, 77, 79]
        
        # Add some variation based on threshold
        if new_threshold > int(data['threshold']*100):
            approval_rates = [max(0, r - (new_threshold - int(data['threshold']*100))) for r in approval_rates]
        elif new_threshold < int(data['threshold']*100):
            approval_rates = [min(100, r + (int(data['threshold']*100) - new_threshold)) for r in approval_rates]
        
        # Create animated bar chart
        fig1 = go.Figure(
            data=[go.Bar(
                x=regions,
                y=approval_rates,
                marker_color=px.colors.qualitative.Plotly[:len(regions)],
                text=approval_rates,
                texttemplate='%{text:.1f}%',
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Approval Rate: %{y:.1f}%<extra></extra>'
            )]
        )
        
        # Add threshold line
        fig1.add_hline(
            y=new_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Current Threshold: {new_threshold}%",
            annotation_position="bottom right"
        )
        
        # Update layout for better interactivity
        fig1.update_layout(
            title="Approval Rate by Region",
            xaxis_title="Region",
            yaxis_title="Approval Rate %",
            yaxis_range=[0, 100],
            hovermode="closest",
            showlegend=False,
            transition_duration=500,
            margin=dict(t=40, b=40, l=40, r=40)
        )
        
        # Add animation
        fig1.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="category"
            )
        )
        
        st.plotly_chart(fig1, use_container_width=True, theme=None)
    
    # Charts
    st.markdown("### Approval Rates by Segment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Region chart
        fig_region = px.bar(
            data['region_data'],
            x='Region',
            y='Approval Rate',
            title='Approval Rate by Region',
            color='Approval Rate',
            color_continuous_scale='RdYlGn',
            range_color=[0.5, 0.8],
            text_auto='.1%'
        )
        fig_region.update_layout(
            yaxis_tickformat='.0%',
            yaxis_range=[0, 1],
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_region, use_container_width=True)
    
    with col2:
        # Tenure chart
        fig_tenure = px.bar(
            data['tenure_data'],
            x='Tenure (months)',
            y='Approval Rate',
            title='Approval Rate by Tenure',
            color='Approval Rate',
            color_continuous_scale='RdYlGn',
            range_color=[0.3, 0.85],
            text_auto='.1%'
        )
        fig_tenure.update_layout(
            yaxis_tickformat='.0%',
            yaxis_range=[0, 1],
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_tenure, use_container_width=True)
    
    # Methodology
    with st.expander("📊 Methodology"):
        st.markdown("""
        ### How We Measure Fairness
        
        - **Equal Opportunity Gap**: Difference in true positive rates between groups
          - *Target*: < 5 percentage points
          
        - **Demographic Parity Gap**: Difference in approval rates between groups
          - *Target*: < 5 percentage points
          
        - **Segmentation**: We analyze approval rates across:
          - Geographic regions
          - Tenure on platform
          - Other protected attributes (not shown)
          
        *Note: This dashboard shows simulated data for demonstration purposes.*
        """)
    
    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("← Back to Home"):
            st.session_state.page = "home"
            st.rerun()
    
    with col2:
        if st.button("Open Ops Console →"):
            st.session_state.page = "ops"
            st.rerun()

if __name__ == "__main__":
    main()
