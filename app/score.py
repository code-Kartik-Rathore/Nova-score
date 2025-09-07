"""
Streamlit UI for Nova Score demo.
Provides an interactive interface for credit scoring Grab partners.
"""
import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# Page config
st.set_page_config(
    page_title="ControlDesk - Nova Score",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="collapsed"  # Collapse the default sidebar
)

# Custom CSS for modern layout
st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 0;
    }
    
    /* Top Navigation Bar */
    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 2rem;
        background: linear-gradient(90deg, #1a237e, #283593);
        color: white;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        position: sticky;
        top: 0;
        z-index: 1000;
    }
    
    .nav-title {
        font-size: 1.5rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .nav-links {
        display: flex;
        gap: 1rem;
    }
    
    .nav-button {
        background: rgba(255, 255, 255, 0.1);
        border: none;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    
    .nav-button:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: translateY(-2px);
    }
    
    /* Main content */
    .content {
        padding: 2rem;
        max-width: 1400px;
        margin: 0 auto;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .score-card {
        text-align: center;
        padding: 2rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .score-high {
        background-color: #e6f7e6;
        border-left: 5px solid #4CAF50;
    }
    .score-medium {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
    }
    .score-low {
        background-color: #fde8e8;
        border-left: 5px solid #f44336;
    }
    .reason-card {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
        background-color: #f8f9fa;
    }
    .positive {
        color: #4CAF50;
    }
    .negative {
        color: #f44336;
    }
    </style>
""", unsafe_allow_html=True)

# API configuration
API_BASE_URL = "http://localhost:8001"

@dataclass
class Partner:
    partner_id: str
    months_on_platform: int
    weekly_trips: int
    cancel_rate: float
    on_time_rate: float
    avg_rating: float
    earnings_volatility: float
    region: str
    score: float = 0.0
    decision: str = ""

def generate_sample_partners(n=50):
    """Generate sample partner data for demo purposes."""
    np.random.seed(42)
    
    partners = []
    regions = ['North', 'South', 'East', 'West', 'Central']
    
    for i in range(n):
        partner = Partner(
            partner_id=f'P{100000 + i}',
            months_on_platform=int(np.random.uniform(1, 36)),
            weekly_trips=int(np.random.poisson(25) + 5),
            cancel_rate=float(np.clip(np.random.beta(2, 20), 0, 0.3)),
            on_time_rate=float(np.clip(np.random.beta(15, 2), 0.7, 1.0)),
            avg_rating=float(np.clip(np.random.normal(4.5, 0.3), 3.0, 5.0)),
            earnings_volatility=float(np.clip(np.random.gamma(2, 0.1), 0.05, 0.5)),
            region=np.random.choice(regions)
        )
        
        # Calculate a simple score (0-100)
        score = 50 + (partner.months_on_platform * 0.5) + \
                (partner.weekly_trips * 0.2) + \
                (partner.on_time_rate * 20) + \
                (partner.avg_rating * 5) - \
                (partner.cancel_rate * 100) - \
                (partner.earnings_volatility * 50)
                
        partner.score = max(0, min(100, score))
        partner.decision = "Approve" if partner.score >= 60 else "Review" if partner.score >= 40 else "Reject"
        partners.append(partner)
    
    return partners

def generate_sample_data():
    """Generate single partner data for the form."""
    partner = generate_sample_partners(1)[0]
    return {
        'partner_id': partner.partner_id,
        'months_on_platform': partner.months_on_platform,
        'weekly_trips': partner.weekly_trips,
        'cancel_rate': partner.cancel_rate,
        'on_time_rate': partner.on_time_rate,
        'avg_rating': partner.avg_rating,
        'earnings_volatility': partner.earnings_volatility,
        'region': partner.region
    }

def get_decision_color(decision):
    """Get color for decision badge."""
    if decision == "pre_approved":
        return "green"
    elif decision == "review":
        return "orange"
    else:
        return "red"

def display_score_card(score, decision):
    """Display the score card with appropriate styling."""
    if decision == "pre_approved":
        card_class = "score-card score-high"
        decision_text = "Pre-Approved"
    elif decision == "review":
        card_class = "score-card score-medium"
        decision_text = "Needs Review"
    else:
        card_class = "score-card score-low"
        decision_text = "Not Approved"
    
    st.markdown(f"""
    <div class="{card_class}">
        <h1>Nova Score</h1>
        <h2 style="font-size: 4rem; margin: 0.5rem 0;">{score}</h2>
        <div style="font-size: 1.5rem; margin: 1rem 0;">
            <span class="badge" style="background-color: {get_decision_color(decision)}; 
                                    color: white; 
                                    padding: 0.25rem 1rem; 
                                    border-radius: 20px;">
                {decision_text}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_reason_codes(reasons):
    """Display the reason codes in an expandable section."""
    with st.expander("View factors affecting this score", expanded=True):
        st.markdown("### Key factors influencing this score:")
        
        for reason in reasons:
            if 'error' in reason.get('feature', '').lower():
                st.warning(reason.get('value', 'Error getting reason codes'))
                continue
                
            impact = reason.get('impact', 0)
            impact_class = "positive" if impact > 0 else "negative"
            impact_icon = "⬆️" if impact > 0 else "⬇️"
            
            st.markdown(f"""
            <div class="reason-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong>{reason.get('feature', 'Unknown')}:</strong> {reason.get('message', '')}
                    </div>
                    <span class="{impact_class}" style="font-weight: bold;">
                        {impact_icon} {abs(impact):.2f}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

def display_feature_importance():
    """Display feature importance visualization."""
    try:
        # This would typically come from your model's metadata or SHAP values
        # For demo, we'll use sample data
        features = [
            'On-time Rate', 'Cancel Rate', 'Average Rating', 
            'Weekly Trips', 'Tenure (months)', 'Earnings Volatility'
        ]
        importance = [0.28, 0.24, 0.18, 0.15, 0.10, 0.05]
        
        df = pd.DataFrame({
            'Feature': features,
            'Importance': importance
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            df, 
            x='Importance', 
            y='Feature',
            orientation='h',
            title='Feature Importance',
            labels={'Importance': 'Relative Importance', 'Feature': ''},
            color='Importance',
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'},
            margin=dict(l=0, r=0, t=40, b=20),
            height=300,
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.warning(f"Could not load feature importance: {str(e)}")

def get_feature_importance() -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Get feature importance data and impact directions."""
    # These values would typically come from your model
    feature_importance = {
        'on_time_rate': 0.28,
        'cancel_rate': 0.22,
        'avg_rating': 0.20,
        'months_on_platform': 0.15,
        'earnings_volatility': 0.10,
        'weekly_trips': 0.05
    }
    
    # Impact direction (1 for positive, -1 for negative)
    impact_direction = {
        'on_time_rate': 1,
        'cancel_rate': -1,
        'avg_rating': 1,
        'months_on_platform': 1,
        'earnings_volatility': -1,
        'weekly_trips': 1
    }
    
    # Create DataFrame
    df = pd.DataFrame({
        'feature': list(feature_importance.keys()),
        'importance': list(feature_importance.values()),
        'impact': [impact_direction[f] for f in feature_importance]
    })
    
    # Sort by importance
    df = df.sort_values('importance', ascending=False)
    
    return df, impact_direction

def display_score_distribution(partners: List[Dict]) -> None:
    """Display score distribution and regional analysis."""
    df = pd.DataFrame(partners)
    
    st.subheader("📊 Score Analysis")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Histogram", "By Region", "Feature Impact"])
    
    with tab1:
        # Score distribution histogram
        fig = px.histogram(
            df, 
            x="score",
            nbins=20,
            labels={"score": "Credit Score"},
            title="Distribution of Credit Scores",
            color_discrete_sequence=['#FF4B4B']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Average score by region
        region_avg = df.groupby('region')['score'].mean().reset_index()
        fig = px.bar(
            region_avg,
            x='region',
            y='score',
            title='Average Score by Region',
            labels={'score': 'Average Score', 'region': 'Region'},
            color='score',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with tab3:
        # Feature importance visualization
        importance_df, impact_dirs = get_feature_importance()
        
        # Create a horizontal bar chart with color-coded impact
        fig = go.Figure()
        
        # Add positive impact bars
        pos_df = importance_df[importance_df['impact'] > 0]
        fig.add_trace(go.Bar(
            y=pos_df['feature'],
            x=pos_df['importance'],
            name='Positive Impact',
            orientation='h',
            marker_color='#4CAF50',
            hovertemplate='%{y}: +%{x:.0%} impact<extra></extra>'
        ))
        
        # Add negative impact bars
        neg_df = importance_df[importance_df['impact'] < 0]
        fig.add_trace(go.Bar(
            y=neg_df['feature'],
            x=neg_df['importance'],
            name='Negative Impact',
            orientation='h',
            marker_color='#F44336',
            hovertemplate='%{y}: -%{x:.0%} impact<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title='Feature Impact on Credit Score',
            xaxis_title='Impact on Score',
            yaxis_title='Feature',
            barmode='relative',
            height=400,
            showlegend=True,
            xaxis=dict(tickformat=".0%")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        with st.expander("ℹ️ How to interpret this chart"):
            st.markdown("""
            - **Green bars** indicate features that **increase** the credit score when improved
            - **Red bars** indicate features that **decrease** the credit score when they increase
            - The width of each bar shows the relative importance of each feature
            - Hover over the bars to see the exact impact percentage
            
            **Example**: Improving your on-time rate will have the most positive impact on your score, 
            while reducing your cancel rate is the most important factor to avoid score decreases.
            """) 

def display_leaderboard(partners: List[Partner], top_n: int = 10):
    """Display the top N partners in a leaderboard."""
    st.markdown("## 🏆 Partner Leaderboard")
    
    # Sort partners by score
    top_partners = sorted(partners, key=lambda x: x.score, reverse=True)[:top_n]
    
    # Create a DataFrame for display
    leaderboard_data = []
    for i, partner in enumerate(top_partners, 1):
        leaderboard_data.append({
            'Rank': i,
            'Partner ID': partner.partner_id,
            'Score': f"{partner.score:.1f}",
            'Decision': partner.decision,
            'Trips/Week': partner.weekly_trips,
            'Rating': f"{partner.avg_rating:.1f} ⭐",
            'Region': partner.region
        })
    
    # Display the leaderboard with color coding
    df = pd.DataFrame(leaderboard_data)
    st.dataframe(
        df,
        column_config={
            'Score': st.column_config.ProgressColumn(
                'Score',
                format="%.1f",
                min_value=0,
                max_value=100,
            ),
            'Decision': st.column_config.TextColumn(
                'Decision',
                help="Approval Status"
            )
        },
        hide_index=True,
        use_container_width=True
    )
    
    # Add key metrics at the top
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Partners", len(partners))
    with col2:
        st.metric("Average Score", f"{np.mean([p.score for p in partners]):.1f}")
    with col3:
        approval_rate = sum(1 for p in partners if p.decision == 'Approve') / len(partners) * 100
        st.metric("Approval Rate", f"{approval_rate:.1f}%")
    
    # Create tabs for visualizations
    tab1, tab2 = st.tabs(["Score Distribution", "Feature Importance"])
    
    with tab1:
        # Score distribution by decision
        decision_counts = pd.DataFrame(leaderboard_data)['Decision'].value_counts().reset_index()
        decision_counts.columns = ['Decision', 'Count']
        
        fig = px.pie(
            decision_counts,
            values='Count',
            names='Decision',
            title='Decision Distribution',
            color='Decision',
            color_discrete_map={
                'Approve': '#4CAF50',
                'Review': '#FFC107',
                'Reject': '#F44336'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Display feature importance
        try:
            importance_df, _ = get_feature_importance()
            
            # Create a horizontal bar chart with color-coded impact
            fig = go.Figure()
            
            # Add positive impact bars
            pos_df = importance_df[importance_df['impact'] > 0]
            fig.add_trace(go.Bar(
                y=pos_df['feature'],
                x=pos_df['importance'],
                name='Positive Impact',
                orientation='h',
                marker_color='#4CAF50',
                hovertemplate='%{y}: +%{x:.0%} impact<extra></extra>'
            ))
            
            # Add negative impact bars
            neg_df = importance_df[importance_df['impact'] < 0]
            if not neg_df.empty:
                fig.add_trace(go.Bar(
                    y=neg_df['feature'],
                    x=neg_df['importance'],
                    name='Negative Impact',
                    orientation='h',
                    marker_color='#F44336',
                    hovertemplate='%{y}: -%{x:.0%} impact<extra></extra>'
                ))
            
            # Update layout
            fig.update_layout(
                title='Feature Impact on Credit Score',
                xaxis_title='Impact on Score',
                yaxis_title='Feature',
                barmode='relative',
                height=400,
                showlegend=True,
                xaxis=dict(tickformat=".0%")
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation
            with st.expander("ℹ️ How to interpret this chart"):
                st.markdown("""
                - **Green bars** indicate features that **increase** the credit score when improved
                - **Red bars** indicate features that **decrease** the credit score when they increase
                - The width of each bar shows the relative importance of each feature
                - Hover over the bars to see the exact impact percentage
                
                **Example**: Improving your on-time rate will have the most positive impact on your score, 
                while reducing your cancel rate is the most important factor to avoid score decreases.
                """)
            
        except Exception as e:
            st.warning(f"Could not display feature importance: {str(e)}")

def main():
    # Set up the sidebar
    st.sidebar.title("Nova Score")
    st.sidebar.markdown("### Partner Credit Scoring System")
    st.sidebar.markdown("---")
    
    # Add navigation
    page = st.sidebar.radio("Navigation", ["Score Calculator", "Leaderboard", "Documentation"])

    if 'partners' not in st.session_state:
        st.session_state.partners = generate_sample_partners(100)

    if page == "Score Calculator":
        # Main content
        st.markdown("<a id='dashboard'></a>", unsafe_allow_html=True)
        st.title("📊 Partner Credit Scoring")
        st.markdown("### Evaluate partner creditworthiness in real-time")
        
        # Create two columns for the form
        col1, col2 = st.columns(2)
        
        with col1:
            # Partner Information
            st.markdown("#### Partner Information")
            partner_id = st.text_input("Partner ID", value=st.session_state.get('partner_id', ''))
            region = st.selectbox(
                "Region",
                ["North", "South", "East", "West", "Central"],
                index=4 if st.session_state.get('region') == 'Central' else 0
            )
            
            # Performance Metrics
            st.markdown("#### Performance Metrics")
            months_on_platform = st.slider(
                "Months on Platform",
                min_value=1,
                max_value=36,
                value=st.session_state.get('months_on_platform', 12),
                step=1
            )
            
            weekly_trips = st.slider(
                "Weekly Trips",
                min_value=0,
                max_value=100,
                value=st.session_state.get('weekly_trips', 30),
                step=1
            )
            
            cancel_rate = st.slider(
                "Cancel Rate (%)",
                min_value=0.0,
                max_value=30.0,
                value=st.session_state.get('cancel_rate', 10.0),
                step=0.5,
                format="%.1f%%"
            ) / 100.0
            
        with col2:
            # Service Quality
            st.markdown("#### Service Quality")
            
            on_time_rate = st.slider(
                "On-time Delivery Rate",
                min_value=70.0,
                max_value=100.0,
                value=st.session_state.get('on_time_rate', 95.0),
                step=0.5,
                format="%.1f%%"
            ) / 100.0
            
            avg_rating = st.slider(
                "Average Rating",
                min_value=3.0,
                max_value=5.0,
                value=st.session_state.get('avg_rating', 4.5),
                step=0.1,
                format="%.1f"
            )
            
            earnings_volatility = st.slider(
                "Earnings Volatility",
                min_value=5.0,
                max_value=50.0,
                value=st.session_state.get('earnings_volatility', 20.0),
                step=1.0,
                format="%.1f%%"
            ) / 100.0
            
            # Add some space
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Submit button
            if st.button("Calculate Score", key="calculate_score", use_container_width=True):
                # Prepare the request data
                data = {
                    "partner_id": partner_id,
                    "months_on_platform": months_on_platform,
                    "weekly_trips": weekly_trips,
                    "cancel_rate": cancel_rate,
                    "on_time_rate": on_time_rate,
                    "avg_rating": avg_rating,
                    "earnings_volatility": earnings_volatility,
                    "region": region
                }
                
                try:
                    # Call the API
                    response = requests.post(
                        "http://localhost:8001/score",
                        json=data
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state['score_result'] = result
                        
                        # Display the score card
                        display_score_card(
                            result.get('score', 0),
                            result.get('decision', 'Unknown')
                        )
                        
                        # Display reason codes if available
                        if 'reason_codes' in result:
                            display_reason_codes(result['reason_codes'])
                        
                        # Display feature importance
                        display_feature_importance()
                        
                    else:
                        st.error(f"Error: {response.text}")
                        
                except Exception as e:
                    st.error(f"Failed to connect to the scoring service: {str(e)}")
        
        # Display sample data button
        if st.sidebar.button("Load Sample Data"):
            sample_data = generate_sample_data()
            for key, value in sample_data.items():
                st.session_state[key] = value
            st.experimental_rerun()
    
    elif page == "Leaderboard":
        st.title("Partner Leaderboard & Analytics")
        
        # Display the leaderboard
        display_leaderboard(st.session_state.partners)
        
        # Display score distribution and metrics
        display_score_distribution(st.session_state.partners)
        
        # Add a refresh button
        if st.button("🔄 Refresh Data"):
            st.session_state.partners = generate_sample_partners(100)
            st.experimental_rerun()
            
    elif page == "Documentation":
        st.title("Documentation")
        st.markdown("""
        ### About Nova Score
        Nova Score is a credit scoring system designed specifically for Grab partners. 
        It evaluates various factors to determine a partner's creditworthiness.
        
        ### How It Works
        1. **Data Collection**: We collect various metrics about partner performance.
        2. **Scoring**: Our machine learning model analyzes these metrics to generate a score.
        3. **Decision**: Based on the score, we make a credit decision.
        
        ### Score Interpretation
        - **80-100**: Excellent - Low risk
        - **60-79**: Good - Moderate risk
        - **40-59**: Review Required - Higher risk
        - **0-39**: Reject - High risk
        
        ### Leaderboard Features
        - View top-performing partners
        - Compare performance across regions
        - Analyze score distributions
        
        ### API Documentation
        Our scoring API is available at `POST /score` with the following JSON body:
        ```json
        {
            "partner_id": "string",
            "months_on_platform": number,
            "weekly_trips": number,
            "cancel_rate": number,
            "on_time_rate": number,
            "avg_rating": number,
            "earnings_volatility": number,
            "region": "string"
        }
        ```
        """)

if __name__ == "__main__":
    # Close the content div
    st.markdown("</div>", unsafe_allow_html=True)
    
    if 'months_on_platform' not in st.session_state:
        sample_data = generate_sample_data()
        for key, value in sample_data.items():
            st.session_state[key] = value
    
    main()
