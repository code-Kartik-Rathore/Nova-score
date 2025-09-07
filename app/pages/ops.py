"""
ControlDesk - Interactive dashboard for reviewing and managing partner applications
with smooth animations and micro-interactions
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.animations import add_loading_animation, add_success_message, add_error_message
except ImportError:
    # Fallback implementations if animations module is not available
    def add_loading_animation():
        """Dummy function if animations module is not available"""
        pass
    
    def add_success_message(message, icon="✅"):
        """Dummy function for success message"""
        st.success(f"{icon} {message}")
    
    def add_error_message(message, icon="❌"):
        """Dummy function for error message"""
        st.error(f"{icon} {message}")

# Page config
st.set_page_config(
    page_title="ControlDesk - Nova Score",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for modern layout
st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 0;
        background-color: #ffffff;
        font-family: "Inter", sans-serif;
    }
    
    /* Ensure all text is visible */
    body {
        color: #1f2937 !important;
        background-color: #ffffff !important;
    }
    
    /* Main header styling */
    .main-header {
        background: #ffffff !important;
        padding: 1.5rem 2rem !important;
        margin: -1.5rem -2rem 2rem -2rem !important;
        border-bottom: 1px solid #e2e8f0 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    }
    
    .main-header h1 {
        color: #1e293b !important;
        margin: 0 0 0.5rem 0 !important;
        font-size: 1.875rem !important;
        font-weight: 700 !important;
    }
    
    .main-header p {
        color: #475569 !important;
        margin: 0 !important;
        font-size: 1.1rem !important;
        line-height: 1.5 !important;
    }
    
    /* Text and Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #111827 !important;
        margin-bottom: 1rem;
    }
    
    p, span, div {
        color: #374151 !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #1a365d !important;
    }
    
    /* Cards */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f8fafc;
        padding: 4px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 8px 16px;
        transition: all 0.3s ease;
        margin: 0 2px;
        color: #334155;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: #4f46e5 !important;
        color: white !important;
        font-weight: 600;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-color: #4338ca;
    }
    
    /* Buttons */
    .stButton>button {
        border: 1px solid #e2e8f0 !important;
        color: #4a5568 !important;
        background-color: #f8fafc !important;
        border-radius: 8px !important;
        transition: all 0.3s !important;
        box-shadow: none !important;
    }
    
    .stButton>button:hover {
        background-color: #f1f5f9 !important;
        border-color: #cbd5e0 !important;
        color: #1e293b !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
    }
    
    /* Status badges */
    .status-badge {
        padding: 0.4rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        background-color: #f8fafc;
        color: #1e293b;
        border: 1px solid #e2e8f0;
    }
    
    /* Metrics Cards */
    .metric-card {
        background: white !important;  /* Lighter background */
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
        transition: all 0.3s ease !important;
    }
    
    .metric-card:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
    }
    
    .metric-card .stMetric {
        margin: 0 !important;
    }
    
    .metric-card .stMetricLabel, 
    .metric-card .stMetricValue, 
    .metric-card .stMetricDelta,
    .stMetric,
    .stMetricLabel, 
    .stMetricValue, 
    .stMetricDelta {
        color: #1e293b !important;  /* Darker text */
    }
    
    .metric-card .stMetricValue {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
    }
    
    .metric-card .stMetricLabel {
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        opacity: 0.8;
    }
    
    /* Control Panel Card */
    .control-panel {
        background-color: white !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    }
    
    .control-panel h2, 
    .control-panel h3, 
    .control-panel h4, 
    .control-panel p,
    .control-panel label,
    .control-panel .stTextInput,
    .control-panel .stSelectbox,
    .control-panel .stSlider {
        color: #1e293b !important;
    }
    
    /* Partner Cards */
    /* User Cards */
    .user-card {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin-bottom: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .user-card:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
    }
    
    .user-card h3 {
        color: #1e293b !important;
        margin-top: 0 !important;
        margin-bottom: 0.75rem !important;
        font-size: 1.25rem !important;
    }
    
    .user-card p {
        color: #475569 !important;
        margin: 0.25rem 0 !important;
        font-size: 0.95rem !important;
    }
    
    .user-card .stButton>button {
        margin-top: 1rem !important;
        width: 100% !important;
    }
    
    /* Partner Cards */
    .partner-card {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin-bottom: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .partner-card:hover {
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Application Cards */
    .application-card {
        background: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin-bottom: 1.25rem !important;
        transition: all 0.3s ease !important;
        cursor: pointer;
        color: #1e293b !important;
    }
    
    .application-card:hover {
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
        transform: translateY(-2px) !important;
    }
    
    .application-card h3 {
        color: #1e293b !important;
        margin-top: 0 !important;
        margin-bottom: 0.75rem !important;
    }
    
    .application-card p, 
    .application-card .stText {
        color: #475569 !important;
        margin: 0.25rem 0 !important;
    }
    
    .partner-card h3, 
    .partner-card p,
    .partner-card .stButton>button {
        color: #1e293b !important;
    }
    
    .status-approved { 
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white; 
        box-shadow: 0 2px 4px rgba(16, 185, 129, 0.2);
        font-weight: 600;
    }
    .status-pending { 
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        box-shadow: 0 2px 4px rgba(245, 158, 11, 0.2);
        font-weight: 600;
    }
    .status-declined { 
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        box-shadow: 0 2px 4px rgba(239, 68, 68, 0.2);
        font-weight: 600;
    }
    .status-review { 
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    }
    
    .application-card h4 {
        color: #1e293b !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        margin: 0 0 0.5rem 0 !important;
    }
    
    .application-card div[style*="color: rgb"],
    .application-card span[style*="color: rgb"] {
        color: #4a5568 !important;
    }
    
    .application-card small {
        color: #718096 !important;
    }
    
    .application-card .status-badge {
        background-color: #f8fafc !important;
        color: #1e293b !important;
        border: 1px solid #e2e8f0 !important;
    }
    
    .application-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Form elements */
    .stTextInput>div>div>input, 
    .stSelectbox>div>div>div>div {
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        background-color: white !important;
        color: #1e293b !important;
        padding: 0.5rem 1rem !important;
    }
    
    .stSelectbox>div>div>div>div:hover {
        border-color: #cbd5e0 !important;
    }
    
    .stSelectbox>div>div>div[aria-expanded="true"]>div {
        border-color: #4f46e5 !important;
        box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.2) !important;
    }
    
    .stSelectbox [class*="singleValue"] {
        color: #1e293b !important;
    }
    
    /* Metrics */
    .stMetric {
        background: #ffffff !important;
        padding: 1.5rem 1rem !important;
        border-radius: 12px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
        border: 1px solid #e2e8f0 !important;
        color: #1e293b !important;
        transition: all 0.3s ease !important;
    }
    
    .stMetric > div > div:first-child {
        color: #4a5568 !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.025em !important;
        opacity: 1 !important;
    }
    
    .stMetric > div > div:last-child {
        color: #1e293b !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        text-shadow: none !important;
    }
    
    /* Table */
    .dataframe {
        width: 100%;
        border-collapse: collapse;
    }
    
    .dataframe th, .dataframe td {
        padding: 0.75rem;
        text-align: left;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .dataframe th {
        background-color: #f7fafc;
        font-weight: 600;
        color: #4a5568;
    }
    
    .dataframe tr:hover {
        background-color: #f7fafc;
    }
    </style>
""", unsafe_allow_html=True)

# Mock data generation
def generate_recent_applications(n=20):
    """Generate mock application data."""
    np.random.seed(42)
    
    # Generate random timestamps in the last 7 days
    now = datetime.now()
    timestamps = [now - timedelta(days=np.random.uniform(0, 7)) for _ in range(n)]
    
    # Generate random partner IDs
    partner_ids = [f"P{10000 + i}" for i in range(n)]
    
    # Random scores and bands
    scores = np.clip(np.random.normal(65, 15, n), 0, 100).astype(int)
    bands = ["Approve" if x >= 70 else "Review" if x >= 50 else "Decline" for x in scores]
    
    # Random regions
    regions = ["North", "South", "East", "West", "Central"]
    
    # Create DataFrame
    data = {
        "Partner ID": partner_ids,
        "Score": scores,
        "Band": bands,
        "Region": np.random.choice(regions, n, p=[0.2, 0.3, 0.15, 0.15, 0.2]),
        "Timestamp": timestamps,
        "Status": ["Pending"] * n,
        "Assigned To": [""] * n
    }
    
    # Mark some as completed
    for i in range(min(5, n)):
        data["Status"][i] = np.random.choice(["Approved", "Declined", "More Info Requested"], 
                                           p=[0.6, 0.3, 0.1])
        data["Assigned To"][i] = "agent@example.com"
    
    df = pd.DataFrame(data)
    
    # Sort by timestamp (newest first)
    df = df.sort_values("Timestamp", ascending=False).reset_index(drop=True)
    
    return df

def get_application_details(partner_id):
    """Get detailed application data for a partner."""
    # In a real app, this would fetch from a database
    return {
        "partner_id": partner_id,
        "name": f"Partner {partner_id[1:]}",
        "type": np.random.choice(["Driver", "Merchant"], p=[0.7, 0.3]),
        "score": np.random.randint(30, 95),
        "on_time_rate": round(np.random.uniform(0.7, 0.98), 2),
        "cancel_rate": round(np.random.uniform(0.01, 0.2), 2),
        "avg_rating": round(np.random.uniform(3.0, 5.0), 1),
        "tenure_months": np.random.randint(1, 36),
        "region": np.random.choice(["North", "South", "East", "West", "Central"]),
        "last_updated": (datetime.now() - timedelta(days=np.random.randint(0, 7))).strftime("%Y-%m-%d %H:%M")
    }

def show_application_modal(partner_id):
    """Show application details in a modal."""
    details = get_application_details(partner_id)
    
    with st.expander(f"Application Details - {partner_id}", expanded=True):
        st.markdown(f"### {details['name']} ({details['type']})")
        
        # Score and metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Score", f"{details['score']}/100")
            st.metric("On-time Rate", f"{details['on_time_rate']:.0%}")
            
        with col2:
            st.metric("Cancel Rate", f"{details['cancel_rate']:.0%}")
            st.metric("Avg Rating", f"{details['avg_rating']}/5.0")
        
        # SHAP values (mock)
        st.markdown("### Key Factors")
        
        # Mock SHAP values
        factors = {
            "On-time Rate": max(0.1, min(0.9, details['on_time_rate'] - 0.2)),
            "Cancel Rate": -max(0.1, min(0.9, details['cancel_rate'] * 2)),
            "Avg Rating": max(0.1, min(0.9, (details['avg_rating'] - 3) / 2)),
            "Tenure": max(0.1, min(0.9, details['tenure_months'] / 50)),
            "Region": np.random.uniform(-0.3, 0.3)
        }
        
        # Sort by absolute value
        sorted_factors = sorted(factors.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Display as horizontal bars
        for factor, value in sorted_factors:
            color = "green" if value > 0 else "red"
            width = abs(value) * 100
            st.markdown(
                f"<div style='margin: 5px 0;'>{factor}: "
                f"<div style='background: {color}; width: {width}%; height: 20px; display: flex; align-items: center; padding-left: 10px; color: white;'>{value:+.2f}</div>"
                "</div>",
                unsafe_allow_html=True
            )
        
        # Initialize session state for form if not exists
        form_key = f'form_{partner_id}'
        if form_key not in st.session_state:
            st.session_state[form_key] = {
                'notes': '',
                'action': None,
                'processed': False
            }
        
        # Get the form state
        form_state = st.session_state[form_key]
        
        # Display form
        with st.form(key=f'review_form_{partner_id}'):
            st.markdown("### Review Notes")
            notes = st.text_area("Add internal notes", 
                              value=form_state['notes'],
                              height=100, 
                              placeholder="Add your review notes here...")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                approve_clicked = st.form_submit_button("✅ Approve", use_container_width=True, type="primary")
            
            with col2:
                decline_clicked = st.form_submit_button("❌ Decline", use_container_width=True, type="secondary")
            
            with col3:
                request_info_clicked = st.form_submit_button("📄 Request Info", use_container_width=True)
            
            if st.form_submit_button:
                if approve_clicked:
                    form_state.update({
                        'action': 'approved',
                        'notes': notes,
                        'processed': True
                    })
                    add_success_message(f"✅ Successfully approved {partner_id}")
                    st.session_state.last_action = f"Approved {partner_id}"
                    st.session_state.last_action_time = datetime.now()
                
                elif decline_clicked:
                    form_state.update({
                        'action': 'declined',
                        'notes': notes,
                        'processed': True
                    })
                    add_error_message(f"❌ Application {partner_id} has been declined")
                    st.session_state.last_action = f"Declined {partner_id}"
                    st.session_state.last_action_time = datetime.now()
                
                elif request_info_clicked:
                    form_state.update({
                        'action': 'info_requested',
                        'notes': notes,
                        'processed': True
                    })
                    st.warning(f"ℹ️ Additional information requested for {partner_id}")
                    st.session_state.last_action = f"Requested info for {partner_id}"
                    st.session_state.last_action_time = datetime.now()
                
                # Update the session state
                st.session_state[form_key] = form_state
                
                # Clear the form
                st.session_state[form_key]['notes'] = ''
                st.rerun()

def main():
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.5rem 0 0;
        opacity: 0.9;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    </style>
    <div class='main-header'>
        <h1>🖥️ ControlDesk</h1>
        <p>Streamlined partner management with real-time insights and controls</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Filters
    st.sidebar.header("Filters")
    
    # Status filter
    status_options = ["All", "Pending", "Approved", "Declined", "More Info Requested"]
    selected_status = st.sidebar.selectbox("Status", status_options)
    
    # Region filter
    regions = ["All"] + ["North", "South", "East", "West", "Central"]
    selected_region = st.sidebar.selectbox("Region", regions)
    
    # Date range
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(
            datetime.now() - timedelta(days=7),
            datetime.now()
        ),
        max_value=datetime.now()
    )
    
    # Load data
    df = generate_recent_applications()
    
    # Apply filters
    if selected_status != "All":
        df = df[df["Status"] == selected_status]
    
    if selected_region != "All":
        df = df[df["Region"] == selected_region]
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        df = df[
            (df["Timestamp"].dt.date >= start_date) & 
            (df["Timestamp"].dt.date <= end_date)
        ]
    
    # Show summary stats with animation
    stats_col1, stats_col2, stats_col3 = st.columns([2, 3, 2])
    
    with stats_col1:
        st.metric("Total Applications", len(df))
    
    with stats_col2:
        if hasattr(st.session_state, 'last_action'):
            time_diff = (datetime.now() - st.session_state.get('last_action_time', datetime.now())).total_seconds()
            if time_diff < 10:  # Only show recent actions
                st.success(f"✅ {st.session_state.last_action}")
            else:
                st.info(f"Last action: {st.session_state.last_action}")
    
    with stats_col3:
        if st.button("🔄 Refresh Data", use_container_width=True):
            with st.spinner("Refreshing data..."):
                time.sleep(0.5)
                st.rerun()
    
    st.markdown("---")
    
    # Display table
    st.dataframe(
        df[["Partner ID", "Score", "Band", "Region", "Timestamp", "Status", "Assigned To"]],
        column_config={
            "Timestamp": st.column_config.DatetimeColumn("Date"),
            "Score": st.column_config.ProgressColumn(
                "Score",
                format="%d",
                min_value=0,
                max_value=100,
            ),
        },
        hide_index=True,
        use_container_width=True
    )
    
    # Show details for selected application with animation
    st.markdown("### Application Details")
    
    # Add search and filter row
    search_col, filter_col = st.columns([3, 1])
    
    with search_col:
        search_term = st.text_input("Search by Partner ID", "", 
                                 placeholder="Enter Partner ID...")
    
    with filter_col:
        status_filter = st.selectbox("Status", ["All"] + df["Status"].unique().tolist())
    
    # Filter applications
    filtered_df = df.copy()
    if search_term:
        filtered_df = filtered_df[filtered_df["Partner ID"].str.contains(search_term, case=False)]
    if status_filter != "All":
        filtered_df = filtered_df[filtered_df["Status"] == status_filter]
    
    # Display filtered applications with hover effects
    st.markdown("""
    <style>
        .application-card {
            padding: 15px;
            border-radius: 10px;
            margin: 5px 0;
            transition: all 0.3s ease;
            border: 1px solid #e0e0e0;
        }
        .application-card:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }
        .status-approved { background-color: #e8f5e9; }
        .status-declined { background-color: #ffebee; }
        .status-pending { background-color: #fff8e1; }
        .status-review { background-color: #e3f2fd; }
    </style>
    """, unsafe_allow_html=True)
    
    # Display application cards
    for _, row in filtered_df.iterrows():
        status_class = {
            'Approved': 'status-approved',
            'Pending': 'status-pending',
            'Declined': 'status-declined',
            'Review': 'status-review',
            'More Info Requested': 'status-pending'
        }.get(row['Status'], 'status-pending')
        
        st.markdown(f"""
        <div class="application-card" style="cursor: pointer;" onclick="document.getElementById('select_{row['Partner ID']}').click()">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h4 style="margin: 0 0 0.5rem 0;">{row['Partner ID']} - {row['Score']}/100</h4>
                    <div style="display: flex; align-items: center; gap: 1rem;">
                        <span class="status-badge {status_class}">{row['Status']}</span>
                        <span style="color: #4a5568;">• {row['Region']}</span>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.9rem; color: #4a5568;">{row['Region']}</div>
                    <small style="color: #718096;">{row['Timestamp'].strftime('%b %d, %Y')}</small>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Hidden button to trigger selection
        st.markdown(f'<button type="button" id="select_{row['Partner ID']}" style="display:none;"></button>', 
                   unsafe_allow_html=True)
        if st.button(f"Select {row['Partner ID']}", key=f"btn_select_{row['Partner ID']}"):
            selected_id = row['Partner ID']
    
    # Dropdown for manual selection (fallback)
    selected_id = st.selectbox("Or select an application", 
                              [""] + df["Partner ID"].tolist(),
                              key="manual_select")
    
    if selected_id:
        show_application_modal(selected_id)
    
    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("← Back to Home"):
            st.session_state.page = "home"
            st.rerun()
    
    with col2:
        if st.button("View Fairness Dashboard →"):
            st.session_state.page = "fairness"
            st.rerun()

if __name__ == "__main__":
    main()
