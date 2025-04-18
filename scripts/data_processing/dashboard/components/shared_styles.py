import streamlit as st

# Common styling configurations
def apply_shared_styles():
    """Apply shared styles across the dashboard"""
    st.set_page_config(
        page_title="PSX Stock Analysis Dashboard",
        page_icon="📈",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

# Color scheme
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ffbb33',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40'
}

# Chart configurations
CHART_CONFIG = {
    'displayModeBar': True,
    'displaylogo': False,
    'responsive': True
} 