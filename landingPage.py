import streamlit as st

# Page configuration
st.set_page_config(
    page_title="DZA PriceSight | AI Price Intelligence",
    page_icon="🍉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
    <style>
    /* Dark theme colors */
    .stApp {
        background-color: #1a1a1a;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #262626;
        border-right: 1px solid #3a3a3a;
    }
    
    [data-testid="stSidebar"] .element-container {
        color: #FAFAFA;
    }
    
    /* Cards */
    .kpi-card {
        background-color: #262626;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #3a3a3a;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 16px;
    }
    
    .category-card {
        background-color: #262626;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #3a3a3a;
        cursor: pointer;
        transition: all 0.3s;
        text-align: center;
        margin-bottom: 16px;
    }
    
    .category-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
        border-color: #4CAF50;
    }
    
    .insight-card {
        background-color: #262626;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #3a3a3a;
        border-left: 4px solid #4CAF50;
        margin-bottom: 16px;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .badge-success {
        background-color: rgba(76, 175, 80, 0.2);
        color: #66BB6A;
    }
    
    .badge-warning {
        background-color: rgba(255, 152, 0, 0.2);
        color: #FFA726;
    }
    
    .badge-danger {
        background-color: rgba(244, 67, 54, 0.2);
        color: #EF5350;
    }
    
    .badge-info {
        background-color: rgba(33, 150, 243, 0.2);
        color: #42A5F5;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #FAFAFA !important;
    }
    
    p, div, span {
        color: #B0B0B0;
    }
    
    .big-number {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FAFAFA;
        margin: 8px 0;
    }
    
    .label {
        font-size: 0.9rem;
        color: #808080;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Header */
    .header-container {
        background-color: #262626;
        padding: 20px 30px;
        border-radius: 12px;
        border: 1px solid #3a3a3a;
        margin-bottom: 24px;
    }
    
    .watermelon-icon {
        font-size: 2rem;
        margin-right: 12px;
    }
    
    .app-title {
        font-size: 1.8rem;
        font-weight: bold;
        color: #FAFAFA;
        margin: 0;
    }
    
    .app-subtitle {
        font-size: 0.95rem;
        color: #808080;
        margin: 0;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3);
    }
    
    /* Remove default streamlit padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Fix plotly dark mode */
    .js-plotly-plot .plotly .bg {
        fill: #1a1a1a !important;
    }
    
    </style>
""", unsafe_allow_html=True)

# ============== Sidebar Navigation ==============
with st.sidebar:
    # Logo and title
    st.markdown("""
        <div style='text-align: left; padding: 20px 0;'>
            <span class='watermelon-icon'>🍉</span>
            <div style='display: inline-block; vertical-align: middle;'>
                <div class='app-title'>DZA PriceSight</div>
                <div class='app-subtitle'>AI Price Intel</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main Navigation
    st.markdown("<p class='label'>Main</p>", unsafe_allow_html=True)
    st.page_link("landingPage.py", label="📊 Overview", icon="🏠")
    
    st.markdown("---")
    
    # Categories
    st.markdown("<p class='label'>Categories</p>", unsafe_allow_html=True)
    st.page_link("pages/5_Food_Agriculture.py", label="🌾 Food & Agriculture")
    st.page_link("pages/6_Laptop.py", label="💻 Laptops")
    
    # Coming soon categories
    st.markdown("""
        <div style='padding: 8px 12px; margin: 4px 0; color: #808080; opacity: 0.5;'>
            🚗 Cars 🔜
        </div>
        <div style='padding: 8px 12px; margin: 4px 0; color: #808080; opacity: 0.5;'>
            🏠 Immobilier 🔜
        </div>
        <div style='padding: 8px 12px; margin: 4px 0; color: #808080; opacity: 0.5;'>
            📱 Phones 🔜
        </div>
    """, unsafe_allow_html=True)

# ============== Header ==============
st.markdown("""
    <div class='header-container'>
        <div style='display: flex; align-items: center; justify-content: space-between;'>
            <div style='display: flex; align-items: center;'>
                <span class='watermelon-icon'>🍉</span>
                <div>
                    <div class='app-title'>DZA PriceSight</div>
                    <div class='app-subtitle'>AI-Powered Price Intelligence <span class='badge badge-info'>BETA</span></div>
                </div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# ============== Main Landing Page ==============
st.markdown("# 🏠 Welcome to DZA PriceSight")
st.markdown("### Your AI-Powered Price Intelligence Platform for Algeria")

st.markdown("---")

# Quick info
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div class='insight-card'>
            <div style='font-size: 2rem; text-align: center; margin-bottom: 12px;'>📊</div>
            <div style='font-weight: 600; color: #FAFAFA; margin-bottom: 8px; text-align: center;'>Real-Time Analytics</div>
            <div style='text-align: center;'>Track price trends across multiple categories in real-time</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class='insight-card'>
            <div style='font-size: 2rem; text-align: center; margin-bottom: 12px;'>🤖</div>
            <div style='font-weight: 600; color: #FAFAFA; margin-bottom: 8px; text-align: center;'>AI Predictions</div>
            <div style='text-align: center;'>Machine learning powered price forecasts and anomaly detection</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class='insight-card'>
            <div style='font-size: 2rem; text-align: center; margin-bottom: 12px;'>🗺️</div>
            <div style='font-weight: 600; color: #FAFAFA; margin-bottom: 8px; text-align: center;'>Regional Insights</div>
            <div style='text-align: center;'>Compare prices across different regions and markets in Algeria</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Getting Started
st.markdown("## 🚀 Getting Started")

st.markdown("""
Use the sidebar navigation to explore different sections:

- **📊 Overview**: Quick overview of market trends and key metrics
- **🤖 AI Insights**: AI-powered analysis and predictions (Coming Soon)
- **⚖️ Comparison**: Compare prices across categories and regions (Coming Soon)
- **📈 BI Analytics**: Detailed business intelligence analytics (Coming Soon)

### Categories Available:
- **🌾 Food & Agriculture**: Comprehensive food price analytics with predictions and anomaly detection
- **🚗 Cars, 🏠 Immobilier, 📱 Phones, 💻 Laptops**: Coming Soon!
""")

st.markdown("---")

# Footer
st.markdown("""
    <div style='text-align: center; padding: 40px 20px; color: #808080;'>
        <div style='margin-bottom: 8px;'>Powered by Prophet ML & Streamlit</div>
        <div>© 2026 DZA PriceSight - AI Price Intelligence Platform</div>
    </div>
""", unsafe_allow_html=True)
