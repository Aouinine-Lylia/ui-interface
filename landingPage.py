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
        padding: 30px;
        border-radius: 12px;
        border: 1px solid #3a3a3a;
        cursor: pointer;
        transition: all 0.3s;
        text-align: left; /* Changed to left for better list readability */
        margin-bottom: 16px;
        height: 100%;
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
    
    /* Feature List inside Category Cards */
    .feature-list {
        margin-top: 15px;
        padding-left: 0;
        list-style: none;
    }
    
    .feature-list li {
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        color: #B0B0B0;
    }
    
    .feature-list li span.icon {
        margin-right: 10px;
        width: 20px;
        text-align: center;
    }
    
    </style>
""", unsafe_allow_html=True)

# ============== Sidebar Navigation ==============
with st.sidebar:
    # Logo and title

    col1, col2 = st.columns([1, 10])

    with col1:
        st.image("watermelon.png", width=64, use_container_width=False)

    with col2:
        st.markdown(
            "<div style='font-size:28px; font-weight:700; margin-top:-10px;'>DZA PriceSight</div>"
           , unsafe_allow_html=True
        )
    st.markdown("<div style='font-size:14px; color:gray;'>AI Price Insights</div>", unsafe_allow_html=True)
    st.markdown("---")
    # Main Navigation
    st.markdown("<p class='label'>Main</p>", unsafe_allow_html=True)
    st.page_link("landingPage.py", label="Overview", icon="🏠")
    
    st.markdown("---")
    
    # Categories
    st.markdown("<p class='label'>Active Dashboards</p>", unsafe_allow_html=True)
    st.page_link("pages/5_Food_Agriculture.py", label="Food & Agriculture", icon="🌾")
    st.page_link("pages/6_Laptop.py", label="Laptops", icon="💻")

# ============== Header ==============
st.markdown("<div class='header-container'></div>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 10])

with col1:
        st.image("watermelon.png", width=64, use_container_width=False)

with col2:
        st.markdown(
            "<div style='font-size:48px; font-weight:700; margin-top:-6px;'>DZA PriceSight</div>"
           , unsafe_allow_html=True
        )

# ============== Main Landing Page ==============
st.markdown("### ")
st.markdown("### Real-time market intelligence for the Algerian Economy")

st.markdown("---")

# Quick info / Core Features
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div class='insight-card'>
            <div style='font-size: 2rem; text-align: center; margin-bottom: 12px;'>📊</div>
            <div style='font-weight: 600; color: #FAFAFA; margin-bottom: 8px; text-align: center;'>Deep Analytics</div>
            <div style='text-align: center;'>Historical trends, volatility metrics, and executive KPIs across multiple sectors.</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class='insight-card'>
            <div style='font-size: 2rem; text-align: center; margin-bottom: 12px;'>🔮</div>
            <div style='font-weight: 600; color: #FAFAFA; margin-bottom: 8px; text-align: center;'>Prophet Forecasting</div>
            <div style='text-align: center;'>12-month future price projections using advanced time-series machine learning models.</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class='insight-card'>
            <div style='font-size: 2rem; text-align: center; margin-bottom: 12px;'>🤖</div>
            <div style='font-weight: 600; color: #FAFAFA; margin-bottom: 8px; text-align: center;'>AI Insights</div>
            <div style='text-align: center;'>Generative AI analysis of market conditions to detect risks and opportunities.</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Getting Started / Categories
st.markdown("## 🚀 Explore Active Markets")

st.markdown("Select a category below to access detailed dashboards, forecasts, and predictive models.")

col_cat1, col_cat2 = st.columns(2)

# FOOD CARD
with col_cat1:
    # Using a link wrapper with st.markdown for the card effect
    st.markdown("""
        <div class='category-card' onclick="window.location.href='pages/5_Food_Agriculture.py'">
            <div style='display: flex; align-items: center; margin-bottom: 15px;'>
                <span style='font-size: 3rem; margin-right: 15px;'>🌾</span>
                <div>
                    <h3 style='margin: 0; color: #FAFAFA;'>Food & Agriculture</h3>
                </div>
            </div>
            <p style='margin-bottom: 15px; font-size: 1rem;'>
                Comprehensive analysis of staple foods, cereals, and commodities across all 58 Wilayas.
            </p>
            <hr style='border-color: #3a3a3a; margin: 15px 0;'>
            <ul class='feature-list'>
                <li><span class='icon'>📈</span> Prophet-based Price Forecasting</li>
                <li><span class='icon'>🗺️</span> Geographic Heatmap & Regional Pricing</li>
                <li><span class='icon'>⚠️</span> Trend Anomaly & Volatility Detection</li>
                <li><span class='icon'>🥕</span> Price Classification API (Cheap/Normal/Expensive)</li>
                <li><span class='icon'>🤖</span> AI Agricultural Insights</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# LAPTOP CARD
with col_cat2:
    st.markdown("""
        <div class='category-card' onclick="window.location.href='pages/6_Laptop.py'">
            <div style='display: flex; align-items: center; margin-bottom: 15px;'>
                <span style='font-size: 3rem; margin-right: 15px;'>💻</span>
                <div>
                    <h3 style='margin: 0; color: #FAFAFA;'>Laptops & Hardware</h3>
                </div>
            </div>
            <p style='margin-bottom: 15px; font-size: 1rem;'>
                Market intelligence for the Algerian tech sector, tracking hardware specs and pricing.
            </p>
            <hr style='border-color: #3a3a3a; margin: 15px 0;'>
            <ul class='feature-list'>
                <li><span class='icon'>📈</span> Hardware Tier Forecasting</li>
                <li><span class='icon'>🏙️</span> City-wise Price Comparison</li>
                <li><span class='icon'>🧮</span> Regression-based Price Prediction Tool</li>
                <li><span class='icon'>📊</span> Market Segmentation (CPU/GPU)</li>
                <li><span class='icon'>🤖</span> AI Tech Market Analysis</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Footer / Updates
st.markdown("## ℹ️ System Updates & Info")
st.markdown("""
<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px;'>
    <div style='background: #262626; padding: 15px; border-radius: 8px; border: 1px solid #3a3a3a;'>
        <div style='font-weight: bold; color: #4CAF50; margin-bottom: 5px;'>New Feature</div>
        <div style='font-size: 0.9rem;'>Forecasting is now live for specific commodities and laptop tiers using the Prophet model.</div>
    </div>
    <div style='background: #262626; padding: 15px; border-radius: 8px; border: 1px solid #3a3a3a;'>
        <div style='font-weight: bold; color: #42A5F5; margin-bottom: 5px;'>AI Integration</div>
        <div style='font-size: 0.9rem;'>Groq Llama 3.1 integration provides instant strategic market insights on every dashboard.</div>
    </div>
    <div style='background: #262626; padding: 15px; border-radius: 8px; border: 1px solid #3a3a3a;'>
        <div style='font-weight: bold; color: #FFA726; margin-bottom: 5px;'>Coming Soon</div>
        <div style='font-size: 0.9rem;'>Automotive (Cars) & Real Estate (Immobilier) dashboards are currently in development.</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style='text-align: center; padding: 40px 20px; color: #808080;'>
        <div>© 2026 DZA PriceSight - AI Price Intelligence Platform</div>
    </div>
""", unsafe_allow_html=True)