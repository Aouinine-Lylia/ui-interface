import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import requests
import json
import glob
from sklearn.linear_model import LinearRegression

# Page configuration
st.set_page_config(
    page_title="Food & Agriculture | DZA PriceSight",
    page_icon="🌾",
    layout="wide"
)

# Algerian Wilayas List for Dropdown
ALGERIAN_WILAYAS_LIST = [
    "Adrar", "Chlef", "Laghouat", "Oum El Bouaghi", "Batna", "Béjaïa", "Biskra", "Béchar", 
    "Blida", "Bouira", "Tamanrasset", "Tébessa", "Tlemcen", "Tiaret", "Tizi Ouzou", "Alger", 
    "Djelfa", "Jijel", "Sétif", "Saïda", "Skikda", "Sidi Bel Abbès", "Annaba", "Guelma", 
    "Constantine", "Médéa", "Mostaganem", "Msila", "Mascara", "Ouargla", "Oran", "El Bayadh", 
    "Illizi", "Bordj Bou Arreridj", "Boumerdès", "El Tarf", "Tindouf", "Tissemsilt", "El Oued", 
    "Khenchela", "Souk Ahras", "Tipaza", "Mila", "Aïn Defla", "Naâma", "Aïn Témouchent", 
    "Ghardaïa", "Relizane", "Timimoun", "Bordj Badji Mokhtar", "Ouled Djellal", "Béni Abbès", 
    "In Salah", "In Guezzam", "Touggourt", "Djanet", "El M'Ghair", "El Meniaa"
]

# Dark theme CSS
st.markdown("""
    <style>
    .stApp {background-color: #1a1a1a;}
    [data-testid="stSidebar"] {background-color: #262626; border-right: 1px solid #3a3a3a;}
    [data-testid="stSidebar"] .element-container {color: #FAFAFA;}
    h1, h2, h3, h4, h5, h6 {color: #FAFAFA !important;}
    p, div, span {color: #B0B0B0;}
    .kpi-card {
        background-color: #262626;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #3a3a3a;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 16px;
    }
    .insight-card {
        background-color: #262626;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #4CAF50;
        margin-bottom: 16px;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-success { background-color: rgba(76, 175, 80, 0.2); color: #66BB6A; }
    .badge-danger { background-color: rgba(244, 67, 54, 0.2); color: #EF5350; }
    .badge-warning { background-color: rgba(255, 152, 0, 0.2); color: #FFA726; }
    .badge-info { background-color: rgba(33, 150, 243, 0.2); color: #42A5F5; }
    .big-number { font-size: 2.5rem; font-weight: bold; color: #FAFAFA; margin: 8px 0; }
    .label { font-size: 0.9rem; color: #808080; text-transform: uppercase; letter-spacing: 0.5px; }
    .watermelon-icon { font-size: 2rem; margin-right: 12px; }
    .app-title { font-size: 1.8rem; font-weight: bold; color: #FAFAFA; margin: 0; }
    .app-subtitle { font-size: 0.95rem; color: #808080; margin: 0; }

    /* Enhanced Confidence Bar */
    .confidence-bar-container {
        margin-top: 8px;
        height: 8px;
        background: #3a3a3a;
        border-radius: 4px;
        overflow: hidden;
    }
    .confidence-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #4CAF50, #66BB6A);
        border-radius: 4px;
        transition: width 0.5s ease-in-out;
    }
    
    /* Stats Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin-top: 15px;
        background: #121212;
        padding: 12px;
        border-radius: 8px;
    }
    .stat-item {
        display: flex;
        flex-direction: column;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #808080;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .stat-value {
        font-size: 1.1rem;
        font-weight: bold;
        color: #FAFAFA;
    }
    </style>
""", unsafe_allow_html=True)

# Wilaya to lat/lon mapping
ALGERIAN_WILAYAS_COORDS = {
    "Adrar": {"lat": 27.88, "lon": -0.28},
    "Chlef": {"lat": 36.16, "lon": 1.33},
    "Laghouat": {"lat": 33.80, "lon": 2.88},
    "Oum El Bouaghi": {"lat": 35.86, "lon": 7.11},
    "Batna": {"lat": 35.55, "lon": 6.17},
    "Béjaïa": {"lat": 36.75, "lon": 5.06},
    "Biskra": {"lat": 34.85, "lon": 5.73},
    "Béchar": {"lat": 31.61, "lon": -2.22},
    "Blida": {"lat": 36.47, "lon": 2.83},
    "Bouira": {"lat": 36.37, "lon": 3.90},
    "Tamanrasset": {"lat": 22.78, "lon": 5.52},
    "Tébessa": {"lat": 35.40, "lon": 8.12},
    "Tlemcen": {"lat": 34.88, "lon": -1.31},
    "Tiaret": {"lat": 35.37, "lon": 1.32},
    "Tizi Ouzou": {"lat": 36.70, "lon": 4.05},
    "Alger": {"lat": 36.75, "lon": 3.04},
    "Djelfa": {"lat": 34.67, "lon": 3.26},
    "Jijel": {"lat": 36.80, "lon": 5.76},
    "Sétif": {"lat": 36.19, "lon": 5.41},
    "Saïda": {"lat": 34.83, "lon": 0.14},
    "Skikda": {"lat": 36.87, "lon": 6.91},
    "Sidi Bel Abbès": {"lat": 35.19, "lon": -0.64},
    "Annaba": {"lat": 36.89, "lon": 7.76},
    "Guelma": {"lat": 36.46, "lon": 7.43},
    "Constantine": {"lat": 36.36, "lon": 6.61},
    "Médéa": {"lat": 36.26, "lon": 2.75},
    "Mostaganem": {"lat": 35.93, "lon": 0.09},
    "Msila": {"lat": 35.70, "lon": 4.54},
    "Mascara": {"lat": 35.39, "lon": 0.14},
    "Ouargla": {"lat": 31.95, "lon": 5.32},
    "Oran": {"lat": 35.69, "lon": -0.64},
    "El Bayadh": {"lat": 33.68, "lon": 1.01},
    "Illizi": {"lat": 26.49, "lon": 8.44},
    "Bordj Bou Arreridj": {"lat": 36.07, "lon": 4.76},
    "Boumerdès": {"lat": 36.75, "lon": 3.47},
    "El Tarf": {"lat": 36.77, "lon": 8.30},
    "Tindouf": {"lat": 27.67, "lon": -8.14},
    "Tissemsilt": {"lat": 35.60, "lon": 1.81},
    "El Oued": {"lat": 33.35, "lon": 6.86},
    "Khenchela": {"lat": 35.42, "lon": 7.14},
    "Souk Ahras": {"lat": 36.28, "lon": 7.95},
    "Tipaza": {"lat": 36.58, "lon": 2.45},
    "Mila": {"lat": 36.45, "lon": 6.26},
    "Aïn Defla": {"lat": 36.25, "lon": 1.95},
    "Naâma": {"lat": 33.26, "lon": -0.31},
    "Aïn Témouchent": {"lat": 35.30, "lon": -1.14},
    "Ghardaïa": {"lat": 32.49, "lon": 3.67},
    "Relizane": {"lat": 35.93, "lon": 0.55},
    "Timimoun": {"lat": 29.25, "lon": 0.25},
    "Bordj Badji Mokhtar": {"lat": 21.33, "lon": 0.95},
    "Ouled Djellal": {"lat": 34.52, "lon": 4.72},
    "Béni Abbès": {"lat": 30.13, "lon": -2.16},
    "In Salah": {"lat": 27.20, "lon": 2.47},
    "In Guezzam": {"lat": 23.00, "lon": 5.70},
    "Touggourt": {"lat": 33.10, "lon": 6.06},
    "Djanet": {"lat": 24.55, "lon": 9.47},
    "El M'Ghair": {"lat": 33.30, "lon": 6.30},
    "El Meniaa": {"lat": 30.58, "lon": 2.91}
}

# ============== Sidebar Navigation ==============
with st.sidebar:
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
    st.markdown("<p class='label'>Main</p>", unsafe_allow_html=True)
    st.page_link("landingPage.py", label="📊 Overview", icon="🏠")
    st.markdown("---")
    st.markdown("<p class='label'>Categories</p>", unsafe_allow_html=True)
    st.page_link("pages/5_Food_Agriculture.py", label="🌾 Food & Agriculture")
    st.page_link("pages/6_Laptop.py", label="💻 Laptops")
    
    st.markdown("""
        <div style='padding: 8px 12px; margin: 4px 0; color: #808080; opacity: 0.5;'>🚗 Cars 🔜</div>
        <div style='padding: 8px 12px; margin: 4px 0; color: #808080; opacity: 0.5;'>🏠 Immobilier 🔜</div>
        <div style='padding: 8px 12px; margin: 4px 0; color: #808080; opacity: 0.5;'>📱 Phones 🔜</div>
    """, unsafe_allow_html=True)

# ============== Data Loading ==============
@st.cache_data
def load_food_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    try:
        df_prices = pd.read_csv(os.path.join(parent_dir, 'wfp_food_prices_dza.csv'))
        df_prices['date'] = pd.to_datetime(df_prices['date'])
        df_prices['price'] = pd.to_numeric(df_prices['price'], errors='coerce')
        df_prices = df_prices.dropna(subset=['price', 'date'])
    except Exception as e:
        st.error(f"Error loading main data: {e}")
        return pd.DataFrame(), None, None, None, {}
    
    try:
        df_enhanced = pd.read_csv(os.path.join(os.path.dirname(parent_dir), 'prophet_data_enhanced.csv'))
        df_enhanced['ds'] = pd.to_datetime(df_enhanced['ds'])
    except:
        df_enhanced = None
    
    try:
        df_forecast = pd.read_csv(os.path.join(os.path.dirname(parent_dir), 'outputs/forecast_full.csv'))
        df_forecast['date'] = pd.to_datetime(df_forecast['date'])
    except:
        df_forecast = None
    
    try:
        df_future = pd.read_csv(os.path.join(os.path.dirname(parent_dir), 'outputs/forecast_future.csv'))
        df_future['date'] = pd.to_datetime(df_future['date'])
    except:
        df_future = None
    
    category_data = {}
    try:
        top_commodities = df_prices['commodity'].value_counts().head(6).index.tolist()
        for comm in top_commodities:
            category_data[comm] = df_prices[df_prices['commodity'] == comm].copy()
            category_data[comm] = category_data[comm].rename(columns={'date': 'ds', 'price': 'y'})
    except:
        pass
    
    return df_prices, df_enhanced, df_forecast, df_future, category_data

# Load data
df_prices, df_enhanced, df_forecast, df_future, category_data = load_food_data()

if df_prices.empty:
    st.error("No data available. Please check the CSV file path.")
    st.stop()

# ============== Header Section ==============
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("# 🌾 Food & Agriculture")
    st.markdown("### Detailed analytics and predictions for food & agriculture products")
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if 'show_food_predict_form' not in st.session_state:
        st.session_state.show_food_predict_form = False
    if st.button("🔮 Predict Food Price", use_container_width=True, type="primary"):
        st.session_state.show_food_predict_form = True

# ============== FIXED PREDICTION FORM ==============
if st.session_state.get('show_food_predict_form'):
    with st.expander("🥕 Food Specification Input", expanded=True):
        with st.form("food_predict_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                commodity = st.text_input("Commodity", value="Tomatoes")
                category = st.selectbox("Category", [
                    "cereals and tubers", "meat, fish and eggs", "milk and dairy",
                    "miscellaneous food", "non-food", "oil and fats", "pulses and nuts",
                    "vegetables and fruits"
                ], index=7)
                price_per_kg = st.number_input("Price per kg (DZD)", min_value=0.0, value=200.0, step=1.0)
            
            with col_b:
                wilaya = st.selectbox("Wilaya", options=ALGERIAN_WILAYAS_LIST, index=15)
            
            submit = st.form_submit_button("Predict Price Classification", use_container_width=True)

        if submit:
            # Get coordinates
            lat = ALGERIAN_WILAYAS_COORDS[wilaya]["lat"]
            lon = ALGERIAN_WILAYAS_COORDS[wilaya]["lon"]
            today = datetime.now().strftime("%Y-%m-%d")
            
            payload = {
                "commodity": commodity,
                "category": category,
                "price_per_kg": price_per_kg,
                "date": today,
                "latitude": lat,
                "longitude": lon
            }
            
            try:
                api_url = "http://localhost:5000/predict-food"
                
                with st.spinner("🔄 Analyzing price data..."):
                    res = requests.post(api_url, json=payload, timeout=10)
                
                if res.status_code == 200:
                    result = res.json()
                    
                    # Extract data
                    prediction_label = result.get('prediction', 'Unknown')
                    confidence_score = result.get('confidence', 0.0)
                    deviation_pct = result.get('price_deviation_percentage', 0.0)
                    historical_mean = result.get('historical_mean', 0.0)
                    
                    # Determine styling based on prediction
                    if prediction_label == 'CHEAP':
                        badge_class = 'badge-success'
                        status_text = "Great Deal"
                        status_desc = "Underpriced compared to historical average"
                        icon = "🟢"
                        deviation_color = "#66BB6A"
                    elif prediction_label == 'NORMAL':
                        badge_class = 'badge-info'
                        status_text = "Fair Market Value"
                        status_desc = "Matches historical average closely"
                        icon = "⚖️"
                        deviation_color = "#42A5F5"
                    elif prediction_label == 'EXPENSIVE':
                        badge_class = 'badge-danger'
                        status_text = "Overpriced"
                        status_desc = "Significantly higher than historical average"
                        icon = "🔴"
                        deviation_color = "#EF5350"
                    else:
                        badge_class = 'badge-warning'
                        status_text = "Unknown"
                        status_desc = "Unable to classify price"
                        icon = "❓"
                        deviation_color = "#B0B0B0"

                    confidence_width = f"{confidence_score * 100:.1f}%"
                    
                    # Render results
                    st.markdown(f"""
                        <div class='insight-card'>
                            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;'>
                                <h3 style='margin:0; color: #4CAF50; display: flex; align-items: center; gap: 8px;'>
                                    {icon} AI Price Analysis
                                </h3>
                                <span class='badge {badge_class}'>{status_text}</span>
                            </div>

                            <div style='display: flex; align-items: baseline; margin-bottom: 15px;'>
                                <span style='font-size: 1.2rem; color: #B0B0B0; margin-right: 10px;'>Your Input:</span>
                                <span style='font-size: 2.5rem; font-weight: bold; color: #FAFAFA;'>{price_per_kg:.0f} DZD</span>
                            </div>

                            <div style='font-size: 0.9rem; color: #B0B0B0; margin-bottom: 20px; line-height: 1.4;'>
                                {status_desc} for <strong>{commodity}</strong> in <strong>{wilaya}</strong>.
                            </div>
                            
                            <div style='margin-bottom: 20px;'>
                                <div style='display: flex; justify-content: space-between; margin-bottom: 4px;'>
                                    <span class='label'>Model Confidence</span>
                                    <span style='font-weight: bold; color: #4CAF50;'>{confidence_width}</span>
                                </div>
                                <div class='confidence-bar-container'>
                                    <div class='confidence-bar-fill' style='width: {confidence_width};'></div>
                                </div>
                            </div>

                            <div class='stats-grid'>
                                <div class='stat-item'>
                                    <span class='stat-label'>Price Deviation</span>
                                    <span class='stat-value' style='color: {deviation_color};'>
                                        {deviation_pct:+.1f}%
                                    </span>
                                </div>
                                <div class='stat-item'>
                                    <span class='stat-label'>Historical Mean</span>
                                    <span class='stat-value'>{historical_mean:.0f} DZD</span>
                                </div>
                            </div>
                        </div> 
                    """, unsafe_allow_html=True)
                    
                else:
                    st.error(f"❌ API Error: {res.status_code} - {res.text}")
                    
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. Please check if the API server is running.")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Connection error. Make sure the API is running at http://localhost:5000")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")

st.markdown("---")

# ============== Filters ==============
st.markdown("### 🔍 Data Filters")

all_commodities = sorted(df_prices['commodity'].unique().tolist())
all_regions = sorted(df_prices['admin1'].unique().tolist()) if 'admin1' in df_prices.columns else []

col_filter1, col_filter2, col_date = st.columns(3)

with col_filter1:
    selected_commodities = st.multiselect("Select Commodity", all_commodities, default=None)

with col_filter2:
    if all_regions:
        selected_regions = st.multiselect("Select Region", all_regions, default=[])
    else:
        selected_regions = []

df_filtered = df_prices.copy()
if selected_commodities:
    df_filtered = df_filtered[df_filtered['commodity'].isin(selected_commodities)]
if selected_regions:
    df_filtered = df_filtered[df_filtered['admin1'].isin(selected_regions)]

min_date = df_filtered['date'].min()
max_date = df_filtered['date'].max()

with col_date:
    default_start = min_date
    date_range = st.date_input("Date Range", [default_start, max_date])

if len(date_range) == 2:
    df_filtered = df_filtered[(df_filtered['date'] >= pd.to_datetime(date_range[0])) & 
                              (df_filtered['date'] <= pd.to_datetime(date_range[1]))]

if df_filtered.empty:
    st.warning("No data matches your filters. Adjust selections.")
    st.stop()

# ============== KPI Calculation ==============
df_filtered = df_filtered.sort_values('date')
avg_price = df_filtered['price'].mean()
max_d = df_filtered['date'].max()
last_period = df_filtered[df_filtered['date'] >= (max_d - timedelta(days=30))]
prev_period = df_filtered[(df_filtered['date'] >= (max_d - timedelta(days=60))) & 
                          (df_filtered['date'] < (max_d - timedelta(days=30)))]

if not prev_period.empty and not last_period.empty:
    curr_mean = last_period['price'].mean()
    prev_mean = prev_period['price'].mean()
    price_change = ((curr_mean - prev_mean) / prev_mean) * 100
    price_change_str = f"{'+' if price_change >= 0 else ''}{price_change:.1f}%"
    price_change_class = 'badge-success' if price_change >= 0 else 'badge-danger'
else:
    price_change_str = "N/A"
    price_change_class = 'badge-warning'

price_std = df_filtered['price'].std()
volatility_coef = (price_std / avg_price) * 100 if avg_price > 0 else 0
vol_str = f"{volatility_coef:.1f}%"

sample_size_score = min(len(df_filtered) / 100, 1.0) * 50
volatility_score = max(0, 50 - volatility_coef)
confidence = min(max(sample_size_score + volatility_score, 10.0), 99.0)

active_listings = len(df_filtered)

st.markdown("### 📊 Key Metrics")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(f"""
        <div class='kpi-card'>
            <div style='display: flex; justify-content: space-between; align-items: flex-start;'>
                <div style='font-size: 2rem;'>💰</div>
                <span class='badge {price_change_class}'>{price_change_str}</span>
            </div>
            <div class='label' style='margin-top: 16px;'>Average Price</div>
            <div class='big-number'>{avg_price:.0f} DZD</div>
            <div style='font-size: 0.85rem; color: #808080;'>Selected period</div>
        </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
        <div class='kpi-card'>
            <div style='display: flex; justify-content: space-between; align-items: flex-start;'>
                <div style='font-size: 2rem;'>📉</div>
                <span class='badge badge-warning'>{vol_str}</span>
            </div>
            <div class='label' style='margin-top: 16px;'>Volatility (CV)</div>
            <div class='big-number'>{price_std:.1f}</div>
            <div style='font-size: 0.85rem; color: #808080;'>Std Dev of Price</div>
        </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
        <div class='kpi-card'>
            <div style='font-size: 2rem; margin-bottom: 16px;'>🎯</div>
            <div class='label'>Data Confidence</div>
            <div class='big-number'>{confidence:.1f}%</div>
            <div style='font-size: 0.85rem; color: #808080;'>Based on density & variance</div>
            <div style='margin-top: 12px; height: 32px; background: #3a3a3a; border-radius: 4px;'>
                <div style='width: {confidence}%; height: 100%; background: #4CAF50; border-radius: 4px;'></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
        <div class='kpi-card'>
            <div style='font-size: 2rem; margin-bottom: 16px;'>📦</div>
            <div class='label'>Data Points</div>
            <div class='big-number'>{active_listings:,}</div>
            <div style='font-size: 0.85rem; color: #808080;'>Across {df_filtered['market'].nunique()} markets</div>
        </div>
    """, unsafe_allow_html=True)

# ============== Charts Section ==============
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📈 Historical Price Trend")
    st.markdown("<div style='color: #808080; margin-bottom: 16px;'>Monthly average evolution</div>", unsafe_allow_html=True)
    
    # Group by month for cleaner chart
    trend_data = df_filtered.groupby(pd.Grouper(key='date', freq='M'))['price'].mean().reset_index()
    
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=trend_data['date'], 
        y=trend_data['price'],
        mode='lines+markers',
        name='Price',
        line=dict(color='#4CAF50', width=3),
        marker=dict(size=6)
    ))
    
    fig_trend.update_layout(
        template='plotly_dark',
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#262626',
        font=dict(color='#FAFAFA'),
        xaxis_title="Date",
        yaxis_title="Price (DZD)",
        height=400,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with col2:
    st.markdown("### 📊 Price Distribution")
    st.markdown("<div style='color: #808080; margin-bottom: 16px;'>Frequency of price points</div>", unsafe_allow_html=True)
    
    fig_dist = px.histogram(
        df_filtered, 
        x="price", 
        nbins=30,
        color_discrete_sequence=['#4CAF50']
    )
    
    fig_dist.update_layout(
        template='plotly_dark',
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#262626',
        font=dict(color='#FAFAFA'),
        xaxis_title="Price (DZD)",
        yaxis_title="Count",
        bargap=0.1,
        height=400,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig_dist, use_container_width=True)

st.markdown("---")

# ============== Map Visualization (NEW) ==============
st.markdown("### 🗺️ Market Distribution")
st.markdown("<div style='color: #808080; margin-bottom: 16px;'>Geographic price tracking</div>", unsafe_allow_html=True)

# Map Logic
if 'latitude' in df_filtered.columns and 'longitude' in df_filtered.columns:
    # Group by location to get average price per city
    map_data = df_filtered.groupby(['admin1', 'market', 'latitude', 'longitude'])['price'].mean().reset_index()
    
    # Filter out entries with no coordinates
    map_data = map_data.dropna(subset=['latitude', 'longitude'])
    
    if not map_data.empty:
        fig_map = px.scatter_mapbox(
            map_data,
            lat="latitude",
            lon="longitude",
            size="price",
            color="price",
            hover_name="market",
            hover_data={"admin1": True, "price": ":.2f"},
            color_continuous_scale=px.colors.sequential.Viridis,
            size_max=25,
            zoom=5,
            center={"lat": 34.0, "lon": 3.0} # Centered on Algeria
        )
        
        fig_map.update_layout(
            mapbox_style="carto-darkmatter",
            margin={"r":0,"t":0,"l":0,"b":0},
            paper_bgcolor='#1a1a1a',
            height=500
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("No coordinate data available for the selected regions.")
else:
    st.info("Map data not available (missing latitude/longitude).")

st.markdown("---")

# ============== Forecast Logic (Robust) ==============

if df_forecast is not None and not df_forecast.empty:
    future_dates = df_forecast['date']
    predictions = df_forecast['predicted_price'] if 'predicted_price' in df_forecast.columns else None
    upper = df_forecast['upper_bound'] if 'upper_bound' in df_forecast.columns else None
    lower = df_forecast['lower_bound'] if 'lower_bound' in df_forecast.columns else None
else:
    future_dates, predictions, upper, lower = None, None, None, None

if predictions is not None:
    pred_price = predictions.iloc[-1]
    curr_avg = df_filtered['price'].iloc[-1]
    pred_change = ((pred_price - curr_avg) / curr_avg) * 100
    pred_change_str = f"{'+' if pred_change >= 0 else ''}{pred_change:.1f}%"
    pred_change_class = 'badge-success' if pred_change >= 0 else 'badge-danger'
    lower_bound_val = lower.iloc[-1] if lower is not None else 0
    upper_bound_val = upper.iloc[-1] if upper is not None else 0
else:
    pred_price = 0
    pred_change_str = "N/A"
    lower_bound_val = 0
    upper_bound_val = 0
    pred_change_class = 'badge-warning'

# ============== Drivers Calculation (Real Stats) ==============
# 1. Seasonality: Compare current month avg to same month last year
current_month = df_filtered['date'].max().month
this_year_month_data = df_filtered[df_filtered['date'].dt.month == current_month]
if not this_year_month_data.empty:
    seasonality_impact = "High" if len(this_year_month_data) > len(df_filtered)/12 * 1.5 else "Normal"
else:
    seasonality_impact = "Unknown"

# 2. Regional Variance (Competition)
if 'admin1' in df_filtered.columns:
    region_variance = df_filtered.groupby('admin1')['price'].mean().std()
    competition_score = (region_variance / avg_price) * 100 if avg_price > 0 else 0
    comp_str = f"{competition_score:.1f}% Variance"
else:
    comp_str = "N/A"

# 3. Trend (Slope)
if len(df_filtered) > 1:
    first_p = df_filtered['price'].iloc[0]
    last_p = df_filtered['price'].iloc[-1]
    overall_trend = ((last_p - first_p)/first_p) * 100
    trend_str = f"{'+' if overall_trend >= 0 else ''}{overall_trend:.1f}%"
else:
    trend_str = "0%"


# ============== New Forecast Logic (Individual Item Selection) ==============

# 1. Define path to forecasting folder
# Assuming dashboard is in parent dir of training-food
FORECAST_DIR = os.path.join(os.path.dirname(__file__), "forecasting")

# Check if folder exists
if os.path.exists(FORECAST_DIR):
    # 2. Find all CSV files and extract commodity names
    forecast_files = glob.glob(os.path.join(FORECAST_DIR, '*.csv'))
    available_forecasts = []
    
    for file in forecast_files:
        # Extract name: "Pasta_forecast.csv" -> "Pasta"
        filename = os.path.basename(file)
        name = filename.replace('_forecast.csv', '').replace('_', ' ')
        available_forecasts.append(name)
    
    # Sort alphabetically
    available_forecasts.sort()

    # 3. UI: Select Box to choose item
    st.markdown("### 🔮 Select Item to Forecast")
    st.markdown("<div style='color: #808080; margin-bottom: 16px;'>Choose a specific commodity to view its 12-month prediction</div>", unsafe_allow_html=True)
    
    selected_forecast_item = st.selectbox(
        "Available Forecasts",
        available_forecasts,
        label_visibility="collapsed"
    )
    
    # 4. Load the specific forecast data
    # Reconstruct the filename (reverse the cleaning process)
    safe_name = selected_forecast_item.replace(' ', '_')
    forecast_path = os.path.join(FORECAST_DIR, f"{safe_name}_forecast.csv")
    
    try:
        df_future = pd.read_csv(forecast_path)
        # Rename columns to match your dashboard's standard naming
        df_future.columns = ['date', 'predicted_price', 'lower_bound', 'upper_bound']
        df_future['date'] = pd.to_datetime(df_future['date'])
        
        # 5. Load Historical Data for THIS specific commodity
        # We filter the main dataset to get history for the selected item
        comm_history = df_prices[df_prices['commodity'] == selected_forecast_item].copy()
        
        # Group by month to match the forecast frequency
        comm_history_monthly = comm_history.groupby(pd.Grouper(key='date', freq='MS'))['price'].mean().reset_index()
        comm_history_monthly.columns = ['date', 'price']
        
        # Calculate KPIs for the specific item
        if not comm_history_monthly.empty:
            last_hist_price = comm_history_monthly['price'].iloc[-1]
            next_pred = df_future['predicted_price'].iloc[0]
            pred_change = ((next_pred - last_hist_price) / last_hist_price) * 100
        else:
            last_hist_price = 0
            pred_change = 0
            
        # Display KPIs
        kpi_f1, kpi_f2, kpi_f3 = st.columns(3)
        
        # Trend Color Logic
        pred_change_class = 'badge-success' if pred_change >= 0 else 'badge-danger'
        pred_change_str = f"{'+' if pred_change >= 0 else ''}{pred_change:.1f}%"
        
        with kpi_f1:
            st.markdown(f"""
                <div class='kpi-card'>
                    <div class='label'>Next Month Prediction</div>
                    <div class='big-number' style='color: #2196F3;'>{next_pred:.0f} DZD</div>
                    <div style='margin-top: 12px;'>
                        <span class='badge {pred_change_class}'>{pred_change_str}</span>
                    </div>
                    <div style='font-size: 0.8rem; color: #808080; margin-top: 5px;'>vs last month avg</div>
                </div>
            """, unsafe_allow_html=True)

        with kpi_f2:
            # Calculate uncertainty range (Average width of the confidence interval)
            avg_uncertainty = (df_future['upper_bound'] - df_future['lower_bound']).mean()
            st.markdown(f"""
                <div class='kpi-card'>
                    <div class='label'>Avg Uncertainty</div>
                    <div class='big-number'>±{avg_uncertainty/2:.0f} DZD</div>
                    <div style='font-size: 0.85rem; color: #808080;'>Width of prediction band</div>
                </div>
            """, unsafe_allow_html=True)
            
        with kpi_f3:
            # Historical volatility of this specific item
            hist_vol = comm_history_monthly['price'].std()
            st.markdown(f"""
                <div class='kpi-card'>
                    <div class='label'>Item Volatility</div>
                    <div class='big-number'>{hist_vol:.1f}</div>
                    <div style='font-size: 0.85rem; color: #808080;'>Std Dev (Historical)</div>
                </div>
            """, unsafe_allow_html=True)

        # 6. Plotting
        
        # FIX 1: Convert the ENTIRE date column to strings first
        # This prevents Plotly from trying to do math on timestamps
        comm_history_monthly['plot_date'] = comm_history_monthly['date'].astype(str)
        df_future['plot_date'] = df_future['date'].astype(str)
        
        # FIX 2: Get the "Today" date FROM the string column (not the timestamp column)
        last_date_str = comm_history_monthly['plot_date'].iloc[-1]

        st.markdown(f"### 📈 Forecast Chart: {selected_forecast_item}")
        fig_item_forecast = go.Figure()
        
        # Trace 1: Historical
        fig_item_forecast.add_trace(go.Scatter(
            x=comm_history_monthly['plot_date'], 
            y=comm_history_monthly['price'],
            mode='lines+markers',
            name='Historical',
            line=dict(color='#4CAF50', width=2),
            marker=dict(size=4)
        ))
        
        # Trace 2: Forecast
        fig_item_forecast.add_trace(go.Scatter(
            x=df_future['plot_date'], 
            y=df_future['predicted_price'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#2196F3', width=2, dash='dash'),
            marker=dict(size=4)
        ))
        
        # Trace 3: Confidence Interval
        fig_item_forecast.add_trace(go.Scatter(
            x=pd.concat([df_future['plot_date'], df_future['plot_date'][::-1]]),
            y=pd.concat([df_future['upper_bound'], df_future['lower_bound'][::-1]]),
            fill='toself',
            fillcolor='rgba(33, 150, 243, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% CI',
            showlegend=False
        ))
        
        # Trace 4: "Today" Line (Manual implementation)
        # Calculate Y-range to draw a vertical line across the chart
        all_y_values = list(comm_history_monthly['price']) + list(df_future['upper_bound']) + list(df_future['lower_bound'])
        min_y = min(all_y_values)
        max_y = max(all_y_values)
        
        fig_item_forecast.add_trace(go.Scatter(
            x=[last_date_str, last_date_str],
            y=[min_y, max_y],
            mode='lines',
            name='Today',
            line=dict(color='gray', width=1, dash='dot'),
            hoverinfo='skip'
        ))

        # Add annotation for "Today"
        fig_item_forecast.add_annotation(
            x=last_date_str,
            y=max_y,
            text="Today",
            showarrow=False,
            yshift=10,
            font=dict(color='gray')
        )

        fig_item_forecast.update_layout(
            template='plotly_dark',
            paper_bgcolor='#1a1a1a',
            plot_bgcolor='#262626',
            font=dict(color='#FAFAFA'),
            xaxis_title="Date",
            yaxis_title="Price (DZD)",
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(x=0.01, y=0.99),
            xaxis_type='date' # Tell plotly these strings are actually dates
        )
        st.plotly_chart(fig_item_forecast, use_container_width=True)

        # Display raw data table (Optional, collapsible)
        with st.expander(f"View Raw Forecast Data for {selected_forecast_item}"):
            st.dataframe(df_future)

    except Exception as e:
        st.error(f"Error loading forecast for {selected_forecast_item}: {e}")
else:
    st.info("⚠️ Forecast folder not found. Please run the forecasting script first.")
st.markdown("---")

# ============== Trend Anomaly Detection ==============
st.markdown("### ⚠️ Market Trend Anomalies")
st.markdown("<div style='color: #808080; margin-bottom: 16px;'>Identifying months where prices deviated significantly from the expected market trend</div>", unsafe_allow_html=True)

# Prepare Data: Aggregate by Month to see the Trend
# We use 'date' and 'price' as defined in the food script
df_trend = df_filtered.copy().set_index('date')

# Resample by Month Start (MS) to get average price per month
# This smooths out noise from individual markets
monthly_trend = df_trend['price'].resample('MS').mean().dropna()
df_trend_data = monthly_trend.reset_index()
df_trend_data.columns = ['date', 'avg_price']

# Check if we have enough data (at least 4 months) to calculate a trend
if len(df_trend_data) >= 4:
    # Calculate Rolling Statistics to define the "Normal" Trend
    window = 3 # Compare current month to average of previous 3 months
    df_trend_data['trend_line'] = df_trend_data['avg_price'].rolling(window=window, center=True).mean()
    df_trend_data['trend_std'] = df_trend_data['avg_price'].rolling(window=window, center=True).std()
    
    # Calculate Z-Score: How far is this month from the trend?
    df_trend_data['z_score'] = (df_trend_data['avg_price'] - df_trend_data['trend_line']) / df_trend_data['trend_std']
    
    # Identify Anomalies (Z-score > 1.5 indicates a significant deviation from trend)
    # We drop NaNs created by the rolling window
    df_anomalies = df_trend_data.dropna(subset=['z_score'])
    trend_anomalies = df_anomalies[df_anomalies['z_score'].abs() > 1.5].sort_values('date', ascending=False)

    # ============== Visualization ==============
    fig_trend_anomaly = go.Figure()

    # 1. Plot the Trend Line (Moving Average)
    fig_trend_anomaly.add_trace(go.Scatter(
        x=df_trend_data['date'],
        y=df_trend_data['trend_line'],
        mode='lines',
        name='Expected Trend (3-Month Avg)',
        line=dict(color='#808080', width=2, dash='dot'),
    ))

    # 2. Plot Actual Monthly Prices (Color coded by anomaly status)
    normal_mask = df_trend_data['z_score'].abs() <= 1.5
    
    # Normal Points
    fig_trend_anomaly.add_trace(go.Scatter(
        x=df_trend_data[normal_mask]['date'],
        y=df_trend_data[normal_mask]['avg_price'],
        mode='markers+lines',
        name='Normal Prices',
        line=dict(color='#4CAF50', width=1),
        marker=dict(size=8, color='#4CAF50')
    ))
    
    # Anomaly Points
    if not trend_anomalies.empty:
        fig_trend_anomaly.add_trace(go.Scatter(
            x=trend_anomalies['date'],
            y=trend_anomalies['avg_price'],
            mode='markers',
            name='Anomaly Detected',
            marker=dict(
                size=12, 
                color=trend_anomalies['z_score'].apply(lambda x: '#EF5350' if x > 0 else '#66BB6A'), # Red for high, Green for low
                line=dict(color='white', width=2)
            ),
            text=trend_anomalies['date'].dt.strftime('%b %Y'),
            hovertemplate='<b>%{text}</b><br>Price: %{y:,.0f} DZD<extra>Anomaly</extra>'
        ))

    fig_trend_anomaly.update_layout(
        template='plotly_dark',
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#262626',
        font=dict(color='#FAFAFA'),
        xaxis_title="Date",
        yaxis_title="Average Price (DZD)",
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_trend_anomaly, use_container_width=True)

    # ============== Text Details ==============
    if not trend_anomalies.empty:
        st.markdown("#### 🔎 Detected Market Shifts")
        for _, row in trend_anomalies.head(3).iterrows():
            # Calculate deviation from trend
            deviation = ((row['avg_price'] - row['trend_line']) / row['trend_line']) * 100
            
            # Styling
            color = "badge-danger" if row['z_score'] > 0 else "badge-success" # Red = Price Spike
            icon = "📈" if row['z_score'] > 0 else "📉"
            period_str = row['date'].strftime('%B %Y')
            
            st.markdown(f"""
                <div class='insight-card' style='border-left-color: #FF9800;'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div>
                            <div style='font-size: 1.1rem; margin-bottom: 4px;'>{icon} <strong>{period_str}</strong></div>
                            <div style='font-size: 0.85rem; color: #808080;'>
                                Avg Price: {row['avg_price']:,.0f} DZD vs Trend: {row['trend_line']:,.0f} DZD
                            </div>
                        </div>
                        <span class='badge {color}' style='font-size: 1rem;'>{deviation:+.1f}%</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("✅ No significant trend anomalies detected. Prices are following the expected seasonal progression.")

else:
    st.info("ℹ️ Not enough historical data available (less than 4 months) to calculate trend anomalies.")

st.markdown("---")
# ============== AI Insights ==============
st.markdown("### 🤖 AI-Powered Insights")

# Prepare data for AI (Only pure data, no formatting)
ai_context = {
"avg_price": avg_price,
"trend_30d": price_change_str,
"volatility": vol_str,
"forecast_next_price": pred_price,
"commodities": selected_commodities if selected_commodities else "All",
"regions": selected_regions if selected_regions else "All"
}

if st.button("🚀 Generate AI Insights", type="primary", use_container_width=True):
    # This section relies on an external API.
    # In a production environment, ensure API keys are stored in st.secrets, not hardcoded.
    api_key = "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"

    prompt = f"""
    Analyze this Algerian food market data:
    - Commodities: {ai_context['commodities']}
    - Current Avg Price: {ai_context['avg_price']:.2f} DZD
    - 30-Day Trend: {ai_context['trend_30d']}
    - Market Volatility: {ai_context['volatility']}
    - 30-Day Forecast: {ai_context['forecast_next_price']:.0f} DZD

    Provide 3 concise, bulleted strategic insights for a wholesaler.
    Focus on timing (buy/hold) and risk.
    """

    try:
        with st.spinner("Analyzing market patterns..."):
            response = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
            json={
            'model': 'llama-3.1-8b-instant',
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.7,
            'max_tokens': 300
            }
            )

        if response.status_code == 200:
            insight_text = response.json()['choices'][0]['message']['content']
            st.markdown(f"<div class='insight-card'>{insight_text}</div>", unsafe_allow_html=True)
        else:
            st.error("AI Service currently unavailable.")
    except Exception as e:
        st.error(f"Connection error: {e}")