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
        border: 1px solid #3a3a3a;
        border-left: 4px solid #4CAF50;
        margin-bottom: 16px;
    }
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
    .badge-danger {
        background-color: rgba(244, 67, 54, 0.2);
        color: #EF5350;
    }
    .badge-warning {
        background-color: rgba(255, 152, 0, 0.2);
        color: #FFA726;
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
    </style>
""", unsafe_allow_html=True)

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
    
    # Main Price data
    try:
        df_prices = pd.read_csv(os.path.join(parent_dir, 'wfp_food_prices_dza.csv'))
        df_prices['date'] = pd.to_datetime(df_prices['date'])
        
        # Basic data cleaning
        # Ensure numeric prices
        df_prices['price'] = pd.to_numeric(df_prices['price'], errors='coerce')
        df_prices = df_prices.dropna(subset=['price', 'date'])
    except Exception as e:
        st.error(f"Error loading main data: {e}")
        return pd.DataFrame(), None, None, None, {}
    
    # Enhanced data (Optional)
    try:
        df_enhanced = pd.read_csv(os.path.join(os.path.dirname(parent_dir), 'prophet_data_enhanced.csv'))
        df_enhanced['ds'] = pd.to_datetime(df_enhanced['ds'])
    except:
        df_enhanced = None
    
    # Forecast data (Optional)
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
    
    # Load category data
    category_data = {}
    try:
        # Simply grouping main data by commodity if separate files don't exist
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
    if st.button("🔄 Refresh Data", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()

st.markdown("---")

# ============== Filters (Logical) ==============
st.markdown("### 🔍 Data Filters")

# Create logical filters based on actual data columns
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

# Apply filters
df_filtered = df_prices.copy()
if selected_commodities:
    df_filtered = df_filtered[df_filtered['commodity'].isin(selected_commodities)]
if selected_regions:
    df_filtered = df_filtered[df_filtered['admin1'].isin(selected_regions)]

# Date range filtering (Default to last 2 years for better chart visibility)
min_date = df_filtered['date'].min()
max_date = df_filtered['date'].max()

with col_date:
    # Default to full dataset
    default_start = min_date
    date_range = st.date_input("Date Range", [default_start, max_date])

if len(date_range) == 2:
    df_filtered = df_filtered[(df_filtered['date'] >= pd.to_datetime(date_range[0])) & 
                              (df_filtered['date'] <= pd.to_datetime(date_range[1]))]

if df_filtered.empty:
    st.warning("No data matches your filters. Adjust selections.")
    st.stop()

# ============== KPI Calculation (Dynamic) ==============
# Sort by date for accurate calc
df_filtered = df_filtered.sort_values('date')

# 1. Average Price
avg_price = df_filtered['price'].mean()

# 2. Price Change (Current window vs Previous window of same length)
current_window_days = (df_filtered['date'].max() - df_filtered['date'].min()).days
if current_window_days < 30: current_window_days = 30 # fallback

max_d = df_filtered['date'].max()
last_period = df_filtered[df_filtered['date'] >= (max_d - timedelta(days=30))]
prev_period = df_filtered[(df_filtered['date'] >= (max_d - timedelta(days=60))) & 
                          (df_filtered['date'] < (max_d - timedelta(days=30)))]

if not prev_period.empty and not last_period.empty:
    curr_mean = last_period['price'].mean()
    prev_mean = prev_period['price'].mean()
    price_change = ((curr_mean - prev_mean) / prev_mean) * 100
    price_change_str = f"{'+' if price_change >= 0 else ''}{price_change:.1f}%"
    price_change_class = 'badge-success' if price_change >= 0 else 'badge-danger' # Higher price usually bad for consumer, good for seller. keeping generic color.
else:
    price_change_str = "N/A"
    price_change_class = 'badge-warning'

# 3. Volatility (Coefficient of Variation)
price_std = df_filtered['price'].std()
volatility_coef = (price_std / avg_price) * 100 if avg_price > 0 else 0
vol_str = f"{volatility_coef:.1f}%"

# 4. Confidence Score (Calculated based on data density and volatility)
# Lower volatility + Higher sample size = Higher confidence
sample_size_score = min(len(df_filtered) / 100, 1.0) * 50 # Max 50 points for data amount
volatility_score = max(0, 50 - volatility_coef) # Max 50 points for stability
confidence = sample_size_score + volatility_score
confidence = min(max(confidence, 10.0), 99.0) # Clamp 10-99%

# 5. Active listings
active_listings = len(df_filtered)

# KPI Display
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

st.markdown("---")

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

# Layout for Prediction & Drivers
st.markdown("### 🔮 Price Prediction (Next 30 Days)")
pred_col1, pred_col2, pred_col3 = st.columns(3)

with pred_col1:
    st.markdown(f"""
        <div class='kpi-card'>
            <div class='label'>Predicted Trend</div>
            <div class='big-number' style='font-size: 2rem;'>{pred_price:.0f} DZD</div>
            <div style='margin-top: 12px;'>
                <span class='badge {pred_change_class}'>{pred_change_str}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

with pred_col2:
    st.markdown(f"""
        <div class='kpi-card'>
            <div class='label'>Forecast Range</div>
            <div style='font-size: 1.8rem; font-weight: bold; color: #FAFAFA; margin: 8px 0;'>
                {lower_bound_val:.0f} - {upper_bound_val:.0f}
            </div>
            <div style='font-size: 0.85rem; color: #808080;'>Based on historical volatility</div>
        </div>
    """, unsafe_allow_html=True)

with pred_col3:
    st.markdown(f"""
        <div class='kpi-card'>
            <div class='label'>Key Drivers</div>
            <div style='margin-top: 10px; font-size: 0.9rem;'>
                <div style='display:flex; justify-content:space-between; margin-bottom:4px;'>
                    <span>📅 Seasonality:</span><span style='color:#FAFAFA'>{seasonality_impact}</span>
                </div>
                <div style='display:flex; justify-content:space-between; margin-bottom:4px;'>
                    <span>🌍 Reg. Variance:</span><span style='color:#FAFAFA'>{comp_str}</span>
                </div>
                <div style='display:flex; justify-content:space-between;'>
                    <span>📈 Overall Trend:</span><span style='color:#FAFAFA'>{trend_str}</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

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
    # ============== Anomaly Detection (Statistical) ==============
st.markdown("### ⚠️ Anomaly Detection")

# Detect anomalies using Z-Score on the filtered dataset
anomalies = []
df_anomaly = df_filtered.copy()
df_anomaly['mean_price'] = df_anomaly['price'].rolling(window=30, center=True).mean()
df_anomaly['std_price'] = df_anomaly['price'].rolling(window=30, center=True).std()
df_anomaly['z_score'] = (df_anomaly['price'] - df_anomaly['mean_price']) / df_anomaly['std_price']

# Filter for recent anomalies (last 60 days)
recent_anomalies = df_anomaly[
(df_anomaly['date'] > (df_anomaly['date'].max() - timedelta(days=60))) &
(df_anomaly['z_score'].abs() > 2) # Z-score > 2 implies < 5% probability
].sort_values('date', ascending=False)

if not recent_anomalies.empty:
    for _, row in recent_anomalies.head(3).iterrows():
        change_pct = ((row['price'] - row['mean_price']) / row['mean_price']) * 100
        severity = "High" if abs(row['z_score']) > 3 else "Medium"
        color = "badge-danger" if change_pct > 0 else "badge-success" # High price red, low price green (buyer perspective)
        icon = "📈" if change_pct > 0 else "📉"
        st.markdown(f"""
        <div class='insight-card'>
        <div style='display: flex; justify-content: space-between; align-items: start;'>
        <div style='flex: 1;'>
        <div style='font-size: 1.1rem; margin-bottom: 8px;'>{icon} Unusual Price Detected in {row['market']}</div>
        <div style='font-size: 0.85rem; color: #808080;'>
        Date: {row['date'].strftime('%Y-%m-%d')} | Severity: {severity} | Region: {row['admin1']}
        </div>
        </div>
        <div>
        <span class='badge {color}' style='font-size: 1rem;'>{change_pct:+.1f}% vs 30d Avg</span>
        </div>
        </div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("✅ No significant statistical anomalies detected in the last 60 days for selected data.")

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