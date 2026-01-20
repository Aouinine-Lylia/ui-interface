import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import requests
import joblib
import glob

# Page configuration
st.set_page_config(
    page_title="Laptops | DZA PriceSight",
    page_icon="💻",
    layout="wide"
)
# ============== Load & Preprocess Laptop Data ==============
@st.cache_data
def load_laptops():
    try:
        # 1. Find the CSV file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Try looking in script dir first
        csv_path = os.path.join(script_dir, 'kaggle_laptop_data.csv')
        
        # If not there, try the 'ui-interface' parent folder
        if not os.path.exists(csv_path):
             csv_path = '/home/lylia/Documents/SIC-samsung/training-food/ui-interface/kaggle_laptop_data.csv'

        df = pd.read_csv(csv_path)
        
        # 2. Parse Date (Handling messy format "2021 10 01T18...")
        def parse_date(date_str):
            try:
                clean_str = str(date_str).replace('T', ' ').replace('Z', '').strip()
                return pd.to_datetime(clean_str, format='%Y %m %d %H:%M:%S.%f')
            except:
                return pd.NaT
        
        # THE FIX: Overwrite 'created_at' column directly with datetime objects.
        # This fixes the TypeError in the KPI section.
        df['created_at'] = df['created_at'].apply(parse_date)
        
        # Also ensure 'date' column exists (needed for Forecast section later)
        df['date'] = df['created_at']
        
        # Drop rows where parsing failed
        df = df.dropna(subset=['date', 'price_preview'])
        
        # 3. Extract CPU Tier (Must match logic used in forecasting script)
        def get_cpu_tier(cpu_str):
            if pd.isna(cpu_str): return "Unknown"
            cpu_str = str(cpu_str).upper()
            if "I9" in cpu_str or "R9" in cpu_str: return "Ultra High-End"
            if "I7" in cpu_str or "R7" in cpu_str or "THREADRIPPER" in cpu_str: return "High-End"
            if "I5" in cpu_str or "R5" in cpu_str: return "Mid-Range"
            if "I3" in cpu_str or "R3" in cpu_str or "CELERON" in cpu_str: return "Entry-Level"
            return "Other"
            
        df['cpu_tier'] = df['CPU'].apply(get_cpu_tier)
        
        return df
    except Exception as e:
        st.error(f"Error loading laptops data: {e}")
        return pd.DataFrame()

df_laptops = load_laptops()
@st.cache_resource
def load_my_model():
    # Replace with your actual path
    return joblib.load('pages/laptop_price_regression_model.pkl')

model = load_my_model()


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
        border-left: 4px solid #2196F3;
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
    .badge-info {
        background-color: rgba(33, 150, 243, 0.2);
        color: #42A5F5;
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

# Sidebar Navigation
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


# Initialize session state
if 'gpu_filter' not in st.session_state:
    st.session_state.gpu_filter = False
if 'gaming_filter' not in st.session_state:
    st.session_state.gaming_filter = False
if 'new_filter' not in st.session_state:
    st.session_state.new_filter = False
if 'brand_filter' not in st.session_state:
    st.session_state.brand_filter = 'All'

# Header Section
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("# 💻 Laptops")
    st.markdown("### Comprehensive market analytics and price intelligence for laptops")


with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔮 Predict Laptop Price", use_container_width=True, type="primary"):
        st.session_state.show_predict_form = True

if st.session_state.get('show_predict_form'):
    with st.expander("💻 Laptop Specification Input", expanded=True):
        with st.form("prediction_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                ram = st.number_input("RAM (GB)", value=16)
                storage = st.number_input("Storage (GB)", value=512)
                cpu_gen = st.number_input("CPU Generation", value=12)
                screen = st.number_input("Screen Size", value=15.6)
            
            with col_b:
                cpu_type = st.selectbox("CPU Type", ["Core i7", "Core i3", "Core i5", "Core i9", "Ryzen", "Other"])
                condition = st.selectbox("Condition", ["JAMAIS UTILIS", "MOYEN", "Unknown"])
                has_gpu = st.checkbox("Has Dedicated GPU", value=True)

            if st.form_submit_button("Calculate Estimated Price"):
                # 1. Prepare the JSON-like feature vector
                features = {
                    "RAM_GB": ram, "Total_Storage": storage, "CPU_Gen": cpu_gen,
                    "Has_GPU": 1 if has_gpu else 0, "SCREEN_SIZE": screen,
                    "CPU_Type_Core i3": 1 if cpu_type == "Core i3" else 0,
                    "CPU_Type_Core i5": 1 if cpu_type == "Core i5" else 0,
                    "CPU_Type_Core i7": 1 if cpu_type == "Core i7" else 0,
                    "CPU_Type_Core i9": 1 if cpu_type == "Core i9" else 0,
                    "CPU_Type_Other": 1 if cpu_type == "Other" else 0,
                    "CPU_Type_Ryzen": 1 if cpu_type == "Ryzen" else 0,
                    "Condition_JAMAIS UTILIS": 1 if condition == "JAMAIS UTILIS" else 0,
                    "Condition_MOYEN": 1 if condition == "MOYEN" else 0,
                    "Condition_Unknown": 1 if condition == "Unknown" else 0
                }
                
                # 2. Convert to DataFrame for model
                input_df = pd.DataFrame([features])
                
                # 3. Predict
                prediction = model.predict(input_df)[0]
                prediction=np.exp(prediction)
                st.success(f"### Estimated Market Price: {prediction:,.2f} DZD")
st.markdown("---")

# Filters Section
st.markdown("### 🔍 Smart Filters")

filter_col1, filter_col2, filter_col3, filter_col4, filter_col5 = st.columns(5)

with filter_col1:
    brands = ['All'] + sorted([str(x) for x in df_laptops['model_name'].unique().tolist()])
    st.session_state.brand_filter = st.selectbox("Brand", brands)

with filter_col2:
    st.session_state.gpu_filter = st.checkbox("Dedicated GPU", value=st.session_state.gpu_filter)

with filter_col3:
    st.session_state.gaming_filter = st.checkbox("Gaming (RTX)", value=st.session_state.gaming_filter)

with filter_col4:
    st.session_state.new_filter = st.checkbox("New Only", value=st.session_state.new_filter)

with filter_col5:
    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
    if st.button("💾 Save Filter"):
        st.success("Filters saved!")

st.markdown("---")

# Apply filters
df_filtered = df_laptops.copy()

if st.session_state.brand_filter != 'All':
    df_filtered = df_filtered[df_filtered['model_name'] == st.session_state.brand_filter]

if st.session_state.gpu_filter:
    df_filtered = df_filtered[df_filtered['DEDICATED_GPU'] != 'INTEGRATED']

if st.session_state.gaming_filter:
    df_filtered = df_filtered[df_filtered['DEDICATED_GPU'].str.contains('RTX', na=False)]

if st.session_state.new_filter:
    df_filtered = df_filtered[df_filtered['spec_Etat'].isin(['JAMAIS UTILISÉ', 'ÉTAT NEUF'])]

# KPI Cards
st.markdown("### 📊 Market Overview")

# Calculate KPIs
avg_price = df_filtered['price_preview'].mean()
median_price = df_filtered['price_preview'].median()
total_listings = len(df_filtered)
unique_models = df_filtered['model_name'].nunique()

# Price change calculation
max_date = df_filtered['created_at'].max()
last_30 = df_filtered[df_filtered['created_at'] >= (max_date - timedelta(days=30))]
prev_30 = df_filtered[(df_filtered['created_at'] >= (max_date - timedelta(days=60))) & 
                      (df_filtered['created_at'] < (max_date - timedelta(days=30)))]

if len(prev_30) > 0 and len(last_30) > 0:
    price_change = ((last_30['price_preview'].mean() - prev_30['price_preview'].mean()) / prev_30['price_preview'].mean()) * 100
    price_change_str = f"{'+' if price_change >= 0 else ''}{price_change:.1f}%"
    price_change_class = 'badge-danger' if price_change > 0 else 'badge-success'
else:
    price_change_str = "N/A"
    price_change_class = 'badge-warning'

# GPU adoption rate
gpu_rate = (len(df_filtered[df_filtered['DEDICATED_GPU'] != 'INTEGRATED']) / len(df_filtered) * 100) if len(df_filtered) > 0 else 0

# Average specs
avg_ram = df_filtered['RAM_SIZE'].value_counts().index[0] if len(df_filtered) > 0 else 'N/A'
most_popular_cpu = df_filtered['CPU'].value_counts().index[0] if len(df_filtered) > 0 else 'N/A'

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(f"""
        <div class='kpi-card'>
            <div style='display: flex; justify-content: space-between; align-items: flex-start;'>
                <div style='font-size: 2rem;'>💰</div>
                <span class='badge {price_change_class}'>{price_change_str}</span>
            </div>
            <div class='label' style='margin-top: 16px;'>Average Price</div>
            <div class='big-number'>{avg_price/1000:.0f}K DZD</div>
            <div style='font-size: 0.85rem; color: #808080; margin-top: 4px;'>Median: {median_price/1000:.0f}K DZD</div>
        </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
        <div class='kpi-card'>
            <div style='font-size: 2rem; margin-bottom: 16px;'>🎮</div>
            <div class='label'>GPU Adoption</div>
            <div class='big-number'>{gpu_rate:.1f}%</div>
            <div style='font-size: 0.85rem; color: #808080; margin-top: 4px;'>Dedicated graphics</div>
        </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
        <div class='kpi-card'>
            <div style='font-size: 2rem; margin-bottom: 16px;'>📦</div>
            <div class='label'>Active Listings</div>
            <div class='big-number'>{total_listings:,}</div>
            <div style='font-size: 0.85rem; color: #808080; margin-top: 4px;'>{unique_models} unique models</div>
        </div>
    """, unsafe_allow_html=True)

with kpi4:
    popular_brand = df_filtered['model_name'].value_counts().index[0] if len(df_filtered) > 0 else 'N/A'
    brand_count = df_filtered['model_name'].value_counts().iloc[0] if len(df_filtered) > 0 else 0
    brand_pct = (brand_count / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
    
    st.markdown(f"""
        <div class='kpi-card'>
            <div style='font-size: 2rem; margin-bottom: 16px;'>🏆</div>
            <div class='label'>Top Brand</div>
            <div style='font-size: 1.8rem; font-weight: bold; color: #FAFAFA; margin: 8px 0;'>{popular_brand}</div>
            <div style='font-size: 0.85rem; color: #808080; margin-top: 4px;'>{brand_pct:.1f}% market share</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Spec Distribution
st.markdown("### 📊 Specification Analysis")

spec_col1, spec_col2, spec_col3 = st.columns(3)

with spec_col1:
    cpu_dist = df_filtered['CPU'].value_counts().head(5)
    st.markdown("<div class='label'>Top CPUs</div>", unsafe_allow_html=True)
    for cpu, count in cpu_dist.items():
        pct = (count / len(df_filtered) * 100)
        st.markdown(f"""
            <div style='padding: 8px 0; border-bottom: 1px solid #3a3a3a;'>
                <div style='display: flex; justify-content: space-between;'>
                    <span>{cpu}</span>
                    <span class='badge badge-info'>{pct:.1f}%</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

with spec_col2:
    ram_dist = df_filtered['RAM_SIZE'].value_counts().head(5)
    st.markdown("<div class='label'>RAM Distribution</div>", unsafe_allow_html=True)
    for ram, count in ram_dist.items():
        pct = (count / len(df_filtered) * 100)
        st.markdown(f"""
            <div style='padding: 8px 0; border-bottom: 1px solid #3a3a3a;'>
                <div style='display: flex; justify-content: space-between;'>
                    <span>{ram}</span>
                    <span class='badge badge-info'>{pct:.1f}%</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

with spec_col3:
    gpu_dist = df_filtered['DEDICATED_GPU'].value_counts().head(5)
    st.markdown("<div class='label'>Top GPUs</div>", unsafe_allow_html=True)
    for gpu, count in gpu_dist.items():
        pct = (count / len(df_filtered) * 100)
        st.markdown(f"""
            <div style='padding: 8px 0; border-bottom: 1px solid #3a3a3a;'>
                <div style='display: flex; justify-content: space-between;'>
                    <span>{gpu[:20]}</span>
                    <span class='badge badge-info'>{pct:.1f}%</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# Charts
st.markdown("### 📈 Market Trends")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("#### Price Trends Over Time")
    
    # Monthly price trend
    df_trend = df_filtered.copy()
    df_trend['month'] = df_trend['created_at'].dt.to_period('M')
    monthly_avg = df_trend.groupby('month')['price_preview'].mean().reset_index()
    monthly_avg['month'] = monthly_avg['month'].dt.to_timestamp()
    
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=monthly_avg['month'],
        y=monthly_avg['price_preview'],
        mode='lines+markers',
        name='Avg Price',
        line=dict(color='#2196F3', width=3),
        marker=dict(size=8)
    ))
    
    fig_trend.update_layout(
        template='plotly_dark',
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#262626',
        font=dict(color='#FAFAFA'),
        xaxis_title="Month",
        yaxis_title="Price (DZD)",
        height=400,
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with chart_col2:
    st.markdown("#### Price Distribution by Brand")
    
    brand_prices = df_filtered.groupby('model_name')['price_preview'].mean().sort_values(ascending=False).head(8)
    
    fig_brand = go.Figure()
    fig_brand.add_trace(go.Bar(
        x=brand_prices.index,
        y=brand_prices.values,
        marker=dict(
            color=brand_prices.values,
            colorscale='Viridis',
            showscale=False
        )
    ))
    
    fig_brand.update_layout(
        template='plotly_dark',
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#262626',
        font=dict(color='#FAFAFA'),
        xaxis_title="Brand",
        yaxis_title="Average Price (DZD)",
        height=400,
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig_brand, use_container_width=True)

st.markdown("---")




# ============== FIXED Laptop Forecast Logic (No History + Translations) ==============

# Initialize forecast variables with defaults
next_pred = avg_price  # Default to current average
avg_uncertainty = 0

# 1. Define path to forecasting folder
FORECAST_DIR = os.path.join(os.path.dirname(__file__), "forecasting_laptops")

# Check if folder exists
if os.path.exists(FORECAST_DIR) and not df_laptops.empty:
    # 2. Find all CSV files and parse names correctly
    forecast_files = glob.glob(os.path.join(FORECAST_DIR, '*.csv'))
    
    # Helper function to translate French/Algerian conditions to English
    def get_readable_condition(condition_key):
        # Normalize key to uppercase for consistent matching
        key_norm = condition_key.upper().replace('_', ' ')
        
        # Mapping dictionary for common terms
        mapping = {
            "BON TAT": "Good State",
            "JAMAIS UTILIS": "Never Used",
            "MOYEN": "Fair",
            "NEUF": "New",
            "TRES BON": "Very Good",
            "ETAT NEUF": "Brand New"
        }
        
        # Return mapped value or default (Title Case)
        return mapping.get(key_norm, condition_key.replace('_', ' ').title())

    # Dictionary to map display name -> actual filename
    forecast_options = {}
    
    for file in forecast_files:
        filename = os.path.basename(file)
        # Remove the suffix '_forecast.csv' to get the core identifier
        # e.g., 'BON_TAT_Entry-Level_forecast.csv' -> 'BON_TAT_Entry-Level'
        name_part = filename.replace('_forecast.csv', '')
        
        # Logic: The filename format is Condition_Tier.
        # Since Condition might contain underscores (e.g., BON_TAT), we split from the RIGHT.
        parts = name_part.rsplit('_', 1)
        
        if len(parts) == 2:
            condition_raw = parts[0] # e.g., 'BON_TAT'
            tier = parts[1]           # e.g., 'Entry-Level'
            
            # Translate the condition to meaningful English
            condition_readable = get_readable_condition(condition_raw)
            
            # Create a readable display name for the UI
            display_name = f"{condition_readable} | {tier}"
            forecast_options[display_name] = filename
        else:
            # Fallback for unexpected formats
            display_name = name_part.replace('_', ' ').title()
            forecast_options[display_name] = filename

    # Sort options alphabetically
    sorted_display_names = sorted(forecast_options.keys())

    # 3. UI: Select Box to choose item
    st.markdown("### 🔮 Select Laptop Group to Forecast")
    st.markdown("<div style='color: #808080; margin-bottom: 16px;'>Select a condition & performance tier</div>", unsafe_allow_html=True)
    
    selected_display_name = st.selectbox(
        "Available Forecasts",
        sorted_display_names,
        label_visibility="collapsed"
    )

    # 4. Disclaimer Note
    st.info("⚠️ **Note:** This forecast predicts market **trends** based on historical data. It estimates future price movements but does not guarantee an exact sale price for a specific item.")
    
    # 5. Retrieve the actual filename
    selected_filename = forecast_options[selected_display_name]
    
    # 6. Load the specific forecast data
    forecast_path = os.path.join(FORECAST_DIR, selected_filename)
    
    try:
        # 1. Read CSV. We do NOT use index_col=0 here so we can see all headers.
        df_future = pd.read_csv(forecast_path)
        
        # 2. Robust Column Renaming Logic
        rename_map = {}
        
        # Standardize column names to lowercase for easier matching (optional, but safer)
        df_future.columns = [str(c).lower() if isinstance(c, str) else c for c in df_future.columns]
        
        # Check and rename columns
        if 'ds' in df_future.columns:
            rename_map['ds'] = 'date'
        
        if 'yhat' in df_future.columns:
            rename_map['yhat'] = 'predicted_price'
            
        if 'yhat_lower' in df_future.columns:
            rename_map['yhat_lower'] = 'lower_bound'
            
        if 'yhat_upper' in df_future.columns:
            rename_map['yhat_upper'] = 'upper_bound'
            
        # Apply the renaming
        df_future = df_future.rename(columns=rename_map)
        
        # 3. Validate that we successfully found the expected columns
        required_columns = ['date', 'predicted_price', 'lower_bound', 'upper_bound']
        missing_cols = [col for col in required_columns if col not in df_future.columns]
        
        if missing_cols:
            st.error(f"CSV format error. The file was loaded but is missing expected columns: {missing_cols}")
            st.write("Detected columns in file:", df_future.columns.tolist())
            st.write("First 5 rows:", df_future.head())
        else:
            # 4. Convert Date column to datetime
            df_future['date'] = pd.to_datetime(df_future['date'])
            
            # =================================================
            # FORECAST ONLY CALCULATIONS (No Historical Data)
            # =================================================
            
            # 1. Get Prediction Values from the CSV
            next_pred = df_future['predicted_price'].iloc[0]
            avg_uncertainty = (df_future['upper_bound'] - df_future['lower_bound']).mean()
            
            # 2. Forecast Trend Calculation (First month vs Last month of forecast)
            # This replaces the "Change vs History" calculation to show direction
            first_pred = df_future['predicted_price'].iloc[0]
            last_pred = df_future['predicted_price'].iloc[-1]
            forecast_trend = ((last_pred - first_pred) / first_pred) * 100 if first_pred > 0 else 0

            # Display KPIs
            kpi_f1, kpi_f2, kpi_f3 = st.columns(3)
            
            trend_class = 'badge-success' if forecast_trend >= 0 else 'badge-danger'
            trend_str = f"{'+' if forecast_trend >= 0 else ''}{forecast_trend:.1f}%"
            
            with kpi_f1:
                st.markdown(f"""
                    <div class='kpi-card'>
                        <div class='label'>Next Month Prediction</div>
                        <div class='big-number' style='color: #2196F3;'>{next_pred:,.0f} DZD</div>
                        <div style='margin-top: 12px;'>
                            <span class='badge {trend_class}'>{trend_str}</span>
                        </div>
                        <div style='font-size: 0.8rem; color: #808080; margin-top: 5px;'>Forecast trend</div>
                    </div>
                """, unsafe_allow_html=True)

            with kpi_f2:
                st.markdown(f"""
                    <div class='kpi-card'>
                        <div class='label'>Avg Uncertainty</div>
                        <div class='big-number'>±{avg_uncertainty/2:.0f} DZD</div>
                        <div style='font-size: 0.85rem; color: #808080;'>Width of prediction band</div>
                    </div>
                """, unsafe_allow_html=True)
                
            with kpi_f3:
                forecast_horizon = len(df_future)
                st.markdown(f"""
                    <div class='kpi-card'>
                        <div class='label'>Forecast Duration</div>
                        <div class='big-number'>{forecast_horizon} Mo</div>
                        <div style='font-size: 0.85rem; color: #808080;'>Months projected</div>
                    </div>
                """, unsafe_allow_html=True)

            # 7. Plotting (Forecast Only)
            st.markdown(f"### 📈 Forecast Chart: {selected_display_name}")
            fig_item_forecast = go.Figure()
            
            # Trace 1: Forecast (Main Line)
            fig_item_forecast.add_trace(go.Scatter(
                x=df_future['date'],
                y=df_future['predicted_price'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#2196F3', width=2, dash='dash'),
                marker=dict(size=4)
            ))
            
            # Trace 2: Confidence Interval (Filled Area)
            fig_item_forecast.add_trace(go.Scatter(
                x=pd.concat([df_future['date'], df_future['date'][::-1]]),
                y=pd.concat([df_future['upper_bound'], df_future['lower_bound'][::-1]]),
                fill='toself',
                fillcolor='rgba(33, 150, 243, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% CI',
                showlegend=True
            ))
            
            # Trace 3: "Today" Line (Start of Forecast)
            # Calculate Y range based on forecast data only
            y_vals = list(df_future['upper_bound']) + list(df_future['lower_bound'])
            
            if y_vals:
                min_y = min(y_vals)
                max_y = max(y_vals)
                start_date = df_future['date'].iloc[0]
                
                fig_item_forecast.add_trace(go.Scatter(
                    x=[start_date, start_date],
                    y=[min_y, max_y],
                    mode='lines',
                    name='Forecast Start',
                    line=dict(color='gray', width=1, dash='dot'),
                    hoverinfo='skip'
                ))
                
                fig_item_forecast.add_annotation(
                    x=start_date,
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
                xaxis=dict(type='date')
            )
            st.plotly_chart(fig_item_forecast, use_container_width=True)

            with st.expander(f"View Raw Forecast Data for {selected_display_name}"):
                st.dataframe(df_future)

    except Exception as e:
        import traceback
        st.error(f"Error loading forecast for {selected_display_name}: {e}")
        st.error(traceback.format_exc()) # Show detailed error in console if needed


else:
    st.info("⚠️ Forecast folder not found or no data loaded. Please run the forecasting script first.")

# NOW prepare AI data AFTER forecast variables are defined
kpi_data_ai = {
    'avg_price': avg_price,
    'price_change': price_change_str,
    'total_listings': total_listings,
    'unique_models': unique_models,
    'gpu_rate': gpu_rate,
    'top_brand': popular_brand,
    'brand_share': brand_pct,
    'predicted_price': next_pred  # NOW it's defined
}

market_data_ai = {
    'top_cpu': most_popular_cpu,
    'top_ram': avg_ram
}
# Regional Analysis
st.markdown("### 🗺️ Regional Market Analysis")

region_col1, region_col2 = st.columns(2)

with region_col1:
    city_avg = df_filtered.groupby('city')['price_preview'].mean().sort_values(ascending=False).head(8)
    
    fig_city = go.Figure()
    fig_city.add_trace(go.Bar(
        y=city_avg.index,
        x=city_avg.values,
        orientation='h',
        marker=dict(
            color=city_avg.values,
            colorscale='Blues',
            showscale=False
        )
    ))
    
    fig_city.update_layout(
        template='plotly_dark',
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#262626',
        font=dict(color='#FAFAFA'),
        xaxis_title="Average Price (DZD)",
        yaxis_title="City",
        height=400,
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig_city, use_container_width=True)

with region_col2:
    city_counts = df_filtered['city'].value_counts().head(8)
    
    fig_count = go.Figure()
    fig_count.add_trace(go.Pie(
        labels=city_counts.index,
        values=city_counts.values,
        hole=0.4,
        marker=dict(colors=px.colors.sequential.Blues)
    ))
    
    fig_count.update_layout(
        template='plotly_dark',
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#262626',
        font=dict(color='#FAFAFA'),
        height=400,
        showlegend=True,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig_count, use_container_width=True)

st.markdown("---")

# AI Insights Section
st.markdown("### 🤖 AI-Powered Market Insights")
st.markdown("<div style='color: #808080; margin-bottom: 16px;'>Grok AI analyzing laptop market data</div>", unsafe_allow_html=True)

def get_grok_insights_laptops(kpi_data, market_data):
    """Call Groq API for laptop market insights"""
    api_key = "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"
    
    if not api_key:
        return {'error': True, 'message': 'API key not configured.'}
    
    prompt = f"""You are an AI analyst for laptop price intelligence in Algeria. Analyze the laptop market and provide 5-7 key insights in bullet points.

Laptop Market Data:
- Average Price: {kpi_data['avg_price']/1000:.0f}K DZD ({kpi_data['price_change']})
- Total Listings: {kpi_data['total_listings']:,}
- Unique Models: {kpi_data['unique_models']}
- GPU Adoption: {kpi_data['gpu_rate']:.1f}%
- Top Brand: {kpi_data['top_brand']} ({kpi_data['brand_share']:.1f}% market share)
- Most Common CPU: {market_data['top_cpu']}
- Most Common RAM: {market_data['top_ram']}
- Price Forecast: {kpi_data['predicted_price']/1000:.0f}K DZD

Provide insights in this format:
• **[Topic]**: [Clear, actionable insight with specific numbers]

Focus on: price trends, spec preferences, regional patterns, and buying recommendations."""

    try:
        response = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'llama-3.1-8b-instant',
                'messages': [
                    {'role': 'system', 'content': 'You are a laptop market analyst providing concise, data-driven insights for Algerian consumers.'},
                    {'role': 'user', 'content': prompt}
                ],
                'temperature': 0.7,
                'max_tokens': 800
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return {'error': False, 'insights': result['choices'][0]['message']['content']}
        else:
            return {'error': True, 'message': f'API Error: {response.status_code}'}
    except Exception as e:
        return {'error': True, 'message': f'Error: {str(e)}'}

# Prepare data for AI
kpi_data_ai = {
    'avg_price': avg_price,
    'price_change': price_change_str,
    'total_listings': total_listings,
    'unique_models': unique_models,
    'gpu_rate': gpu_rate,
    'top_brand': popular_brand,
    'brand_share': brand_pct,
}

market_data_ai = {
    'top_cpu': most_popular_cpu,
    'top_ram': avg_ram
}

st.markdown("---")

# ============== Trend Anomaly Detection ==============
st.markdown("### ⚠️ Trend Anomaly Detection")
st.markdown("<div style='color: #808080; margin-bottom: 16px;'>Identifying months with abnormal price movements compared to the market trend</div>", unsafe_allow_html=True)

# Prepare Data: Aggregate by Month to see the Trend
df_trend = df_filtered.copy().set_index('created_at')

# Resample by Month Start (MS) to get average price per month
# This smooths out noise from individual listings
monthly_trend = df_trend['price_preview'].resample('MS').mean().dropna()
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
        line=dict(color='#2196F3', width=1),
        marker=dict(size=8, color='#2196F3')
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
        st.markdown("#### 🔎 Detected Anomalies")
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
        st.info("✅ No significant trend anomalies detected. Prices are following the expected 3-month moving average.")

else:
    st.info("ℹ️ Not enough historical data available (less than 4 months) to calculate trend anomalies.")

st.markdown("---")

# Condition vs Price Analysis
st.markdown("### 📊 Condition Impact on Pricing")


df_filtered['standardized_condition'] = df_filtered['spec_Etat']

# Use standardized labels for the bar chart
condition_avg = df_filtered.groupby('standardized_condition')['price_preview'].mean().sort_values(ascending=False)

fig_condition = go.Figure()
fig_condition.add_trace(go.Bar(
    x=condition_avg.index,
    y=condition_avg.values,
    marker=dict(
        color=['#4CAF50', '#66BB6A', '#81C784', '#A5D6A7', '#BDBDBD'],
        line=dict(color='#FAFAFA', width=1)
    ),
    text=[f"{v/1000:.0f}K" for v in condition_avg.values],
    textposition='outside'
))

fig_condition.update_layout(
    template='plotly_dark',
    paper_bgcolor='#1a1a1a',
    plot_bgcolor='#262626',
    font=dict(color='#FAFAFA'),
    xaxis_title="Standardized Condition",
    yaxis_title="Average Price (DZD)",
    height=400,
    margin=dict(l=20, r=20, t=20, b=20)
)
st.plotly_chart(fig_condition, use_container_width=True)

st.markdown("---")
# AI Insights Button
if st.button("🚀 Generate AI Insights", type="primary", use_container_width=True):
    with st.spinner("🤖 AI is analyzing laptop market data..."):
        result = get_grok_insights_laptops(kpi_data_ai, market_data_ai)
        
        if result['error']:
            st.error(f"❌ {result['message']}")
            st.info("AI insights require API configuration. Using fallback analysis...")
            
            # Fallback insights based on data
            st.markdown(f"""
                <div class='insight-card'>
                    <div style='font-size: 1.1rem; line-height: 1.8;'>
                        <br>• <strong>Market Overview</strong>: Average laptop price is {avg_price/1000:.0f}K DZD with {price_change_str} change, indicating {'upward' if '+' in price_change_str else 'downward'} price pressure
                        <br><br>• <strong>GPU Market</strong>: {gpu_rate:.1f}% of laptops feature dedicated graphics cards, suggesting {'strong' if gpu_rate > 50 else 'moderate' if gpu_rate > 30 else 'limited'} gaming/professional demand
                        <br><br>• <strong>Brand Leadership</strong>: {popular_brand} dominates with {brand_pct:.1f}% market share across {unique_models} different models
                        <br><br>• <strong>Popular Configuration</strong>: Most common setup is {most_popular_cpu} with {avg_ram} RAM, representing the sweet spot for price-performance
                        <br><br>• <strong>Price Forecast</strong>: Predicted price of {predicted_price/1000:.0f}K DZD suggests {'+' if predicted_price > avg_price else '-'}{abs((predicted_price - avg_price)/avg_price * 100):.1f}% change in next 30 days
                        <br><br>• <strong>Regional Insights</strong>: Price variations across {df_filtered['city'].nunique()} cities suggest opportunities for regional arbitrage
                        <br><br>• <strong>Best Value</strong>: Look for {'used' if avg_price > 80000 else 'new'} laptops with {avg_ram} RAM and {'dedicated' if gpu_rate > 40 else 'integrated'} graphics for optimal value
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.success("✅ AI Insights generated successfully!")
            
            # Parse and display insights
            insights = result['insights']
            insights_html = insights.replace('**', '<strong>').replace('**', '</strong>')
            insights_html = insights_html.replace('• ', '<br><br>• ')
            
            st.markdown(f"""
                <div class='insight-card'>
                    <div style='font-size: 1.1rem; line-height: 1.8;'>
                        {insights_html}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<div style='margin-top: 12px; color: #808080; font-size: 0.85rem;'>💡 Powered by Grok AI | Insights based on real-time market data</div>", unsafe_allow_html=True)

st.markdown("---")


# Footer with data summary
st.markdown(f"""
    <div style='text-align: center; padding: 20px; color: #808080; border-top: 1px solid #3a3a3a;'>
        <strong>Data Summary:</strong> {total_listings:,} laptops analyzed | {unique_models} unique models | 
        {df_filtered['city'].nunique()} cities | Last updated: {max_date.strftime('%B %d, %Y')}
        <br><br>
        <em>Powered by DZA PriceSight AI | Real-time market intelligence for Algeria</em>
    </div>
""", unsafe_allow_html=True)