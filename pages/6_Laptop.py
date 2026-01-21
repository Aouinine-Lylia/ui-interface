import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, date
import os
import requests
import joblib
import glob
import re

# Page configuration
st.set_page_config(
    page_title="Laptops | DZA PriceSight",
    page_icon="💻",
    layout="wide"
)

# ============== Custom CSS (High Contrast & Styles) ==============
st.markdown("""
    <style>
    /* General Dark Theme Overrides */
    .stApp {background-color: #1a1a1a;}
    [data-testid="stSidebar"] {background-color: #262626; border-right: 1px solid #3a3a3a;}
    [data-testid="stSidebar"] .element-container {color: #FAFAFA;}
    h1, h2, h3, h4, h5, h6 {color: #FAFAFA !important;}
    p, div, span {color: #B0B0B0;}
    
    /* High Contrast Buttons */
    /* Targeting the primary buttons for Predict and Generate AI */
    div[data-testid="stForm"] button[kind="primary"],
    div[data-testid="column"] button[kind="primary"] {
        background-color: #ffeb3b !important; /* Bright Yellow for Predict */
        color: #000000 !important;
        font-weight: 800 !important;
        text-transform: uppercase;
        border: none;
        box-shadow: 0 0 10px rgba(255, 235, 59, 0.4);
    }
    
    /* Specific style for Generate AI button if not in a form */
    button[kind="primary"]:contains("Generate AI Insights") {
        background-color: #00e5ff !important; /* Cyan for AI */
        color: #000000 !important;
        font-weight: 800;
        box-shadow: 0 0 10px rgba(0, 229, 255, 0.4);
    }
    
    /* General button hover effects */
    button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }

    /* Card Styles */
    .kpi-card {
        background-color: #262626;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #3a3a3a;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 12px;
        height: 100%;
    }
    
    /* Section Descriptions */
    .section-desc {
        font-style: italic;
        color: #888;
        margin-bottom: 20px;
        font-size: 0.9rem;
        border-left: 3px solid #555;
        padding-left: 10px;
    }

    /* Utility Classes */
    .big-number { font-size: 2rem; font-weight: bold; color: #FAFAFA; margin: 4px 0; }
    .label { font-size: 0.8rem; color: #808080; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }
    .app-title { font-size: 1.8rem; font-weight: bold; color: #FAFAFA; margin: 0; }
    </style>
""", unsafe_allow_html=True)

# ============== Load & Preprocess Laptop Data ==============
@st.cache_data
def load_laptops():
    try:
        # 1. Find the CSV file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, 'kaggle_laptop_data.csv')
        
        if not os.path.exists(csv_path):
            csv_path = 'kaggle_laptop_data.csv'

        # Create dummy data if file missing for demo purposes (remove this block in production)
        if not os.path.exists(csv_path):
             st.warning("CSV file not found. Using dummy data for demonstration.")
             data = {
                 'created_at': pd.date_range(end=datetime.today(), periods=100),
                 'price_preview': np.random.randint(50000, 300000, 100),
                 'RAM_SIZE': np.random.choice([8, 16, 32], 100),
                 'SSD_SIZE': np.random.choice([256, 512, 1024], 100),
                 'HDD_SIZE': 0,
                 'SCREEN_SIZE': 15.6,
                 'spec_Etat': np.random.choice(['JAMAIS UTILISÉ', 'BON ÉTAT', 'MOYEN'], 100),
                 'CPU': ['Core i5 ' + str(i) + 'th Gen' for i in np.random.randint(8, 13, 100)],
                 'DEDICATED_GPU': np.random.choice(['INTEGRATED', 'RTX 3060', 'GTX 1650'], 100),
                 'model_name': np.random.choice(['HP', 'Dell', 'Lenovo', 'Asus'], 100),
                 'city': np.random.choice(['Algiers', 'Oran', 'Constantine', 'Annaba', 'Blida'], 100)
             }
             df = pd.DataFrame(data)
        else:
            df = pd.read_csv(csv_path)
        
        # 2. Parse Date
        def parse_date(date_str):
            try:
                clean_str = str(date_str).strip()
                clean_str = clean_str.replace('T', ' ').replace('Z', '')
                parts = clean_str.split()
                if len(parts) >= 4:
                    year, month, day = parts[0], parts[1], parts[2]
                    time_part = parts[3] if len(parts) > 3 else "00:00:00.000"
                    date_string = f"{year}-{month.zfill(2)}-{day.zfill(2)} {time_part}"
                    return pd.to_datetime(date_string, errors='coerce')
                # Fallback for standard ISO
                return pd.to_datetime(clean_str, errors='coerce')
            except:
                return pd.NaT
        
        if 'created_at' in df.columns:
            df['created_at'] = df['created_at'].apply(parse_date)
            df['date'] = df['created_at']
        else:
            df['date'] = pd.to_datetime('today')
            
        df = df.dropna(subset=['date', 'price_preview'])
        
        # 3. DATA CLEANING
        if 'RAM_SIZE' in df.columns:
            df['RAM_SIZE'] = df['RAM_SIZE'].astype(str).str.extract(r'(\d+)').astype(float).fillna(8.0)
        else:
            df['RAM_SIZE'] = 8.0

        # 4. Storage
        for col in ['SSD_SIZE', 'HDD_SIZE']:
            if col not in df.columns: df[col] = 0
            df[col] = df[col].astype(str).str.extract(r'(\d+)').astype(float).fillna(0)
        
        df['Total_Storage'] = df.get('SSD_SIZE', 0) + df.get('HDD_SIZE', 0)
        df['Total_Storage'] = df['Total_Storage'].replace(0, 256)

        # 5. Screen
        if 'SCREEN_SIZE' in df.columns:
            df['SCREEN_SIZE'] = pd.to_numeric(df['SCREEN_SIZE'], errors='coerce').fillna(15.6)
        else:
            df['SCREEN_SIZE'] = 15.6

        # 6. Price
        df['price_preview'] = pd.to_numeric(df['price_preview'], errors='coerce')
        df = df.dropna(subset=['price_preview'])
        
        # 7. Condition
        condition_mapping = {
            'JAMAIS UTILIS': 'JAMAIS UTILISÉ', 'BON TAT': 'BON ÉTAT',
            'MOYEN': 'MOYEN', 'NEUF': 'NEUF', 'TRES BON': 'TRÈS BON', 'ETAT NEUF': 'ÉTAT NEUF'
        }
        df['spec_Etat'] = df['spec_Etat'].fillna('Unknown')
        df['spec_Etat'] = df['spec_Etat'].str.upper().map(condition_mapping).fillna(df['spec_Etat'])
        
        # 8. CPU Logic
        def extract_cpu_gen(cpu_str):
            if pd.isna(cpu_str): return 10
            cpu_str = str(cpu_str).upper()
            match = re.search(r'(\d+)TH GEN|I\d\s*(\d)(?:\d{3})', cpu_str)
            if match: return int(match.group(1) or match.group(2))
            return 10
        df['CPU_Gen'] = df['CPU'].apply(extract_cpu_gen)
        
        def get_cpu_type(cpu_str):
            if pd.isna(cpu_str): return "Other"
            cpu_str = str(cpu_str).upper()
            if "I9" in cpu_str: return "Core i9"
            if "I7" in cpu_str: return "Core i7"
            if "I5" in cpu_str: return "Core i5"
            if "I3" in cpu_str: return "Core i3"
            if "RYZEN" in cpu_str: return "Ryzen"
            return "Other"
        df['CPU_Type'] = df['CPU'].apply(get_cpu_type)
        
        def get_cpu_tier(cpu_str):
            if pd.isna(cpu_str): return "Unknown"
            cpu_str = str(cpu_str).upper()
            if "I9" in cpu_str or "R9" in cpu_str: return "Ultra High-End"
            if "I7" in cpu_str or "R7" in cpu_str: return "High-End"
            if "I5" in cpu_str or "R5" in cpu_str: return "Mid-Range"
            if "I3" in cpu_str or "R3" in cpu_str or "CELERON" in cpu_str: return "Entry-Level"
            return "Other"
        df['cpu_tier'] = df['CPU'].apply(get_cpu_tier)
        
        # 9. GPU
        df['DEDICATED_GPU'] = df['DEDICATED_GPU'].fillna('INTEGRATED')
        df['Has_GPU'] = df['DEDICATED_GPU'].apply(lambda x: 0 if pd.isna(x) or str(x).upper() == 'INTEGRATED' or x == '' else 1)
        
        # 10. Mock columns
        if 'discount_flag' not in df.columns: df['discount_flag'] = 0 
        if 'stock_status' not in df.columns: df['stock_status'] = 'In Stock'
        df['model_name'] = df['model_name'].fillna('Unknown').str.upper().str.strip()
            
        return df
    except Exception as e:
        st.error(f"Error loading laptops data: {e}")
        return pd.DataFrame()

df_laptops = load_laptops()

@st.cache_resource
def load_my_model():
    try:
        model_path = 'pages/laptop_price_regression_model.pkl'
        if os.path.exists(model_path): return joblib.load(model_path)
        alt_path = 'laptop_price_regression_model.pkl'
        if os.path.exists(alt_path): return joblib.load(alt_path)
    except Exception as e: st.warning(f"Model loading failed: {e}")
    from sklearn.dummy import DummyRegressor
    return DummyRegressor(strategy='mean')

model = load_my_model()

# Sidebar
with st.sidebar:
    st.markdown("""
        <div style='text-align: left; padding: 20px 0;'>
            <span class='watermelon-icon'>🍉</span>
            <div style='display: inline-block; vertical-align: middle;'>
                <div class='app-title'>DZA PriceSight</div>
                <div class='app-subtitle'>AI Price Insights</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<p class='label'>Main</p>", unsafe_allow_html=True)
    st.page_link("landingPage.py", label="Overview", icon="🏠")
    st.markdown("---")
    st.markdown("<p class='label'>Categories</p>", unsafe_allow_html=True)
    st.page_link("pages/5_Food_Agriculture.py", label="Food & Agriculture", icon="🌾")
    st.page_link("pages/6_Laptop.py", label="Laptops", icon="💻")

# Session State Init
if 'brand_filter' not in st.session_state: st.session_state.brand_filter = 'All'
if 'gpu_filter' not in st.session_state: st.session_state.gpu_filter = False
if 'gaming_filter' not in st.session_state: st.session_state.gaming_filter = False

# ============== Header & Prediction ==============
col1, col2 = st.columns([4, 1])
with col1:
    st.markdown("# 💻 Laptop Price Intelligence")
    st.markdown("### Executive Market Dashboard - Algeria")
with col2:
    if st.button("🔮 Predict Price", use_container_width=True, key="predict_food_btn"):
        st.session_state.show_predict_form = True


if st.session_state.get('show_predict_form'):
    with st.expander("💻 Laptop Specification Input", expanded=True):
        with st.form("prediction_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                ram_gb = st.number_input("RAM (GB)", min_value=4, max_value=64, value=16, step=4)
                storage = st.number_input("Total Storage (GB)", min_value=128, max_value=2048, value=512, step=128)
                cpu_gen = st.number_input("CPU Generation", min_value=7, max_value=14, value=12)
                screen = st.number_input("Screen Size", min_value=11.0, max_value=18.0, value=15.6, step=0.1)
            with col_b:
                cpu_type_options = ["Core i3", "Core i5", "Core i7", "Core i9", "Ryzen", "Other"]
                cpu_type = st.selectbox("CPU Type", cpu_type_options, index=2)
                condition_options = ["JAMAIS UTILIS", "MOYEN", "Unknown"]
                condition = st.selectbox("Condition (État)", condition_options, index=0)
                has_gpu = st.checkbox("Has Dedicated GPU", value=True)
            listing_date = st.date_input("Listing Date", value=datetime.today().date())
            # Model name options from training set (one-hot columns)
            model_name_options = [
                "ALIENWARE", "ASPIRE", "BLADE", "COMPAQ", "DYNABOOK", "ELITEBOOK", "ENVY", "GALAXY", "GF", "IDEAPAD", "IMAC", "INSPIRON", "KATANA", "LATITUDE", "LEGION", "MAC", "MACBOOK", "NITRO", "OMEN", "OPTIPLEX", "PAVILION", "PRECISION", "PREDATOR", "PROBOOK", "ROG", "SPECTRE", "SPIN", "STEALTH", "STRIX", "SURFACE", "SWIFT", "SWORD", "THINKBOOK", "THINKPAD", "TRANSFORMER", "TRAVELMATE", "TUF", "VECTOR", "VICTUS", "VIVOBOOK", "VOSTRO", "XPS", "YOGA", "ZBOOK", "ZENBOOK"
            ]
            model_name = st.selectbox("Model Name (Brand)", model_name_options, index=11)
            if st.form_submit_button("Calculate Estimated Price (DZD)"):
                try:
                    days_since_posted = (datetime.now().date() - listing_date).days
                    # One-hot encoding for CPU_Type
                    cpu_type_dict = {f"CPU_Type_{t}": 1 if cpu_type == t else 0 for t in cpu_type_options}
                    # One-hot encoding for Condition
                    condition_dict = {f"Condition_{c}": 1 if condition == c else 0 for c in condition_options}
                    # One-hot encoding for model_name
                    model_name_dict = {f"model_name_{m}": 1 if model_name == m else 0 for m in model_name_options}
                    features = {
                        "RAM_GB": ram_gb,
                        "Total_Storage": storage,
                        "CPU_Gen": cpu_gen,
                        "Has_GPU": 1 if has_gpu else 0,
                        "SCREEN_SIZE": screen,
                        "days_since_posted": days_since_posted,
                    }
                    features.update(cpu_type_dict)
                    features.update(condition_dict)
                    features.update(model_name_dict)
                    # Required column order
                    required_columns = [
                        'RAM_GB', 'Total_Storage', 'CPU_Gen', 'Has_GPU', 'SCREEN_SIZE',
                        'days_since_posted', 'CPU_Type_Core i3', 'CPU_Type_Core i5',
                        'CPU_Type_Core i7', 'CPU_Type_Core i9', 'CPU_Type_Other',
                        'CPU_Type_Ryzen', 'Condition_JAMAIS UTILIS', 'Condition_MOYEN',
                        'Condition_Unknown', 'model_name_ALIENWARE', 'model_name_ASPIRE',
                        'model_name_BLADE', 'model_name_COMPAQ', 'model_name_DYNABOOK',
                        'model_name_ELITEBOOK', 'model_name_ENVY', 'model_name_GALAXY',
                        'model_name_GF', 'model_name_IDEAPAD', 'model_name_IMAC',
                        'model_name_INSPIRON', 'model_name_KATANA', 'model_name_LATITUDE',
                        'model_name_LEGION', 'model_name_MAC', 'model_name_MACBOOK',
                        'model_name_NITRO', 'model_name_OMEN', 'model_name_OPTIPLEX',
                        'model_name_PAVILION', 'model_name_PRECISION', 'model_name_PREDATOR',
                        'model_name_PROBOOK', 'model_name_ROG', 'model_name_SPECTRE',
                        'model_name_SPIN', 'model_name_STEALTH', 'model_name_STRIX',
                        'model_name_SURFACE', 'model_name_SWIFT', 'model_name_SWORD',
                        'model_name_THINKBOOK', 'model_name_THINKPAD', 'model_name_TRANSFORMER',
                        'model_name_TRAVELMATE', 'model_name_TUF', 'model_name_VECTOR',
                        'model_name_VICTUS', 'model_name_VIVOBOOK', 'model_name_VOSTRO',
                        'model_name_XPS', 'model_name_YOGA', 'model_name_ZBOOK',
                        'model_name_ZENBOOK'
                    ]
                    input_df = pd.DataFrame([features])
                    input_df = input_df.reindex(columns=required_columns, fill_value=0)
                    prediction = model.predict(input_df)[0]
                    if prediction < 1000: prediction = np.exp(prediction)
                    st.success(f"### Estimated Market Price: {prediction:,.0f} DZD")
                    st.info(f"This is approximately {prediction/220:.0f} USD (at 220 DZD/USD)")
                    st.caption(f"Model included listing age factor: {days_since_posted} days.")
                except Exception as e:
                    st.error(f"Prediction error (Model might not support new feature): {e}")
                    fallback = df_laptops['price_preview'].median()
                    st.warning(f"Using median market price: {fallback:,.0f} DZD")

st.markdown("---")

# ============== Filters ==============
st.markdown("### 🔍 Market Filters")
f1, f2, f3, f4 = st.columns(4)
with f1:
    brands = ['All'] + sorted([str(x) for x in df_laptops['model_name'].unique().tolist() if x != 'Unknown'])
    st.session_state.brand_filter = st.selectbox("Brand", brands)
with f2:
    st.session_state.gpu_filter = st.checkbox("Dedicated GPU Only")
with f3:
    st.session_state.gaming_filter = st.checkbox("Gaming (RTX/GTX) Only")
with f4:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Apply Filters"):
        st.rerun()

# Apply Logic
df = df_laptops.copy()
if st.session_state.brand_filter != 'All': df = df[df['model_name'] == st.session_state.brand_filter]
if st.session_state.gpu_filter: df = df[df['DEDICATED_GPU'] != 'INTEGRATED']
if st.session_state.gaming_filter: df = df[df['DEDICATED_GPU'].str.contains('RTX|GTX', case=False, na=False)]

if len(df) > 0:
    if df['price_preview'].nunique() >= 3:
        df['segment'] = pd.qcut(df['price_preview'], 3, labels=['Budget', 'Mid-Range', 'Premium'], duplicates='drop')
    else: df['segment'] = 'Mid-Range'
else: df['segment'] = 'Unknown'

df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
monthly_data = df.groupby('month').agg(median_price=('price_preview', 'median'), mean_price=('price_preview', 'mean'), count=('price_preview', 'count')).reset_index()

# ==========================================
# 1. EXECUTIVE OVERVIEW
# ==========================================
st.markdown("## 1. Executive Overview - Algerian Laptop Market")
st.caption("High-level key performance indicators for the current market snapshot.")

# KPI Logic
current_median = df['price_preview'].median() if len(df) > 0 else 0
current_mean = df['price_preview'].mean() if len(df) > 0 else 0

baseline_median = monthly_data.iloc[0]['median_price'] if len(monthly_data) > 1 else current_median
median_change_pct = ((current_median - baseline_median) / baseline_median) * 100 if baseline_median > 0 else 0
price_index = (current_median / baseline_median) * 100 if baseline_median > 0 else 100

trend_indicator = "N/A"
trend_color = "#66BB6A" # Default green
if len(monthly_data) >= 2:
    recent = monthly_data.tail(3)
    if len(recent) > 1:
        slope, _ = np.polyfit(range(len(recent)), recent['median_price'], 1)
        trend_indicator = "↑" if slope > 500 else ("↓" if slope < -500 else "→")
        trend_color = "#EF5350" if slope > 500 else ("#66BB6A" if slope < -500 else "#FFA726")

volatility = (df['price_preview'].std() / df['price_preview'].mean()) if len(df) > 0 and df['price_preview'].mean() > 0 else 0
total_listings = len(df)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(f"""
        <div class='kpi-card'>
            <div class='label'>Median Price (DZD)</div>
            <div class='big-number'>{current_median:,.0f}</div>
            <div class='label' style='margin-top:8px'>≈ {current_median/220:.0f} USD | Δ: {median_change_pct:+.1f}%</div>
        </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
        <div class='kpi-card'>
            <div class='label'>Price Index (Base=100)</div>
            <div class='big-number'>{price_index:.1f}</div>
            <div class='label' style='margin-top:8px'>Market Listings: {total_listings}</div>
        </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
        <div class='kpi-card'>
            <div class='label'>Trend Direction</div>
            <div class='big-number'><span style='color:{trend_color}; font-size:2.5rem'>{trend_indicator}</span></div>
            <div class='label' style='margin-top:8px'>Monthly Movement</div>
        </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
        <div class='kpi-card'>
            <div class='label'>Market Volatility (CV)</div>
            <div class='big-number'>{volatility:.2f}</div>
            <div class='label' style='margin-top:8px'>{"High Variation" if volatility > 0.5 else "Stable Market"}</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# 2. PRICE EVOLUTION
# ==========================================
st.markdown("## 2. Price Evolution Over Time")
st.caption("Historical analysis of pricing trends and item condition distribution.")

c1, c2 = st.columns([2, 1])
with c1:
    st.markdown("### Median Price Trend (DZD)")
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=monthly_data['month'], y=monthly_data['median_price'], mode='lines+markers', name='Median Price', line=dict(color='#2196F3', width=3), marker=dict(size=8)))
    fig_line.update_layout(template='plotly_dark', height=300, margin=dict(l=0, r=0, t=20, b=0), xaxis_title="", yaxis_title="Price (DZD)", hovermode='x unified')
    st.plotly_chart(fig_line, use_container_width=True)

with c2:
    st.markdown("### Condition Distribution")
    condition_counts = df['spec_Etat'].value_counts()
    fig_pie = px.pie(values=condition_counts.values, names=condition_counts.index, title="Laptop Conditions in Market")
    fig_pie.update_layout(template='plotly_dark', height=300, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("### Price Distribution by Segment")
fig_box = px.box(df, x='segment', y='price_preview', color='segment', color_discrete_map={'Budget':'#4CAF50', 'Mid-Range':'#FFC107', 'Premium':'#F44336'}, points='outliers')
fig_box.update_layout(template='plotly_dark', height=400, xaxis_title="Market Segment", yaxis_title="Price (DZD)", showlegend=False)
st.plotly_chart(fig_box, use_container_width=True)

st.markdown("---")

# ==========================================
# 3. MARKET SEGMENTATION
# ==========================================
st.markdown("## 3. Market Segmentation Analysis")
st.caption("Comparing value across hardware tiers and brand popularity.")

c1, c2 = st.columns(2)
with c1:
    st.markdown("### Price by CPU Tier")
    cpu_tier_price = df.groupby('cpu_tier')['price_preview'].median().sort_values(ascending=False)
    fig_cpu = px.bar(x=cpu_tier_price.values, y=cpu_tier_price.index, orientation='h', title="Median Price by Processor Tier")
    fig_cpu.update_layout(template='plotly_dark', height=350, xaxis_title="Price (DZD)", yaxis_title="")
    st.plotly_chart(fig_cpu, use_container_width=True)

with c2:
    st.markdown("### Top Brands by Volume")
    brand_counts = df['model_name'].value_counts().head(10)
    fig_brands = px.bar(x=brand_counts.values, y=brand_counts.index, orientation='h', title="Most Listed Laptop Brands")
    fig_brands.update_layout(template='plotly_dark', height=350, xaxis_title="Listings", yaxis_title="")
    st.plotly_chart(fig_brands, use_container_width=True)

st.markdown("### Value Analysis: Price per GB of RAM")
if df['RAM_SIZE'].sum() > 0:
    df['price_per_gb_ram'] = df['price_preview'] / df['RAM_SIZE'].replace(0, 1)
    value_data = df.groupby('model_name')['price_per_gb_ram'].mean().sort_values(ascending=True).head(10)
    fig_value = px.bar(x=value_data.values, y=value_data.index, orientation='h', title="Best Value Laptops (Lower = Better)", labels={'x': 'DZD per GB RAM', 'y': 'Brand'})
    fig_value.update_layout(template='plotly_dark', height=400)
    fig_value.update_traces(marker_color='#66BB6A')
    st.plotly_chart(fig_value, use_container_width=True)

st.markdown("---")

# ==========================================
# 4. ALGERIAN MARKET SPECIFIC INSIGHTS
# ==========================================
st.markdown("## 4. Algerian Market Specific Insights")
st.caption("Local analysis including geography, condition preferences, and regional pricing.")

m1, m2, m3 = st.columns(3)
with m1:
    used_count = df[df['spec_Etat'] != 'JAMAIS UTILISÉ'].shape[0]
    used_pct = (used_count / len(df)) * 100 if len(df) > 0 else 0
    st.metric("Used/Reconditioned Market Share", f"{used_pct:.1f}%")

with m2:
    gpu_count = df[df['Has_GPU'] == 1].shape[0]
    gpu_pct = (gpu_count / len(df)) * 100 if len(df) > 0 else 0
    st.metric("Laptops with Dedicated GPU", f"{gpu_pct:.1f}%")

with m3:
    avg_storage = df['Total_Storage'].mean() if 'Total_Storage' in df.columns else 0
    st.metric("Average Storage Capacity", f"{avg_storage:.0f} GB")

# NEW: Cheapest Cities Graph
if 'city' in df.columns and df['city'].notna().sum() > 0:
    st.markdown("### Cities by Average Price (Cheapest to Most Expensive)")
    st.caption("Identify regions with the lowest average listing costs to find better deals.")
    
    # Calculate average price per city and sort ascending (Cheapest first)
    city_avg_price = df.groupby('city')['price_preview'].mean().sort_values(ascending=True)
    
    fig_cities = px.bar(
        x=city_avg_price.values,
        y=city_avg_price.index,
        orientation='h',
        title="Average Price by City (DZD)",
        labels={'x': 'Average Price (DZD)', 'y': 'City'},
        color=city_avg_price.values, # Color gradient based on price
        color_continuous_scale='Viridis_r' # Reverse Viridis (Green is low/cheap, Purple is high/expensive)
    )
    fig_cities.update_layout(template='plotly_dark', height=400, xaxis_title="Avg Price (DZD)", yaxis_title="")
    st.plotly_chart(fig_cities, use_container_width=True)

st.markdown("---")

# ==========================================
# 5. OUTLIERS & ALERTS
# ==========================================
st.markdown("## 5. Outliers & Price Alerts")
st.caption("Identify unusual pricing deviations in the current dataset.")

if len(df) > 0:
    df['z_score'] = np.abs((df['price_preview'] - df['price_preview'].mean()) / df['price_preview'].std())
    outliers = df[df['z_score'] > 2.5].sort_values('price_preview', ascending=False)
    if not outliers.empty:
        st.markdown("### ⚠️ Unusual Pricing Detected")
        for _, row in outliers.head(5).iterrows():
            price_color = '#F44336' if row['price_preview'] > current_median * 1.5 else '#66BB6A'
            st.markdown(f"""
            <div style='background:#333; padding:10px; border-radius:5px; margin-bottom:5px; border-left: 3px solid {price_color}'>
            <strong>{row['model_name']}</strong> | {row['CPU']} | {row['spec_Etat']}<br>
            Price: <span style='color:{price_color}'>{row['price_preview']:,.0f} DZD</span> 
            ({row['price_preview']/220:.0f} USD)
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No extreme price outliers detected - Market is stable")
else:
    st.warning("No data to analyze outliers.")

st.markdown("---")

# ============== FORECASTING ==============
st.markdown("## 7. Market Forecasting")
st.caption("Projected price movements based on historical data.")

# NEW NOTE
st.info("ℹ️ **Note:** These forecasts estimate expected market price movement, not individual laptop prices.")

avg_price = df['price_preview'].mean() if len(df) > 0 else 0
next_pred = avg_price

FORECAST_DIR = os.path.join(os.path.dirname(__file__), "forecasting_laptops")

if os.path.exists(FORECAST_DIR) and not df_laptops.empty:
    forecast_files = glob.glob(os.path.join(FORECAST_DIR, '*.csv'))
    
    def get_readable_condition(condition_key):
        key_norm = condition_key.upper().replace('_', ' ')
        mapping = {"BON TAT": "Good State", "JAMAIS UTILIS": "Never Used", "MOYEN": "Fair", "NEUF": "New", "TRES BON": "Very Good", "ETAT NEUF": "Brand New"}
        return mapping.get(key_norm, condition_key.replace('_', ' ').title())

    forecast_options = {}
    for file in forecast_files:
        filename = os.path.basename(file)
        name_part = filename.replace('_forecast.csv', '')
        parts = name_part.rsplit('_', 1)
        if len(parts) == 2:
            condition_raw = parts[0]
            tier = parts[1]
            condition_readable = get_readable_condition(condition_raw)
            display_name = f"{condition_readable} | {tier}"
            forecast_options[display_name] = filename
        else:
            display_name = name_part.replace('_', ' ').title()
            forecast_options[display_name] = filename

    sorted_display_names = sorted(forecast_options.keys())
    selected_display_name = st.selectbox("Available Forecasts", sorted_display_names, label_visibility="collapsed")
    selected_filename = forecast_options[selected_display_name]
    forecast_path = os.path.join(FORECAST_DIR, selected_filename)
    
    try:
        df_future = pd.read_csv(forecast_path)
        df_future.columns = [str(c).lower() if isinstance(c, str) else c for c in df_future.columns]
        rename_map = {'ds': 'date', 'yhat': 'predicted_price', 'yhat_lower': 'lower_bound', 'yhat_upper': 'upper_bound'}
        df_future = df_future.rename(columns={k:v for k,v in rename_map.items() if k in df_future.columns})
        
        required_columns = ['date', 'predicted_price']
        if all(col in df_future.columns for col in required_columns):
            df_future['date'] = pd.to_datetime(df_future['date'])
            next_pred = df_future['predicted_price'].iloc[0]
            
            kpi_f1, kpi_f3 = st.columns(2)
            with kpi_f1: st.metric("Next Month Prediction", f"{next_pred:,.0f} DZD")
            with kpi_f3: st.metric("Forecast Horizon", f"{len(df_future)} Months")

            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=df_future['date'], y=df_future['predicted_price'], mode='lines+markers', name='Forecast', line=dict(color='#2196F3')))
            if 'lower_bound' in df_future.columns:
                fig_f.add_trace(go.Scatter(x=df_future['date'], y=df_future['upper_bound'], mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
                fig_f.add_trace(go.Scatter(x=df_future['date'], y=df_future['lower_bound'], mode='lines', line_color='rgba(0,0,0,0)', fill='tonexty', fillcolor='rgba(33, 150, 243, 0.2)', name='95% CI'))
            
            fig_f.update_layout(template='plotly_dark', height=400, xaxis_title="Date", yaxis_title="Price (DZD)")
            st.plotly_chart(fig_f, use_container_width=True)
        else:
            st.error("Forecast file missing required columns.")
    except Exception as e:
        st.error(f"Error loading forecast: {e}")
else:
    st.info("⚠️ Forecast folder not found.")

st.markdown("---")

# ============== AI INSIGHTS (Kept from Original) ==============
st.markdown("## 8. AI-Powered Market Insights")
st.caption("Generative analysis of current market conditions.")

top_brand = df['model_name'].value_counts().index[0] if len(df) > 0 else "Unknown"
price_trend = "Up" if trend_indicator == "↑" else ("Down" if trend_indicator == "↓" else "Stable")

context = f"""
- Average Price: {avg_price:,.0f} DZD
- Price Trend: {price_trend}
- Top Segment: {df['segment'].value_counts().index[0] if len(df) > 0 else 'Unknown'}
- Top Brand: {top_brand}
- Forecast Prediction: {next_pred:,.0f} DZD
"""

def get_ai_insights(context):
    api_key = st.secrets["grok"]["api_key"]
    try:
        response = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
            json={
                'model': 'llama-3.1-8b-instant',
                'messages': [
                    {'role': 'system', 'content': 'You are a retail analyst. Provide 3-5 concise bullet points using the data provided.'},
                    {'role': 'user', 'content': f"Analyze this laptop market data:\n{context}"}
                ],
                'temperature': 0.7,
                'max_tokens': 500
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        return None
    except:
        return None

# HIGH CONTRAST BUTTON
if st.button("🚀 Generate AI Insights" ,use_container_width=True, key="ai_btn"):
    with st.spinner("Analyzing..."):
        insights = get_ai_insights(context)
        if insights:
            st.markdown(f"### 🤖 Analysis\n{insights.replace('•', '<br>•')}", unsafe_allow_html=True)
        else:
            st.info("AI Service unavailable. Showing basic stats.")
            st.write(f"Market is currently trending {price_trend} with a median price of {current_median:,.0f} DZD.")

# Footer
st.markdown("<div style='text-align:center; color:#666; margin-top:40px;'>DZA PriceSight | Retail Intelligence Dashboard</div>", unsafe_allow_html=True)