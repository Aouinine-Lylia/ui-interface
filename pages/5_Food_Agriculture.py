import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import requests
import glob

# Page configuration
st.set_page_config(
    page_title="Food & Agriculture | DZA PriceSight",
    page_icon="🌾",
    layout="wide"
)

# ============== CONSTANTS ==============
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

ALGERIAN_WILAYAS_COORDS = {
    "Adrar": {"lat": 27.88, "lon": -0.28}, "Chlef": {"lat": 36.16, "lon": 1.33},
    "Laghouat": {"lat": 33.80, "lon": 2.88}, "Oum El Bouaghi": {"lat": 35.86, "lon": 7.11},
    "Batna": {"lat": 35.55, "lon": 6.17}, "Béjaïa": {"lat": 36.75, "lon": 5.06},
    "Biskra": {"lat": 34.85, "lon": 5.73}, "Béchar": {"lat": 31.61, "lon": -2.22},
    "Blida": {"lat": 36.47, "lon": 2.83}, "Bouira": {"lat": 36.37, "lon": 3.90},
    "Tamanrasset": {"lat": 22.78, "lon": 5.52}, "Tébessa": {"lat": 35.40, "lon": 8.12},
    "Tlemcen": {"lat": 34.88, "lon": -1.31}, "Tiaret": {"lat": 35.37, "lon": 1.32},
    "Tizi Ouzou": {"lat": 36.70, "lon": 4.05}, "Alger": {"lat": 36.75, "lon": 3.04},
    "Djelfa": {"lat": 34.67, "lon": 3.26}, "Jijel": {"lat": 36.80, "lon": 5.76},
    "Sétif": {"lat": 36.19, "lon": 5.41}, "Saïda": {"lat": 34.83, "lon": 0.14},
    "Skikda": {"lat": 36.87, "lon": 6.91}, "Sidi Bel Abbès": {"lat": 35.19, "lon": -0.64},
    "Annaba": {"lat": 36.89, "lon": 7.76}, "Guelma": {"lat": 36.46, "lon": 7.43},
    "Constantine": {"lat": 36.36, "lon": 6.61}, "Médéa": {"lat": 36.26, "lon": 2.75},
    "Mostaganem": {"lat": 35.93, "lon": 0.09}, "Msila": {"lat": 35.70, "lon": 4.54},
    "Mascara": {"lat": 35.39, "lon": 0.14}, "Ouargla": {"lat": 31.95, "lon": 5.32},
    "Oran": {"lat": 35.69, "lon": -0.64}, "El Bayadh": {"lat": 33.68, "lon": 1.01},
    "Illizi": {"lat": 26.49, "lon": 8.44}, "Bordj Bou Arreridj": {"lat": 36.07, "lon": 4.76},
    "Boumerdès": {"lat": 36.75, "lon": 3.47}, "El Tarf": {"lat": 36.77, "lon": 8.30},
    "Tindouf": {"lat": 27.67, "lon": -8.14}, "Tissemsilt": {"lat": 35.60, "lon": 1.81},
    "El Oued": {"lat": 33.35, "lon": 6.86}, "Khenchela": {"lat": 35.42, "lon": 7.14},
    "Souk Ahras": {"lat": 36.28, "lon": 7.95}, "Tipaza": {"lat": 36.58, "lon": 2.45},
    "Mila": {"lat": 36.45, "lon": 6.26}, "Aïn Defla": {"lat": 36.25, "lon": 1.95},
    "Naâma": {"lat": 33.26, "lon": -0.31}, "Aïn Témouchent": {"lat": 35.30, "lon": -1.14},
    "Ghardaïa": {"lat": 32.49, "lon": 3.67}, "Relizane": {"lat": 35.93, "lon": 0.55},
    "Timimoun": {"lat": 29.25, "lon": 0.25}, "Bordj Badji Mokhtar": {"lat": 21.33, "lon": 0.95},
    "Ouled Djellal": {"lat": 34.52, "lon": 4.72}, "Béni Abbès": {"lat": 30.13, "lon": -2.16},
    "In Salah": {"lat": 27.20, "lon": 2.47}, "In Guezzam": {"lat": 23.00, "lon": 5.70},
    "Touggourt": {"lat": 33.10, "lon": 6.06}, "Djanet": {"lat": 24.55, "lon": 9.47},
    "El M'Ghair": {"lat": 33.30, "lon": 6.30}, "El Meniaa": {"lat": 30.58, "lon": 2.91}
}

# ============== CSS (Matching Laptop Dashboard) ==============
st.markdown("""
    <style>
    .stApp {background-color: #1a1a1a;}
    [data-testid="stSidebar"] {background-color: #262626; border-right: 1px solid #3a3a3a;}
    [data-testid="stSidebar"] .element-container {color: #FAFAFA;}
    h1, h2, h3, h4, h5, h6 {color: #FAFAFA !important;}
    p, div, span {color: #B0B0B0;}
    
    /* High Contrast Buttons */
    div[data-testid="stForm"] button[kind="primary"],
    div[data-testid="column"] button[kind="primary"] {
        background-color: #ffeb3b !important; 
        color: #000000 !important;
        font-weight: 800 !important;
        text-transform: uppercase;
        border: none;
        box-shadow: 0 0 10px rgba(255, 235, 59, 0.4);
    }
    button:hover { opacity: 0.9; transform: translateY(-1px); }

    .kpi-card {
        background-color: #262626;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #3a3a3a;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 12px;
        height: 100%;
    }
    
    .section-desc {
        font-style: italic;
        color: #888;
        margin-bottom: 20px;
        font-size: 0.9rem;
        border-left: 3px solid #555;
        padding-left: 10px;
    }

    .big-number { font-size: 2rem; font-weight: bold; color: #FAFAFA; margin: 4px 0; }
    .label { font-size: 0.8rem; color: #808080; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }
    .app-title { font-size: 1.8rem; font-weight: bold; color: #FAFAFA; margin: 0; }
    </style>
""", unsafe_allow_html=True)

# ============== Sidebar ==============
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
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

    return df_prices

df_prices = load_food_data()

if df_prices.empty:
    st.error("No data available.")
    st.stop()

# ============== Header & Prediction ==============
col1, col2 = st.columns([4, 1])
with col1:
    st.markdown("# 🌾 Food & Agriculture Intelligence")
    st.markdown("### Executive Market Dashboard - Algeria")
with col2:
    if st.button("🔮 Predict Price", use_container_width=True, key="predict_food_btn"):
        st.session_state.show_predict_form = True

# Prediction Form
if st.session_state.get('show_predict_form', False):
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
            lat = ALGERIAN_WILAYAS_COORDS[wilaya]["lat"]
            lon = ALGERIAN_WILAYAS_COORDS[wilaya]["lon"]
            payload = {
                "commodity": commodity, "category": category, "price_per_kg": price_per_kg,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "latitude": lat, "longitude": lon
            }
            try:
                res = requests.post("http://localhost:5000/predict-food", json=payload, timeout=10)
                if res.status_code == 200:
                    result = res.json()
                    pred_label = result.get('prediction', 'Unknown')
                    conf = result.get('confidence', 0.0)
                    badge_color = "#66BB6A" if pred_label == 'CHEAP' else ("#EF5350" if pred_label == 'EXPENSIVE' else "#42A5F5")
                    st.success(f"**Status:** {pred_label} | **Confidence:** {conf*100:.1f}%")
                    st.info(f"Deviation: {result.get('price_deviation_percentage', 0):+.1f}%")
                else:
                    st.error(f"API Error: {res.status_code}")
            except Exception as e:
                st.error(f"Connection Error: {e}")

st.markdown("---")

# ============== Filters ==============
st.markdown("### 🔍 Data Filters")
f1, f2, f3 = st.columns(3)
with f1:
    all_commodities = sorted(df_prices['commodity'].unique().tolist())
    selected_commodities = st.multiselect("Commodity", all_commodities, default=None)
with f2:
    all_regions = sorted(df_prices['admin1'].unique().tolist()) if 'admin1' in df_prices.columns else []
    selected_regions = st.multiselect("Region (Wilaya)", all_regions, default=[])
with f3:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Apply Filters"): st.rerun()

# Apply Filters
df_filtered = df_prices.copy()
if selected_commodities: df_filtered = df_filtered[df_filtered['commodity'].isin(selected_commodities)]
if selected_regions: df_filtered = df_filtered[df_filtered['admin1'].isin(selected_regions)]

if df_filtered.empty:
    st.warning("No data matches filters.")
    st.stop()

# Prepare Monthly Data for Trends
df_filtered = df_filtered.sort_values('date')
df_filtered['month'] = df_filtered['date'].dt.to_period('M').dt.to_timestamp()
monthly_data = df_filtered.groupby('month').agg(
    median_price=('price', 'median'),
    mean_price=('price', 'mean'),
    count=('price', 'count')
).reset_index()

# ==========================================
# 1. EXECUTIVE OVERVIEW
# ==========================================
st.markdown("## 1. Executive Overview - Food Market")
st.caption("High-level key performance indicators for the selected market scope.")

# Metrics Logic
current_median = df_filtered['price'].median()
current_mean = df_filtered['price'].mean()

if len(monthly_data) >= 2:
    last_month_val = monthly_data.iloc[-1]['median_price']
    prev_month_val = monthly_data.iloc[-2]['median_price']
    trend_change_pct = ((last_month_val - prev_month_val) / prev_month_val) * 100 if prev_month_val > 0 else 0
else:
    trend_change_pct = 0

trend_indicator = "↑" if trend_change_pct > 0 else ("↓" if trend_change_pct < 0 else "→")
trend_color = "#EF5350" if trend_change_pct > 2 else ("#66BB6A" if trend_change_pct < -2 else "#FFA726")

volatility = (df_filtered['price'].std() / df_filtered['price'].mean()) if df_filtered['price'].mean() > 0 else 0
total_records = len(df_filtered)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(f"""
        <div class='kpi-card'>
            <div class='label'>Median Price (DZD/kg)</div>
            <div class='big-number'>{current_median:,.0f}</div>
            <div class='label' style='margin-top:8px'>MoM Change: {trend_change_pct:+.1f}%</div>
        </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
        <div class='kpi-card'>
            <div class='label'>Price Trend</div>
            <div class='big-number'><span style='color:{trend_color}; font-size:2.5rem'>{trend_indicator}</span></div>
            <div class='label' style='margin-top:8px'>Monthly Direction</div>
        </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
        <div class='kpi-card'>
            <div class='label'>Market Volatility</div>
            <div class='big-number'>{volatility:.2f}</div>
            <div class='label' style='margin-top:8px'>{"High Variance" if volatility > 0.5 else "Stable"}</div>
        </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
        <div class='kpi-card'>
            <div class='label'>Market Volume</div>
            <div class='big-number'>{total_records:,}</div>
            <div class='label' style='margin-top:8px'>Data Points Analyzed</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# 2. PRICE EVOLUTION
# ==========================================
st.markdown("## 2. Price Evolution Over Time")
st.caption("Historical analysis of pricing trends and distribution.")

c1, c2 = st.columns([2, 1])
with c1:
    st.markdown("### Median Price Trend")
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=monthly_data['month'], 
        y=monthly_data['median_price'], 
        mode='lines+markers', 
        name='Median Price',
        line=dict(color='#4CAF50', width=3), 
        marker=dict(size=8)
    ))
    fig_trend.update_layout(template='plotly_dark', height=300, margin=dict(l=0, r=0, t=20, b=0), xaxis_title="", yaxis_title="Price (DZD)", hovermode='x unified')
    st.plotly_chart(fig_trend, use_container_width=True)

with c2:
    st.markdown("### Price Distribution")
    fig_hist = px.histogram(df_filtered, x="price", nbins=30, color_discrete_sequence=['#4CAF50'])
    fig_hist.update_layout(template='plotly_dark', height=300, margin=dict(l=0, r=0, t=20, b=0), xaxis_title="Price (DZD)", yaxis_title="Count", showlegend=False)
    st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")

# ==========================================
# 3. MARKET SEGMENTATION
# ==========================================
st.markdown("## 3. Market Segmentation Analysis")
st.caption("Comparing value across commodity categories and product popularity.")

c1, c2 = st.columns(2)
with c1:
    st.markdown("### Price by Category")
    if 'category' in df_filtered.columns:
        cat_price = df_filtered.groupby('category')['price'].median().sort_values(ascending=False)
        fig_cat = px.bar(x=cat_price.values, y=cat_price.index, orientation='h', title="Median Price by Category")
        fig_cat.update_layout(template='plotly_dark', height=400, xaxis_title="Price (DZD)", yaxis_title="")
        st.plotly_chart(fig_cat, use_container_width=True)
    else:
        st.info("Category data not available.")

with c2:
    st.markdown("### Top Commodities by Volume")
    comm_counts = df_filtered['commodity'].value_counts().head(10)
    fig_comm = px.bar(x=comm_counts.values, y=comm_counts.index, orientation='h', title="Most Tracked Commodities")
    fig_comm.update_layout(template='plotly_dark', height=400, xaxis_title="Listings", yaxis_title="")
    st.plotly_chart(fig_comm, use_container_width=True)

st.markdown("---")

# ==========================================
# 4. REGIONAL & GEOGRAPHIC INSIGHTS
# ==========================================
st.markdown("## 4. Regional Market Insights")
st.caption("Local analysis including geographic pricing and Wilaya comparisons.")

m1, m2 = st.columns([1, 1])
with m1:
    st.markdown("### Wilayas by Average Price (Cheapest to Most Expensive)")
    st.caption("Identify regions with the lowest average food costs.")
    
    if 'admin1' in df_filtered.columns:
        wilaya_avg_price = df_filtered.groupby('admin1')['price'].mean().sort_values(ascending=True)
        fig_wilaya = px.bar(
            x=wilaya_avg_price.values,
            y=wilaya_avg_price.index,
            orientation='h',
            title="Average Price by Wilaya (DZD)",
            color=wilaya_avg_price.values,
            color_continuous_scale='Viridis_r'
        )
        fig_wilaya.update_layout(template='plotly_dark', height=500, xaxis_title="Avg Price (DZD)", yaxis_title="")
        st.plotly_chart(fig_wilaya, use_container_width=True)

with m2:
    st.markdown("### Geographic Distribution")
    st.caption("Visual heatmap of price points across the country.")
    
    if 'latitude' in df_filtered.columns and 'longitude' in df_filtered.columns:
        map_data = df_filtered.groupby(['admin1', 'market', 'latitude', 'longitude'])['price'].mean().reset_index()
        map_data = map_data.dropna(subset=['latitude', 'longitude'])
        
        if not map_data.empty:
            fig_map = px.scatter_mapbox(
                map_data, lat="latitude", lon="longitude", size="price", color="price",
                hover_name="market", hover_data={"admin1": True, "price": ":.2f"},
                color_continuous_scale=px.colors.sequential.Viridis, size_max=25, zoom=5,
                center={"lat": 34.0, "lon": 3.0}
            )
            fig_map.update_layout(mapbox_style="carto-darkmatter", margin={"r":0,"t":0,"l":0,"b":0}, height=500)
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("No coordinates available.")
    else:
        st.info("Geographic data missing.")

st.markdown("---")

# ==========================================
# 5. FORECASTING (MATCHING LAPTOP UI)
# ==========================================
st.markdown("## 5. Market Forecasting")
st.caption("Projected price movements based on historical data.")

# New Note similar to Laptop dashboard
st.info("ℹ️ **Note:** These forecasts estimate expected market price movement for the selected commodity.")

# Average price fallback
avg_price = df_filtered['price'].mean() if len(df_filtered) > 0 else 0
next_pred = avg_price

FORECAST_DIR = os.path.join(os.path.dirname(__file__), "forecasting")

if os.path.exists(FORECAST_DIR) and not df_prices.empty:
    forecast_files = glob.glob(os.path.join(FORECAST_DIR, '*.csv'))
    
    # Forecast Options Logic
    forecast_options = {}
    for file in forecast_files:
        filename = os.path.basename(file)
        # Clean filename to make it readable: "Pasta_forecast.csv" -> "Pasta"
        name_part = filename.replace('_forecast.csv', '')
        display_name = name_part.replace('_', ' ').title()
        forecast_options[display_name] = filename

    sorted_display_names = sorted(forecast_options.keys())
    
    if sorted_display_names:
        selected_display_name = st.selectbox("Available Forecasts", sorted_display_names, label_visibility="collapsed")
        selected_filename = forecast_options[selected_display_name]
        forecast_path = os.path.join(FORECAST_DIR, selected_filename)
        
        try:
            df_future = pd.read_csv(forecast_path)
            # Normalize column names to lowercase to handle case variations
            df_future.columns = [str(c).lower() if isinstance(c, str) else c for c in df_future.columns]
            
            # Rename standard Prophet columns to our standard names
            rename_map = {'ds': 'date', 'yhat': 'predicted_price', 'yhat_lower': 'lower_bound', 'yhat_upper': 'upper_bound'}
            df_future = df_future.rename(columns={k:v for k,v in rename_map.items() if k in df_future.columns})
            
            required_columns = ['date', 'predicted_price']
            if all(col in df_future.columns for col in required_columns):
                df_future['date'] = pd.to_datetime(df_future['date'])
                next_pred = df_future['predicted_price'].iloc[0]
                
                # KPIs matching Laptop Layout
                kpi_f1, kpi_f3 = st.columns(2)
                with kpi_f1: 
                    st.metric("Next Month Prediction", f"{next_pred:,.0f} DZD")
                with kpi_f3: 
                    st.metric("Forecast Horizon", f"{len(df_future)} Months")

                # Plotting matching Laptop Layout
                fig_f = go.Figure()
                # 1. Forecast Line
                fig_f.add_trace(go.Scatter(
                    x=df_future['date'], 
                    y=df_future['predicted_price'], 
                    mode='lines+markers', 
                    name='Forecast',
                    line=dict(color='#2196F3') # Blue like Laptop
                ))
                 
                fig_f.update_layout(
                    template='plotly_dark', 
                    height=400, 
                    xaxis_title="Date", 
                    yaxis_title="Price (DZD)",
                    hovermode="x unified"
                )
                st.plotly_chart(fig_f, use_container_width=True)
            else:
                st.error("Forecast file missing required columns.")
        except Exception as e:
            st.error(f"Error loading forecast: {e}")
    else:
        st.warning("No forecast files found in the folder.")
else:
    st.info("⚠️ Forecast folder not found.")

st.markdown("---")

# ==========================================
# 6. AI INSIGHTS
# ==========================================
st.markdown("## 6. AI-Powered Market Insights")
st.caption("Generative analysis of current food security and pricing trends.")

top_commodity = df_filtered['commodity'].value_counts().index[0] if not df_filtered.empty else "Unknown"
trend_desc = "Rising" if trend_change_pct > 0 else ("Falling" if trend_change_pct < 0 else "Stable")

context = f"""
- Market Segment: {selected_commodities[0] if selected_commodities else "Mixed"}
- Current Average Price: {current_mean:,.0f} DZD
- Price Trend Direction: {trend_desc}
- Volatility Level: {"High" if volatility > 0.5 else "Low"}
- Most Active Commodity: {top_commodity}
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
                    {'role': 'system', 'content': 'You are an agricultural economist. Provide 3-5 concise bullet points using the data provided.'},
                    {'role': 'user', 'content': f"Analyze this food market data:\n{context}"}
                ],
                'temperature': 0.7,
                'max_tokens': 500
            }, timeout=30
        )
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        return None
    except: return None

if st.button("🚀 Generate AI Insights", use_container_width=True, key="ai_food_btn"):
    with st.spinner("Analyzing market patterns..."):
        insights = get_ai_insights(context)
        if insights:
            st.markdown(f"### 🤖 Analysis\n{insights.replace('•', '<br>•')}", unsafe_allow_html=True)
        else:
            st.info("AI Service unavailable.")