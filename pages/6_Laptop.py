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
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Laptops | DZA PriceSight",
    page_icon="💻",
    layout="wide"
)

# ============== CONSTANTS & DATA MAPS ==============
# Coordinate mapping for Geographic Distribution (Matching Food Dashboard)
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

EVENT_MARKER_COLORS = {
    "institutional": "#9C27B0",  # Purple
    "covid": "#F44336"  ,         # Red
    "season": "#607D8B"         # Blue Grey (not used for markers)
}

EVENT_NAMES_DISPLAY = {
    "institutional": "🏛️ Institutional Events (School, BAC, etc.)",
    "covid": "🦠 COVID-19 Period", 
    "season": "🌦️ Seasons"
}

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
    div[data-testid="stForm"] button[kind="primary"],
    div[data-testid="column"] button[kind="primary"] {
        background-color: #ffeb3b !important;
        color: #000000 !important;
        font-weight: 800 !important;
        text-transform: uppercase;
        border: none;
        box-shadow: 0 0 10px rgba(255, 235, 59, 0.4);
    }
    
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

# ============== Helper Functions ==============
def create_kpi_card_with_explanation(title, value, subtitle, explanation, trend_color="#4CAF50"):
    """Create KPI card with tooltip explanation"""
    return f"""
    <div style='background: linear-gradient(135deg, rgba(30,30,30,0.95) 0%, rgba(50,50,50,0.95) 100%); 
                padding: 20px; border-radius: 12px; border-left: 4px solid {trend_color}; 
                box-shadow: 0 4px 6px rgba(0,0,0,0.3); position: relative;'>
        <div style='display: flex; justify-content: space-between; align-items: start;'>
            <div style='color: #aaa; font-size: 0.9em; margin-bottom: 8px;'>{title}</div>
            <div style='cursor: help; color: #888; font-size: 0.85em;' title='{explanation}'>ℹ️</div>
        </div>
        <div style='color: white; font-size: 2em; font-weight: bold; margin: 10px 0;'>{value}</div>
        <div style='color: {trend_color}; font-size: 0.85em;'>{subtitle}</div>
        <div style='color: #666; font-size: 0.75em; margin-top: 8px; font-style: italic;'>{explanation}</div>
    </div>
    """

def generate_laptop_events(start_year=2018, end_year=2025):
    events = []

    for year in range(start_year, end_year + 1):
        events += [
            ("School Start", date(year, 9, 1), date(year, 9, 15), "institutional"),
            ("BAC Results", date(year, 7, 20), date(year, 7, 25), "institutional"),
            ("New Year", date(year, 1, 1), date(year, 1, 5), "institutional"),
        ]

        # Black Friday (last Friday of November)
        nov = pd.date_range(f"{year}-11-01", f"{year}-11-30", freq="D")
        black_friday = nov[nov.weekday == 4][-1]
        events.append(("Black Friday", black_friday.date(), black_friday.date(), "institutional"))

        # Seasons
        events += [
            ("Winter", date(year, 12, 1), date(year + 1, 2, 28), "season"),
            ("Spring", date(year, 3, 1), date(year, 5, 31), "season"),
            ("Summer", date(year, 6, 1), date(year, 8, 31), "season"),
            ("Autumn", date(year, 9, 1), date(year, 11, 30), "season"),
        ]

    # COVID
    events.append(("COVID-19", date(2020, 3, 1), date(2021, 12, 31), "covid"))

    return events

def filter_significant_events(events, data_start_date, data_end_date, max_events_per_category=5):
    """Filter events to show only the most significant ones"""
    
    if isinstance(data_start_date, pd.Timestamp):
        data_start_date = data_start_date.to_pydatetime()
    if isinstance(data_end_date, pd.Timestamp):
        data_end_date = data_end_date.to_pydatetime()
    
    events_by_category = {}
    for event in events:
        name, start, end, category = event
        
        if isinstance(start, date) and not isinstance(start, datetime):
            start = datetime.combine(start, datetime.min.time())
        if isinstance(end, date) and not isinstance(end, datetime):
            end = datetime.combine(end, datetime.min.time())
        
        if end < data_start_date or start > data_end_date:
            continue
        
        if category not in events_by_category:
            events_by_category[category] = []
        
        events_by_category[category].append((name, start, end, category))
    
    filtered_events = []
    for category in ["covid"]:
        if category in events_by_category:
            filtered_events.extend(events_by_category[category])
    
    for category in ["institutional"]:
        if category in events_by_category:
            category_events = sorted(
                events_by_category[category], 
                key=lambda x: x[1], 
                reverse=True
            )[:max_events_per_category]
            filtered_events.extend(category_events)
    
    return filtered_events

def _convert_to_datetime(dt):
    """Convert various date types to datetime"""
    if isinstance(dt, pd.Timestamp):
        return dt.to_pydatetime()
    elif isinstance(dt, date) and not isinstance(dt, datetime):
        return datetime.combine(dt, datetime.min.time())
    return dt

def plot_seasonal_prices_with_events(df, date_col, price_col, events, title="Seasonal Price Analysis"):
    """Create seasonal price trend plot with event markers"""
    
    df_sorted = df.sort_values(date_col).copy()
    df_sorted['year'] = df_sorted[date_col].dt.year
    df_sorted['month'] = df_sorted[date_col].dt.to_period('M').dt.to_timestamp()
    
    monthly_median = df_sorted.groupby('month')[price_col].median().reset_index()
    monthly_median.columns = ['date', 'median_price']
    
    fig = go.Figure()
    
    # Add COVID-19 background
    covid_events = [e for e in events if e[3] == "covid"]
    for name, start, end, category in covid_events:
        start_dt = _convert_to_datetime(start)
        end_dt = _convert_to_datetime(end)
        
        fig.add_vrect(
            x0=start_dt,
            x1=end_dt,
            fillcolor="rgba(244, 67, 54, 0.1)",
            layer="below",
            line_width=0,
        )
    
    # Add median price line
    fig.add_trace(go.Scatter(
        x=monthly_median['date'],
        y=monthly_median['median_price'],
        mode='lines',
        name='📊 Monthly Median Price',
        line=dict(color='#2196F3', width=3),
        hovertemplate='<b>Date:</b> %{x|%b %Y}<br><b>Median Price:</b> %{y:.2f} DZD<extra></extra>'
    ))
    
    # Process events and add markers
    event_markers_by_category = {}
    
    for name, start, end, category in events:
        if category in ["season", "covid"]:
            continue
        
        start_dt = _convert_to_datetime(start)
        end_dt = _convert_to_datetime(end)
        
        if start_dt < monthly_median['date'].min() or start_dt > monthly_median['date'].max():
            continue
        
        closest_idx = (monthly_median['date'] - start_dt).abs().idxmin()
        event_date = monthly_median.loc[closest_idx, 'date']
        event_price = monthly_median.loc[closest_idx, 'median_price']
        
        if category not in event_markers_by_category:
            event_markers_by_category[category] = {
                'dates': [],
                'prices': [],
                'names': []
            }
        
        event_markers_by_category[category]['dates'].append(event_date)
        event_markers_by_category[category]['prices'].append(event_price)
        event_markers_by_category[category]['names'].append(name)
    
    # Add event markers
    for category, data in event_markers_by_category.items():
        if not data['dates']:
            continue
        
        fig.add_trace(go.Scatter(
            x=data['dates'],
            y=data['prices'],
            mode='markers',
            name=EVENT_NAMES_DISPLAY[category],
            marker=dict(
                color=EVENT_MARKER_COLORS[category],
                size=12,
                symbol='circle',
                line=dict(color='white', width=2)
            ),
            text=data['names'],
            hovertemplate='<b>%{text}</b><br>Date: %{x|%b %d, %Y}<br>Price: %{y:.2f} DZD<extra></extra>'
        ))
    
    # Add COVID-19 to legend
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        name=EVENT_NAMES_DISPLAY['covid'],
        marker=dict(
            color='rgba(244, 67, 54, 0.3)',
            size=15,
            symbol='square'
        ),
        showlegend=True
    ))
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, color='white'),
            x=0.5,
            xanchor="center"
        ),
        xaxis=dict(
            title="Timeline",
            gridcolor='rgba(128, 128, 128, 0.2)',
            showgrid=True,
            rangeslider=dict(visible=True, thickness=0.05)
        ),
        yaxis=dict(
            title="Median Price (DZD)",
            gridcolor='rgba(128, 128, 128, 0.2)',
            showgrid=True
        ),
        template="plotly_dark",
        height=600,
        hovermode='closest',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01,
            bgcolor="rgba(0,0,0,0.7)",
            bordercolor="rgba(255,255,255,0.3)",
            borderwidth=1,
            font=dict(size=11)
        ),
        margin=dict(r=250)
    )
    
    return fig
# ============== Load & Preprocess Laptop Data ==============
@st.cache_data
def load_laptops():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, 'kaggle_laptop_data.csv')
        
        if not os.path.exists(csv_path):
            csv_path = 'kaggle_laptop_data.csv'

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
                 'city': np.random.choice(['Algiers', 'Oran', 'Constantine', 'Annaba', 'Blida'], 100),
                 'wilaya': np.random.choice(['Alger', 'Oran', 'Constantine', 'Annaba', 'Blida'], 100)
             }
             df = pd.DataFrame(data)
        else:
            df = pd.read_csv(csv_path)
        
        # Parse Date
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
                return pd.to_datetime(clean_str, errors='coerce')
            except:
                return pd.NaT
        
        if 'created_at' in df.columns:
            df['created_at'] = df['created_at'].apply(parse_date)
            df['date'] = df['created_at']
        else:
            df['date'] = pd.to_datetime('today')
            
        df = df.dropna(subset=['date', 'price_preview'])
        
        # DATA CLEANING
        if 'RAM_SIZE' in df.columns:
            df['RAM_SIZE'] = df['RAM_SIZE'].astype(str).str.extract(r'(\d+)').astype(float).fillna(8.0)
        else:
            df['RAM_SIZE'] = 8.0

        for col in ['SSD_SIZE', 'HDD_SIZE']:
            if col not in df.columns: df[col] = 0
            df[col] = df[col].astype(str).str.extract(r'(\d+)').astype(float).fillna(0)
        
        df['Total_Storage'] = df.get('SSD_SIZE', 0) + df.get('HDD_SIZE', 0)
        df['Total_Storage'] = df['Total_Storage'].replace(0, 256)

        if 'SCREEN_SIZE' in df.columns:
            df['SCREEN_SIZE'] = pd.to_numeric(df['SCREEN_SIZE'], errors='coerce').fillna(15.6)
        else:
            df['SCREEN_SIZE'] = 15.6

        df['price_preview'] = pd.to_numeric(df['price_preview'], errors='coerce')
        df = df.dropna(subset=['price_preview'])
        
        # Condition
        condition_mapping = {
            'JAMAIS UTILIS': 'JAMAIS UTILISÉ', 'BON TAT': 'BON ÉTAT',
            'MOYEN': 'MOYEN', 'NEUF': 'NEUF', 'TRES BON': 'TRÈS BON', 'ETAT NEUF': 'ÉTAT NEUF'
        }
        df['spec_Etat'] = df['spec_Etat'].fillna('Unknown')
        df['spec_Etat'] = df['spec_Etat'].str.upper().map(condition_mapping).fillna(df['spec_Etat'])
        
        # CPU Logic
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
        
        # GPU
        df['DEDICATED_GPU'] = df['DEDICATED_GPU'].fillna('INTEGRATED')
        df['Has_GPU'] = df['DEDICATED_GPU'].apply(lambda x: 0 if pd.isna(x) or str(x).upper() == 'INTEGRATED' or x == '' else 1)
        
        # Mock columns
        if 'discount_flag' not in df.columns: df['discount_flag'] = 0 
        if 'stock_status' not in df.columns: df['stock_status'] = 'In Stock'
        df['model_name'] = df['model_name'].fillna('Unknown').str.upper().str.strip()
        
        # Ensure city and wilaya columns exist
        if 'city' not in df.columns:
            df['city'] = 'Unknown'
        if 'wilaya' not in df.columns:
            df['wilaya'] = df['city']
            
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
            
            # Classification Input
            price_input = st.number_input(
                "Current Listing Price (DZD)", 
                min_value=0.0, 
                value=0.0, 
                step=1000.0,
                help="Enter the price found in a listing to classify it against the predicted market value."
            )
            
            model_name_options = [
                "ALIENWARE", "ASPIRE", "BLADE", "COMPAQ", "DYNABOOK", "ELITEBOOK", "ENVY", "GALAXY", "GF", "IDEAPAD", "IMAC", "INSPIRON", "KATANA", "LATITUDE", "LEGION", "MAC", "MACBOOK", "NITRO", "OMEN", "OPTIPLEX", "PAVILION", "PRECISION", "PREDATOR", "PROBOOK", "ROG", "SPECTRE", "SPIN", "STEALTH", "STRIX", "SURFACE", "SWIFT", "SWORD", "THINKBOOK", "THINKPAD", "TRANSFORMER", "TRAVELMATE", "TUF", "VECTOR", "VICTUS", "VIVOBOOK", "VOSTRO", "XPS", "YOGA", "ZBOOK", "ZENBOOK"
            ]
            model_name = st.selectbox("Model Name (Brand)", model_name_options, index=11)
            
            if st.form_submit_button("Calculate Estimated Price (DZD)"):
                try:
                    days_since_posted = (datetime.now().date() - listing_date).days
                    cpu_type_dict = {f"CPU_Type_{t}": 1 if cpu_type == t else 0 for t in cpu_type_options}
                    condition_dict = {f"Condition_{c}": 1 if condition == c else 0 for c in condition_options}
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
                    
                    # Classification Logic
                    if price_input > 0:
                        price_clean = price_input
                        diff_pct = ((price_clean - prediction) / prediction) * 100
                        
                        def get_label(diff):
                            if diff < -20: return 'Great Deal'   # 20% Cheaper than expected
                            if diff > 20: return 'Overpriced'    # 20% More expensive than expected
                            return 'Fair Price'
                        
                        label = get_label(diff_pct)
                        
                        if label == 'Great Deal':
                            st.success(f"### Classification: **{label}**")
                            st.caption(f"The listing is {abs(diff_pct):.1f}% cheaper than the estimated market value.")
                        elif label == 'Overpriced':
                            st.error(f"### Classification: **{label}**")
                            st.caption(f"The listing is {diff_pct:.1f}% more expensive than the estimated market value.")
                        else:
                            st.info(f"### Classification: **{label}**")
                            st.caption(f"The listing price is within the normal market range ({abs(diff_pct):.1f}% variance).")
                            
                    else:
                        st.caption("💡 Enter a listing price above to see if it's a Great Deal, Fair Price, or Overpriced.")
                        
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    fallback = df_laptops['price_preview'].median()
                    st.warning(f"Using median market price: {fallback:,.0f} DZD")

st.markdown("---")

# ============== Filters ==============
st.markdown("### 🔍 Market Filters")
f1, f2, f3, f4 = st.columns(4)
with f1:
    brands = ['All'] + sorted([str(x) for x in df_laptops['model_name'].unique().tolist() if x != 'Unknown'])
    # Set THINKPAD as default if present, else fallback to 'All'
    thinkpad_index = next((i for i, b in enumerate(brands) if b.upper() == 'THINKPAD'), 0)
    st.session_state.brand_filter = st.selectbox("Brand", brands, index=thinkpad_index)
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
if st.session_state.brand_filter != 'All': 
    df = df[df['model_name'] == st.session_state.brand_filter]
if st.session_state.gpu_filter: 
    df = df[df['DEDICATED_GPU'] != 'INTEGRATED']
if st.session_state.gaming_filter: 
    df = df[df['DEDICATED_GPU'].str.contains('RTX|GTX', case=False, na=False)]

if len(df) > 0:
    if df['price_preview'].nunique() >= 3:
        df['segment'] = pd.qcut(df['price_preview'], 3, labels=['Budget', 'Mid-Range', 'Premium'], duplicates='drop')
    else: 
        df['segment'] = 'Mid-Range'
else: 
    df['segment'] = 'Unknown'

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

if len(monthly_data) >= 2:
    last_month_val = monthly_data.iloc[-1]['median_price']
    prev_month_val = monthly_data.iloc[-2]['median_price']
    trend_change_pct = ((last_month_val - prev_month_val) / prev_month_val) * 100 if prev_month_val > 0 else 0
else:
    trend_change_pct = 0

trend_indicator = "↑" if trend_change_pct > 0 else ("↓" if trend_change_pct < 0 else "→")
trend_color = "#EF5350" if trend_change_pct > 2 else ("#66BB6A" if trend_change_pct < -2 else "#FFA726")

volatility = (df['price_preview'].std() / df['price_preview'].mean()) if len(df) > 0 and df['price_preview'].mean() > 0 else 0
total_listings = len(df)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(create_kpi_card_with_explanation(
        title="Median Price (DZD)",
        value=f"{current_median:,.0f}",
        subtitle=f"MoM: {trend_change_pct:+.1f}%",
        explanation="The middle value of all prices - half are higher, half are lower. More reliable than average for skewed data.",
        trend_color=trend_color
    ), unsafe_allow_html=True)

with kpi2:
    st.markdown(create_kpi_card_with_explanation(
        title="Price Trend",
        value=trend_indicator,
        subtitle="Monthly Direction",
        explanation="Shows whether prices are rising (↑), falling (↓), or stable (→) compared to last month.",
        trend_color=trend_color
    ), unsafe_allow_html=True)

with kpi3:
    volatility_status = "High Variance" if volatility > 0.5 else "Stable"
    volatility_color = "#EF5350" if volatility > 0.5 else "#66BB6A"
    st.markdown(create_kpi_card_with_explanation(
        title="Market Volatility",
        value=f"{volatility:.2f}",
        subtitle=volatility_status,
        explanation="Coefficient of variation: measures price stability. Lower is more stable. >0.5 indicates high price fluctuation.",
        trend_color=volatility_color
    ), unsafe_allow_html=True)

with kpi4:
    st.markdown(create_kpi_card_with_explanation(
        title="Market Volume",
        value=f"{total_listings:,}",
        subtitle="Active Listings",
        explanation="Total number of laptop listings in the selected timeframe and filters. More data = more reliable insights.",
        trend_color="#42A5F5"
    ), unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# 2. PRICE EVOLUTION
# ==========================================
st.markdown("## 2. Price Evolution Over Time")
st.caption("Historical analysis of pricing trends and market distribution.")

# ==========================================
# 2.5 SEASONAL ANALYSIS
# ==========================================
st.markdown("### Seasonal Price Trends & Event Analysis")
st.caption("Analyze how institutional events (school, BAC, Black Friday) and COVID-19 impact laptop prices over time.")

# Generate laptop events
laptop_events = generate_laptop_events(start_year=2018, end_year=2025)

fig_seasonal = plot_seasonal_prices_with_events(
    df, 
    'date', 
    'price_preview', 
    laptop_events,
    "Laptop Price Trends with Event Markers"
)
st.plotly_chart(fig_seasonal, use_container_width=True)

st.markdown("---")

st.markdown("### Price Distribution by Segment")
fig_box = px.box(
    df, 
    x='segment', 
    y='price_preview', 
    color='segment', 
    color_discrete_map={'Budget':'#4CAF50', 'Mid-Range':'#FFC107', 'Premium':'#F44336'}, 
    points='outliers'
)
fig_box.update_layout(
    template='plotly_dark', 
    height=400, 
    xaxis_title="Market Segment", 
    yaxis_title="Price (DZD)", 
    showlegend=False
)
st.plotly_chart(fig_box, use_container_width=True)

st.markdown("---")



# ==========================================
# 3. MARKET SEGMENTATION (NO FILTER - USE ORIGINAL DATA)
# ==========================================
st.markdown("## 3. Market Segmentation Analysis")
st.caption("Comparing value across hardware tiers and brand popularity (all data, no filters applied).")

c1, c2 = st.columns(2)
with c1:
    st.markdown("### Price by CPU Tier")
    cpu_tier_price = df_laptops.groupby('cpu_tier')['price_preview'].median().sort_values(ascending=False)
    fig_cpu = px.bar(
        x=cpu_tier_price.values, 
        y=cpu_tier_price.index, 
        orientation='h', 
        title="Median Price by Processor Tier (All Data)"
    )
    fig_cpu.update_layout(
        template='plotly_dark', 
        height=350, 
        xaxis_title="Price (DZD)", 
        yaxis_title=""
    )
    st.plotly_chart(fig_cpu, use_container_width=True)

with c2:
    st.markdown("### Value Analysis: Price per GB of RAM")
    if df_laptops['RAM_SIZE'].sum() > 0:
        df_laptops['price_per_gb_ram'] = df_laptops['price_preview'] / df_laptops['RAM_SIZE'].replace(0, 1)
        value_data = df_laptops.groupby('model_name')['price_per_gb_ram'].mean().sort_values(ascending=True).head(10)
        fig_value = px.bar(
            x=value_data.values, 
            y=value_data.index, 
            orientation='h', 
            title="Best Value Laptops (Lower = Better, All Data)", 
            labels={'x': 'DZD per GB RAM', 'y': 'Brand'}
        )
        fig_value.update_layout(template='plotly_dark', height=400)
        fig_value.update_traces(marker_color='#66BB6A')
        st.plotly_chart(fig_value, use_container_width=True)

st.markdown("---")

# ==========================================
# 4. REGIONAL MARKET INSIGHTS
# ==========================================
st.markdown("## 4. Regional Market Insights")
st.caption("Geographic pricing analysis across Algerian wilayas.")

m1, m2, m3 = st.columns(3)
with m1:
    used_count = df_laptops[df_laptops['spec_Etat'] != 'JAMAIS UTILISÉ'].shape[0]
    used_pct = (used_count / len(df_laptops)) * 100 if len(df_laptops) > 0 else 0
    st.metric("Used/Reconditioned Market Share", f"{used_pct:.1f}%")

with m2:
    gpu_count = df_laptops[df_laptops['Has_GPU'] == 1].shape[0]
    gpu_pct = (gpu_count / len(df_laptops)) * 100 if len(df_laptops) > 0 else 0
    st.metric("Laptops with Dedicated GPU", f"{gpu_pct:.1f}%")

with m3:
    avg_storage = df_laptops['Total_Storage'].mean() if 'Total_Storage' in df_laptops.columns else 0
    st.metric("Average Storage Capacity", f"{avg_storage:.0f} GB")

# Regional Analysis Bar Chart
if 'wilaya' in df_laptops.columns and df_laptops['wilaya'].notna().sum() > 0:
    st.markdown("### Wilayas by Average Price (Cheapest to Most Expensive)")
    st.caption("Identify regions with the lowest average laptop costs to find better deals.")
    
    # Create combined city-wilaya column for display
    df_laptops['display_location'] = df_laptops.apply(
        lambda x: f"{x['city']} ({x['wilaya']})" if pd.notna(x['wilaya']) else x['city'], 
        axis=1
    )
    
    # Calculate average price per location
    location_avg_price = df_laptops.groupby('display_location')['price_preview'].mean().sort_values(ascending=True)
    
    # Create red-to-green color scale
    max_price = location_avg_price.max()
    min_price = location_avg_price.min()
    normalized = (location_avg_price.values - min_price) / (max_price - min_price)
    colors = [f'rgb({int(255 * n)}, {int(255 * (1 - n))}, 0)' for n in normalized]
    
    fig_location = go.Figure(data=[
        go.Bar(
            x=location_avg_price.values,
            y=location_avg_price.index,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgba(255,255,255,0.3)', width=1)
            ),
            text=[f'{p:.0f} DZD' for p in location_avg_price.values],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Avg Price: %{x:.2f} DZD<extra></extra>'
        )
    ])
    
    fig_location.update_layout(
        title="Average Price by City (Wilaya)",
        template='plotly_dark', 
        height=500, 
        xaxis_title="Avg Price (DZD)", 
        yaxis_title="",
        showlegend=False
    )
    st.plotly_chart(fig_location, use_container_width=True)

# ============================
# Geographic Distribution Map
# ============================
st.markdown("### Geographic Distribution")
st.caption("Visual heatmap of price points across the country (Median Price per Wilaya).")

# 1. Aggregate data by Wilaya (not individual cities) to get a median price per region
# We group by 'wilaya' and calculate the median price and listing count
map_data = df_laptops.groupby('wilaya')['price_preview'].agg(['median', 'count']).reset_index()
map_data.columns = ['wilaya', 'median_price', 'listings_count']

# 2. Create a normalized lookup dictionary for coordinates (case-insensitive)
NORMALIZED_COORDS = {k.lower(): v for k, v in ALGERIAN_WILAYAS_COORDS.items()}

def get_coords(wilaya):
    """
    Safely get coordinates for the Wilaya.
    Always returns a dict with 'lat' and 'lon' keys.
    """
    if pd.isna(wilaya):
        return {'lat': np.nan, 'lon': np.nan}
    
    wilaya_clean = str(wilaya).strip().lower()
    
    # Attempt direct match
    if wilaya_clean in NORMALIZED_COORDS:
        return NORMALIZED_COORDS[wilaya_clean]
    
    # Handle slight variations (optional)
    if 'algier' in wilaya_clean: return ALGERIAN_WILAYAS_COORDS['Alger']
    if 'constantine' in wilaya_clean: return ALGERIAN_WILAYAS_COORDS['Constantine']
        
    return {'lat': np.nan, 'lon': np.nan}

# 3. Map coordinates
coords_data = map_data['wilaya'].apply(get_coords)
coords_df = pd.DataFrame(coords_data.tolist())

# 4. Merge coordinates into map_data
map_data = pd.concat([map_data.reset_index(drop=True), coords_df], axis=1)

# 5. Drop rows where coordinates couldn't be found
map_data = map_data.dropna(subset=['lat', 'lon'])

if not map_data.empty:
    fig_map = px.scatter_mapbox(
        map_data,
        lat="lat",
        lon="lon",
        size="median_price",  # Size based on median price
        color="median_price", # Color based on median price
        hover_name="wilaya",  # Show Wilaya name on hover
        hover_data={"median_price": ":.2f", "listings_count": True}, # Show median price and count
        color_continuous_scale=['#00FF00', '#FFFF00', '#FF0000'],  # Green -> Yellow -> Red
        size_max=25,
        zoom=5,
        center={"lat": 34.0, "lon": 3.0},
        title="Median Laptop Price by Wilaya"
    )
    fig_map.update_layout(
        mapbox_style="carto-darkmatter", 
        margin={"r":0,"t":30,"l":0,"b":0}, 
        height=500,
        coloraxis_colorbar=dict(
            title="Median Price (DZD)",
            tickformat=".0f"
        )
    )
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.warning("⚠️ No geographic coordinates could be matched.")
# ==========================================
# 5. FORECASTING
# ==========================================
st.markdown("## 5. Market Forecasting")
st.caption("Projected price movements based on historical data.")

st.info("ℹ️ **Note:** These forecasts estimate expected market price movement, not individual laptop prices.")

avg_price = df['price_preview'].mean() if len(df) > 0 else 0
next_pred = avg_price

FORECAST_DIR = os.path.join(os.path.dirname(__file__), "forecasting_laptops")

if os.path.exists(FORECAST_DIR) and not df_laptops.empty:
    forecast_files = glob.glob(os.path.join(FORECAST_DIR, '*.csv'))
    
    def get_readable_condition(condition_key):
        key_norm = condition_key.upper().replace('_', ' ')
        mapping = {
            "BON TAT": "Good State", 
            "JAMAIS UTILIS": "Never Used", 
            "MOYEN": "Fair", 
            "NEUF": "New", 
            "TRES BON": "Very Good", 
            "ETAT NEUF": "Brand New"
        }
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
    
    if sorted_display_names:
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
                with kpi_f1: 
                    st.metric("Next Month Prediction", f"{next_pred:,.0f} DZD")
                with kpi_f3: 
                    st.metric("Forecast Horizon", f"{len(df_future)} Months")

                fig_f = go.Figure()
                fig_f.add_trace(go.Scatter(
                    x=df_future['date'], 
                    y=df_future['predicted_price'], 
                    mode='lines+markers', 
                    name='Forecast', 
                    line=dict(color='#2196F3')
                ))
                if 'lower_bound' in df_future.columns:
                    fig_f.add_trace(go.Scatter(
                        x=df_future['date'], 
                        y=df_future['upper_bound'], 
                        mode='lines', 
                        line_color='rgba(0,0,0,0)', 
                        showlegend=False
                    ))
                    fig_f.add_trace(go.Scatter(
                        x=df_future['date'], 
                        y=df_future['lower_bound'], 
                        mode='lines', 
                        line_color='rgba(0,0,0,0)', 
                        fill='tonexty', 
                        fillcolor='rgba(33, 150, 243, 0.2)', 
                        name='95% CI'
                    ))
                
                fig_f.update_layout(
                    template='plotly_dark', 
                    height=400, 
                    xaxis_title="Date", 
                    yaxis_title="Price (DZD)"
                )
                st.plotly_chart(fig_f, use_container_width=True)
            else:
                st.error("Forecast file missing required columns.")
        except Exception as e:
            st.error(f"Error loading forecast: {e}")
    else:
        st.warning("No forecast files found.")
else:
    st.info("⚠️ Forecast folder not found.")

st.markdown("---")

# ==========================================
# 6. AI INSIGHTS (ENHANCED)
# ==========================================
st.markdown("## 6. AI-Powered Market Insights")
st.caption("Generative analysis of current market conditions.")

top_brand = df['model_name'].value_counts().index[0] if len(df) > 0 else "Unknown"
price_trend = "Rising" if trend_indicator == "↑" else ("Falling" if trend_indicator == "↓" else "Stable")

# Enhanced context with calculated results
top_cpu_tiers = df_laptops.groupby('cpu_tier')['price_preview'].median().sort_values(ascending=False).head(3)
top_brands_volume = df_laptops['model_name'].value_counts().head(3)
cheapest_wilaya = ""
most_expensive_wilaya = ""

if 'display_location' in df_laptops.columns:
    location_prices = df_laptops.groupby('display_location')['price_preview'].mean()
    cheapest_wilaya = f"{location_prices.idxmin()} ({location_prices.min():.0f} DZD)"
    most_expensive_wilaya = f"{location_prices.idxmax()} ({location_prices.max():.0f} DZD)"

forecast_summary = "N/A"
if 'df_future' in locals() and df_future is not None and not df_future.empty:
    forecast_next_pred = df_future['predicted_price'].iloc[0]
    forecast_horizon = len(df_future)
    forecast_summary = f"Next Month: {forecast_next_pred:,.0f} DZD, Horizon: {forecast_horizon} months"

context = f"""
-- Market Segment: Laptop Market Analysis
-- Current Median Price: {current_median:,.0f} DZD (≈ {current_median/220:.0f} USD)
-- Current Mean Price: {current_mean:,.0f} DZD
-- Price Trend Direction: {price_trend} ({trend_indicator}, {trend_change_pct:+.2f}%)
-- Market Volatility: {"High" if volatility > 0.5 else "Low"} (CV: {volatility:.2f})
-- Total Active Listings: {total_listings:,}
-- Top Brand by Volume: {top_brand}
-- Used/Reconditioned Market Share: {used_pct:.1f}%
-- Laptops with Dedicated GPU: {gpu_pct:.1f}%
-- Average Storage Capacity: {avg_storage:.0f} GB
-- Top 3 CPU Tiers by Price: {', '.join([f'{tier}: {price:,.0f} DZD' for tier, price in top_cpu_tiers.items()])}
-- Top 3 Brands by Volume: {', '.join([f'{brand}: {count}' for brand, count in top_brands_volume.items()])}
-- Cheapest Location: {cheapest_wilaya}
-- Most Expensive Location: {most_expensive_wilaya}
-- Forecast Summary: {forecast_summary}
-- Current Filters Applied: Brand={st.session_state.brand_filter}, GPU Only={st.session_state.gpu_filter}, Gaming Only={st.session_state.gaming_filter}
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
                    {'role': 'system', 'content': 'You are a technology market analyst specializing in consumer electronics. Provide 3-5 concise, actionable bullet points using the data provided. Focus on trends, opportunities, and market insights.'},
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

if st.button("🚀 Generate AI Insights", use_container_width=True, key="ai_btn"):
    with st.spinner("Analyzing market patterns..."):
        insights = get_ai_insights(context)
        if insights:
            st.markdown(f"### 🤖 Analysis\n{insights.replace('•', '<br>•')}", unsafe_allow_html=True)
        else:
            st.info("AI Service unavailable. Showing basic stats.")
            st.write(f"Market is currently trending {price_trend} with a median price of {current_median:,.0f} DZD.")

# Footer
st.markdown("<div style='text-align:center; color:#666; margin-top:40px;'>DZA PriceSight | Retail Intelligence Dashboard</div>", unsafe_allow_html=True)