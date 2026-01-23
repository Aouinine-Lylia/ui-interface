import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import requests
import glob
from hijri_converter import Hijri
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
from plotly.subplots import make_subplots
import pickle


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
# ============== Data Loading ==============
@st.cache_resource
def load_prediction_models():
    """Load both regressor and classifier models"""
    import os
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # Load the food regression model (it's a package with metadata)
        with open(os.path.join(script_dir, 'price_regressor.pkl'), 'rb') as f:
            model_package2 = pickle.load(f)
        
        # Extract the actual model and feature columns
        regressor = model_package2['model']
        feature_columns2 = model_package2['feature_columns']
        le_category2 = model_package2['label_encoders']['category']
        le_commodity2 = model_package2['label_encoders']['commodity']
        
        # Load the food classification model
        with open(os.path.join(script_dir, 'price_classifier_improved.pkl'), 'rb') as f:
            model_package = pickle.load(f)
        
        # Extract the actual model and feature columns
        classifier = model_package['model']
        feature_cols = model_package['feature_columns']
        le_category = model_package['label_encoders']['category']
        le_commodity = model_package['label_encoders']['commodity']
        class_map = model_package['class_map']
        
        # Return everything needed for prediction
        return {
            'regressor': regressor,
            'classifier': classifier,
            'regressor_features': feature_columns2,
            'classifier_features': feature_cols,
            'regressor_encoders': {'category': le_category2, 'commodity': le_commodity2},
            'classifier_encoders': {'category': le_category, 'commodity': le_commodity},
            'class_map': class_map,
            'regressor_package': model_package2,
            'classifier_package': model_package
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

def haversine_distance(lat, lon, lat2=36.7538, lon2=3.0588):
    """Calculate distance in km from Algiers"""
    R = 6371
    lat, lon, lat2, lon2 = map(np.radians, [lat, lon, lat2, lon2])
    dlat = lat2 - lat
    dlon = lon2 - lon
    a = np.sin(dlat/2)**2 + np.cos(lat) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def create_regression_features(commodity, category, wilaya, date_obj, price_input=None, models_dict=None):
    """Create features for regression prediction matching your backend"""
    lat = ALGERIAN_WILAYAS_COORDS[wilaya]["lat"]
    lon = ALGERIAN_WILAYAS_COORDS[wilaya]["lon"]
    
    # Get historical data for feature engineering (simplified)
    # In a real app, you'd load your historical dataframe
    df = load_food_data()  # Use your existing data loading function
    
    # Filter data for the commodity
    commodity_hist = df[df['commodity'] == commodity]
    if len(commodity_hist) == 0:
        commodity_hist = df[df['category'] == category]
    
    # Basic feature engineering
    features = {}
    
    # Date features
    features['year'] = date_obj.year
    features['month'] = date_obj.month
    features['quarter'] = (date_obj.month - 1) // 3 + 1
    features['day_of_week'] = date_obj.dayofweek
    features['is_weekend'] = 1 if date_obj.dayofweek >= 5 else 0
    
    # Season
    if date_obj.month in [12, 1, 2]:
        features['season'] = 0
    elif date_obj.month in [3, 4, 5]:
        features['season'] = 1
    elif date_obj.month in [6, 7, 8]:
        features['season'] = 2
    else:
        features['season'] = 3
    
    features['is_harvest_season'] = 1 if 3 <= date_obj.month <= 11 else 0
    
    # Location features
    features['distance_from_capital'] = haversine_distance(lat, lon)
    features['is_coastal'] = 1 if lat > 35.5 else 0
    
    # Product type features
    perishable_kw = ['Potato', 'Tomato', 'Onion', 'Egg', 'Milk', 'Meat', 'Fish', 'Chicken']
    imported_kw = ['Wheat', 'Rice', 'Sugar', 'Oil', 'Tea', 'Coffee', 'Pasta']
    staple_kw = ['Bread', 'Rice', 'Potato', 'Wheat flour', 'Pasta', 'Oil', 'Sugar', 'Milk']
    
    features['is_perishable'] = 1 if any(kw in commodity for kw in perishable_kw) else 0
    features['is_imported'] = 1 if any(kw in commodity for kw in imported_kw) else 0
    features['is_staple'] = 1 if any(kw in commodity for kw in staple_kw) else 0
    
    # Encode categorical variables using loaded encoders
    if models_dict:
        try:
            features['category_encoded'] = models_dict['regressor_encoders']['category'].transform([category])[0]
        except:
            features['category_encoded'] = 0
        
        try:
            features['commodity_encoded'] = models_dict['regressor_encoders']['commodity'].transform([commodity])[0]
        except:
            features['commodity_encoded'] = 0
    
    # Historical stats
    if len(commodity_hist) > 0:
        features['hist_mean'] = commodity_hist['price'].mean()
        features['hist_std'] = commodity_hist['price'].std() if len(commodity_hist) > 1 else commodity_hist['price'].mean() * 0.2
        features['hist_min'] = commodity_hist['price'].min()
        features['hist_max'] = commodity_hist['price'].max()
        features['hist_median'] = commodity_hist['price'].median()
    else:
        features['hist_mean'] = 500
        features['hist_std'] = 100
        features['hist_min'] = 0
        features['hist_max'] = 1000
        features['hist_median'] = 500
    
    # Additional features needed by your model
    features['is_ramadan'] = 0
    features['is_eid_fitr'] = 0
    features['is_eid_adha'] = 0
    features['is_islamic_holiday'] = 0
    
    # Add dummy lag features (in a real app, these would come from actual data)
    features['reg_price_lag1'] = features['hist_mean']
    features['reg_price_lag2'] = features['hist_mean']
    features['reg_price_lag3'] = features['hist_mean']
    features['reg_price_change_3'] = 0
    features['reg_price_change_7'] = 0
    features['reg_rolling_mean_7'] = features['hist_mean']
    features['reg_rolling_mean_30'] = features['hist_mean']
    features['reg_volatility'] = (features['hist_std'] / features['hist_mean']) if features['hist_mean'] > 0 else 0
    features['reg_price_vs_hist'] = 0
    features['days_since_first_obs'] = 1000  # dummy value
    
    # Add any missing features
    if models_dict and 'regressor_features' in models_dict:
        for col in models_dict['regressor_features']:
            if col not in features:
                features[col] = 0
    
    return features

def predict_price_local(commodity, category, wilaya, date_str, price_input=None):
    """
    Predict price using local pkl models - MATCHING API LOGIC
    
    Args:
        commodity: Commodity name
        category: Category name
        wilaya: Wilaya name
        date_str: Date string in format YYYY-MM-DD
        price_input: Price per kg for classification (if None, uses regression result)
    
    Returns:
        dict with price, classification, confidence, probabilities
    """
    models_dict = load_prediction_models()
    
    if models_dict is None:
        return None
    
    date_obj = pd.to_datetime(date_str)
    lat = ALGERIAN_WILAYAS_COORDS[wilaya]["lat"]
    lon = ALGERIAN_WILAYAS_COORDS[wilaya]["lon"]
    
    # Load historical data
    df = load_food_data()
    
    # === STEP 1: PRICE REGRESSION (Matching API) ===
    try:
        # Get historical data BEFORE prediction date
        commodity_hist_reg = df[
            (df['commodity'] == commodity) & 
            (df['date'] < date_obj)
        ].sort_values('date')
        
        if len(commodity_hist_reg) == 0:
            category_hist_reg = df[(df['category'] == category) & (df['date'] < date_obj)]
            if len(category_hist_reg) > 0:
                hist_data = category_hist_reg['price']
            else:
                hist_data = df[df['date'] < date_obj]['price']
        else:
            hist_data = commodity_hist_reg['price']
        
        if len(hist_data) == 0:
            st.error("No historical data available before prediction date")
            return None
        
        # Calculate regression features (matching API)
        reg_hist_mean = hist_data.mean()
        reg_hist_std = hist_data.std() if len(hist_data) > 1 else hist_data.mean() * 0.2
        reg_hist_min = hist_data.min()
        reg_hist_max = hist_data.max()
        reg_hist_median = hist_data.median()
        
        reg_price_lag1 = hist_data.iloc[-1] if len(hist_data) >= 1 else reg_hist_mean
        reg_price_lag2 = hist_data.iloc[-2] if len(hist_data) >= 2 else reg_hist_mean
        reg_price_lag3 = hist_data.iloc[-3] if len(hist_data) >= 3 else reg_hist_mean
        
        reg_price_change_3 = (reg_price_lag1 - hist_data.iloc[-4]) if len(hist_data) >= 4 else 0
        reg_price_change_7 = (reg_price_lag1 - hist_data.iloc[-8]) if len(hist_data) >= 8 else 0
        
        reg_rolling_mean_7 = hist_data.tail(7).mean()
        reg_rolling_mean_30 = hist_data.tail(30).mean()
        
        days_since_first_reg = (date_obj - commodity_hist_reg['date'].min()).days if len(commodity_hist_reg) > 0 else 0
        
        # Build regression features
        perishable_kw = ['Potato', 'Tomato', 'Onion', 'Egg', 'Milk', 'Meat', 'Fish', 'Chicken']
        imported_kw = ['Wheat', 'Rice', 'Sugar', 'Oil', 'Tea', 'Coffee', 'Pasta']
        staple_kw = ['Bread', 'Rice', 'Potato', 'Wheat flour', 'Pasta', 'Oil', 'Sugar', 'Milk']
        
        reg_features = {
            'year': date_obj.year,
            'month': date_obj.month,
            'quarter': (date_obj.month - 1) // 3 + 1,
            'day_of_week': date_obj.dayofweek,
            'is_weekend': 1 if date_obj.dayofweek >= 5 else 0,
            'season': 0 if date_obj.month in [12, 1, 2] else (1 if date_obj.month in [3, 4, 5] else (2 if date_obj.month in [6, 7, 8] else 3)),
            'is_harvest_season': 1 if 3 <= date_obj.month <= 11 else 0,
            'is_ramadan': 0,
            'is_eid_fitr': 0,
            'is_eid_adha': 0,
            'is_islamic_holiday': 0,
            'distance_from_capital': haversine_distance(lat, lon),
            'is_coastal': 1 if lat > 35.5 else 0,
            'is_perishable': 1 if any(kw in commodity for kw in perishable_kw) else 0,
            'is_imported': 1 if any(kw in commodity for kw in imported_kw) else 0,
            'is_staple': 1 if any(kw in commodity for kw in staple_kw) else 0,
            'reg_hist_mean': reg_hist_mean,
            'reg_hist_std': reg_hist_std,
            'reg_hist_min': reg_hist_min,
            'reg_hist_max': reg_hist_max,
            'reg_hist_median': reg_hist_median,
            'reg_price_lag1': reg_price_lag1,
            'reg_price_lag2': reg_price_lag2,
            'reg_price_lag3': reg_price_lag3,
            'reg_price_change_3': reg_price_change_3,
            'reg_price_change_7': reg_price_change_7,
            'reg_rolling_mean_7': reg_rolling_mean_7,
            'reg_rolling_mean_30': reg_rolling_mean_30,
            'reg_volatility': (reg_hist_std / reg_hist_mean) if reg_hist_mean > 0 else 0,
            'reg_price_vs_hist': (reg_price_lag1 - reg_hist_mean) / (reg_hist_std + 0.01),
            'days_since_first_obs': days_since_first_reg
        }
        
        # Encode categorical
        try:
            reg_features['category_encoded'] = models_dict['regressor_encoders']['category'].transform([category])[0]
        except:
            reg_features['category_encoded'] = 0
        
        try:
            reg_features['commodity_encoded'] = models_dict['regressor_encoders']['commodity'].transform([commodity])[0]
        except:
            reg_features['commodity_encoded'] = 0
        
        # Create DataFrame
        X_reg = pd.DataFrame([reg_features])
        
        # Add missing columns
        for col in models_dict['regressor_features']:
            if col not in X_reg.columns:
                X_reg[col] = 0
        
        X_reg = X_reg[models_dict['regressor_features']].fillna(0)
        
        # Predict
        predicted_price = float(models_dict['regressor'].predict(X_reg)[0])
        model_error = models_dict.get('regressor_package', {}).get('performance', {}).get('test_mae', 50)
        
    except Exception as e:
        st.error(f"Regression failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None
    
    # === STEP 2: CLASSIFICATION (Matching API) ===
    # Use provided price or predicted price
    price_for_classification = price_input if price_input else predicted_price
    
    try:
        # Get historical data for classification
        commodity_hist = df[df['commodity'] == commodity]
        if len(commodity_hist) == 0:
            commodity_hist = df[df['category'] == category]
        
        if len(commodity_hist) > 0:
            hist_mean = commodity_hist['price'].mean()
            hist_std = commodity_hist['price'].std() if len(commodity_hist) > 1 else hist_mean * 0.2
            hist_min = commodity_hist['price'].min()
            hist_max = commodity_hist['price'].max()
        else:
            hist_mean = 500
            hist_std = 100
            hist_min = 0
            hist_max = 1000
        
        # Build classification features (matching API - no 'price' in features!)
        class_features = {
            'year': date_obj.year,
            'month': date_obj.month,
            'quarter': (date_obj.month - 1) // 3 + 1,
            'day_of_week': date_obj.dayofweek,
            'is_weekend': 1 if date_obj.dayofweek >= 5 else 0,
            'season': 0 if date_obj.month in [12, 1, 2] else (1 if date_obj.month in [3, 4, 5] else (2 if date_obj.month in [6, 7, 8] else 3)),
            'is_harvest_season': 1 if 3 <= date_obj.month <= 11 else 0,
            'is_ramadan': 0,
            'is_eid_fitr': 0,
            'is_eid_adha': 0,
            'is_islamic_holiday': 0,
            'distance_from_capital': haversine_distance(lat, lon),
            'is_coastal': 1 if lat > 35.5 else 0,
            'is_perishable': 1 if any(kw in commodity for kw in perishable_kw) else 0,
            'is_imported': 1 if any(kw in commodity for kw in imported_kw) else 0,
            'is_staple': 1 if any(kw in commodity for kw in staple_kw) else 0,
            'hist_mean': hist_mean,
            'hist_std': hist_std,
            'hist_min': hist_min,
            'hist_max': hist_max,
            'hist_volatility': (hist_std / hist_mean) if hist_mean > 0 else 0,
            'days_since_first_obs': days_since_first_reg,
            'price_trend': 0
        }
        
        if hist_max > hist_min:
            class_features['price_position_in_range'] = (price_for_classification - hist_min) / (hist_max - hist_min)
        else:
            class_features['price_position_in_range'] = 0.5
        
        # Encode categorical
        try:
            class_features['category_encoded'] = models_dict['classifier_encoders']['category'].transform([category])[0]
        except:
            class_features['category_encoded'] = 0
        
        try:
            class_features['commodity_encoded'] = models_dict['classifier_encoders']['commodity'].transform([commodity])[0]
        except:
            class_features['commodity_encoded'] = 0
        
        # Create DataFrame
        X_class = pd.DataFrame([class_features])
        
        # Add missing columns
        for col in models_dict['classifier_features']:
            if col not in X_class.columns:
                X_class[col] = 0
        
        X_class = X_class[models_dict['classifier_features']].fillna(0)
        
        # Predict
        classification = models_dict['classifier'].predict(X_class)[0]
        probabilities = models_dict['classifier'].predict_proba(X_class)[0]
        
        class_labels = models_dict.get('class_map', {0: "CHEAP", 1: "NORMAL", 2: "EXPENSIVE"})
        classification_label = class_labels.get(int(classification), "UNKNOWN")
        
        return {
            "price": round(predicted_price, 2),
            "classification": classification_label,
            "confidence": float(probabilities[int(classification)]),
            "probabilities": {
                "CHEAP": float(probabilities[0]),
                "NORMAL": float(probabilities[1]),
                "EXPENSIVE": float(probabilities[2])
            },
            "confidence_interval": {
                "lower": round(max(0, predicted_price - model_error), 2),
                "upper": round(predicted_price + model_error, 2)
            },
            "historical_context": {
                "mean": round(hist_mean, 2),
                "std": round(hist_std, 2),
                "z_score": round((price_for_classification - hist_mean) / hist_std, 2) if hist_std > 0 else 0
            }
        }
        
    except Exception as e:
        st.error(f"Classification failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

    
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

def hijri_to_gregorian(year, month, day):
    return Hijri(year, month, day).to_gregorian()

def generate_food_events(start_year=2015, end_year=2025):
    events = []

    for h_year in range(1436, 1447):  # Covers 2015–2025
        # Ramadan
        ramadan_start = hijri_to_gregorian(h_year, 9, 1)
        ramadan_end = ramadan_start + relativedelta(days=29)
        events.append(("Ramadan", ramadan_start, ramadan_end, "religious"))

        # Eid Fitr
        eid_fitr = hijri_to_gregorian(h_year, 10, 1)
        events.append(("Eid al-Fitr", eid_fitr, eid_fitr + relativedelta(days=3), "religious"))

        # Eid Adha
        eid_adha = hijri_to_gregorian(h_year, 12, 10)
        events.append(("Eid al-Adha", eid_adha, eid_adha + relativedelta(days=4), "religious"))

        # Islamic New Year
        muharram = hijri_to_gregorian(h_year + 1, 1, 1)
        events.append(("Islamic New Year", muharram, muharram + relativedelta(days=1), "religious"))

        # Mawlid
        mawlid = hijri_to_gregorian(h_year, 3, 12)
        events.append(("Mawlid", mawlid, mawlid + relativedelta(days=1), "religious"))

    # Yennayer (fixed)
    for year in range(start_year, end_year + 1):
        events.append(("Yennayer", date(year, 1, 12), date(year, 1, 12), "cultural"))

    # Seasons (meteorological)
    for year in range(start_year, end_year + 1):
        events += [
            ("Winter", date(year, 12, 1), date(year + 1, 2, 28), "season"),
            ("Spring", date(year, 3, 1), date(year, 5, 31), "season"),
            ("Summer", date(year, 6, 1), date(year, 8, 31), "season"),
            ("Autumn", date(year, 9, 1), date(year, 11, 30), "season"),
        ]

    # COVID
    events.append(("COVID-19", date(2020, 3, 1), date(2021, 12, 31), "covid"))

    return events


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


# ======================
# COLORS
# ======================
EVENT_MARKER_COLORS = {
    "religious": "#FF9800",      # Orange
    "cultural": "#4CAF50",       # Green
    "institutional": "#9C27B0",  # Purple
    "covid": "#F44336",           # Red (only for background)
    "season": "#607D8B"         # Blue Grey (not used for markers)
}

EVENT_NAMES_DISPLAY = {
    "religious": "🕌 Religious Events (Ramadan, Eid, etc.)",
    "cultural": "🎊 Cultural Events (Yennayer)",
    "institutional": "🏛️ Institutional Events (School, BAC, etc.)",
    "covid": "🦠 COVID-19 Period",
    "season": "🌦️ Seasons"
}

# ====================== SEASONAL PRICE ANALYSIS WITH EVENT MARKERS ====================== 
def plot_seasonal_prices_with_events(df, date_col, price_col, events, title="Seasonal Price Analysis"):
    """
    Create seasonal price trend plot with event markers
    
    Features:
    - Monthly median price line
    - Colored dots for each event occurrence
    - COVID period highlighted in background
    - Legend explaining each event type
    - Year-by-year analysis from 2015-2025
    """
    
    # Prepare monthly median data
    df_sorted = df.sort_values(date_col).copy()
    df_sorted['year'] = df_sorted[date_col].dt.year
    df_sorted['month'] = df_sorted[date_col].dt.to_period('M').dt.to_timestamp()
    
    monthly_median = df_sorted.groupby('month')[price_col].median().reset_index()
    monthly_median.columns = ['date', 'median_price']
    
    # Create figure
    fig = go.Figure()
    
    # Add COVID-19 background shading first (so it's behind everything)
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
    
    # Add median price line (main trend)
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
        # Skip seasons and covid (covid is only background)
        if category in ["covid"]:
            continue
        
        start_dt = _convert_to_datetime(start)
        end_dt = _convert_to_datetime(end)
        
        # Check if event falls within data range
        if start_dt < monthly_median['date'].min() or start_dt > monthly_median['date'].max():
            continue
        
        # Find closest month to event start date
        closest_idx = (monthly_median['date'] - start_dt).abs().idxmin()
        event_date = monthly_median.loc[closest_idx, 'date']
        event_price = monthly_median.loc[closest_idx, 'median_price']
        
        # Store event markers by category
        if category not in event_markers_by_category:
            event_markers_by_category[category] = {
                'dates': [],
                'prices': [],
                'names': []
            }
        
        event_markers_by_category[category]['dates'].append(event_date)
        event_markers_by_category[category]['prices'].append(event_price)
        event_markers_by_category[category]['names'].append(name)
    
    # Add event markers as scatter traces (one per category for legend)
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
    
    # Add COVID-19 to legend (even though it's just background)
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
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, color='white'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title="Timeline (2015-2025)",
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

# ====================== HELPER FUNCTION ====================== 
def _convert_to_datetime(dt):
    """Convert various date types to datetime"""
    if isinstance(dt, pd.Timestamp):
        return dt.to_pydatetime()
    elif isinstance(dt, date) and not isinstance(dt, datetime):
        return datetime.combine(dt, datetime.min.time())
    return dt

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
                    "miscellaneous food", "non-food", "oil and fats",
                    "pulses and nuts", "vegetables and fruits"
                ], index=7)
                wilaya = st.selectbox("Wilaya", options=ALGERIAN_WILAYAS_LIST, index=15)
            
            with col_b:
                predict_date = st.date_input("Date", value=datetime.now())
                
                price_input = st.number_input(
                        "Price per kg (DZD)", 
                        min_value=0.0, 
                        value=200.0, 
                        step=10.0,
                        help="Enter the price to classify"
                    )
            
            submit = st.form_submit_button("🚀 Analyze", use_container_width=True)
            
            if submit:
                with st.spinner("Analyzing market data..."):
                    result = predict_price_local(
                        commodity, 
                        category, 
                        wilaya, 
                        predict_date.strftime("%Y-%m-%d"),
                        price_input
                    )
                    if result:
                        st.markdown(f"### Predicted Price: **{result['price']:.2f} DZD/kg**")
                        if result['classification']=="CHEAP":
                            st.success(f"### Price Classification: **{result['classification']}**")
                        elif result['classification']=="NORMAL":
                            st.info(f"### Price Classification: **{result['classification']}**")
                        else:
                            st.warning(f"### Price Classification: **{result['classification']}**")
                        st.markdown(f"### Confidence: **{result['confidence']*100:.1f}%**")
                    
                    else:
                        st.error("❌ Unable to generate prediction. Please try Classification Only mode with manual price input.")

st.markdown("---")

# ============== Filters ==============
st.markdown("### 🔍 Data Filters")
f1, f2, f3 = st.columns(3)

with f1:
    all_commodities = sorted(df_prices['commodity'].unique().tolist())
    # Set default to most common commodity
    default_commodity = df_prices['commodity'].value_counts().index[0] if not df_prices.empty else None
    selected_commodities = st.multiselect(
        "Commodity", 
        all_commodities, 
        default=[default_commodity] if default_commodity else None
    )

with f2:
    all_regions = sorted(df_prices['admin1'].unique().tolist()) if 'admin1' in df_prices.columns else []
    selected_regions = st.multiselect("Region (Wilaya)", all_regions, default=[])

with f3:
    st.markdown(" ", unsafe_allow_html=True)
    if st.button("Apply Filters"):
        st.rerun()

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
    st.markdown(create_kpi_card_with_explanation(
        title="Median Price (DZD/kg)",
        value=f"{current_median:,.0f}",
        subtitle=f"MoM Change: {trend_change_pct:+.1f}%",
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
        value=f"{total_records:,}",
        subtitle="Data Points Analyzed",
        explanation="Total number of price observations in the selected timeframe and filters. More data = more reliable insights.",
        trend_color="#42A5F5"
    ), unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# 2. PRICE EVOLUTION
# ==========================================
st.markdown("## 2. Price Evolution Over Time")
st.caption("Historical analysis of pricing trends.")

# Generate events
food_events = generate_food_events(start_year=2015, end_year=2025)

st.markdown("### 📊 Seasonal Price Trends & Event Analysis")
st.caption("Analyze how religious, cultural, and institutional events impact food prices over time.")

# Event Legend
st.markdown("---")

# Main seasonal plot with all years
fig_seasonal = plot_seasonal_prices_with_events(
    df_filtered, 
    'date', 
    'price', 
    food_events,
    "Food Price Trends with Event Markers"
)
st.plotly_chart(fig_seasonal, use_container_width=True)

st.markdown("---")

# ==========================================
# 3. MARKET SEGMENTATION
# ==========================================
st.markdown("## 3. Market Segmentation Analysis")
st.caption("Comparing value across commodity categories and product popularity.")

# Use ORIGINAL df_prices (not df_filtered) for this section
c1, c2 = st.columns(2)

with c1:
    st.markdown("### Price by Category")
    if 'category' in df_prices.columns:
        cat_price = df_prices.groupby('category')['price'].median().sort_values(ascending=False)
        
        # Create color scale from red (expensive) to green (cheap)
        max_price = cat_price.max()
        min_price = cat_price.min()
        colors = [f'rgb({int(255 * (p - min_price) / (max_price - min_price))}, '
                  f'{int(255 * (1 - (p - min_price) / (max_price - min_price)))}, 0)'
                  for p in cat_price.values]
        
        fig_cat = px.bar(
            x=cat_price.values, 
            y=cat_price.index, 
            orientation='h', 
            title="Median Price by Category (All Data)"
        )
        fig_cat.update_traces(marker_color=colors)
        fig_cat.update_layout(
            template='plotly_dark', 
            height=400, 
            xaxis_title="Price (DZD)", 
            yaxis_title=""
        )
        st.plotly_chart(fig_cat, use_container_width=True)
    else:
        st.info("Category data not available.")

with c2:
    st.markdown("### Top Commodities by Volume")
    comm_counts = df_prices['commodity'].value_counts().head(10)
    fig_comm = px.bar(
        x=comm_counts.values, 
        y=comm_counts.index, 
        orientation='h', 
        title="Most Tracked Commodities (All Data)"
    )
    fig_comm.update_layout(
        template='plotly_dark', 
        height=400, 
        xaxis_title="Listings", 
        yaxis_title=""
    )
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
        
        # Create red-to-green color scale (green = cheap, red = expensive)
        max_price = wilaya_avg_price.max()
        min_price = wilaya_avg_price.min()
        
        # Normalize to 0-1 range
        normalized = (wilaya_avg_price.values - min_price) / (max_price - min_price)
        
        # Create RGB colors: green (0,255,0) to red (255,0,0)
        colors = [f'rgb({int(255 * n)}, {int(255 * (1 - n))}, 0)' for n in normalized]
        
        fig_wilaya = go.Figure(data=[
            go.Bar(
                x=wilaya_avg_price.values,
                y=wilaya_avg_price.index,
                orientation='h',
                marker=dict(
                    color=colors,
                    line=dict(color='rgba(255,255,255,0.3)', width=1)
                ),
                text=[f'{p:.0f} DZD' for p in wilaya_avg_price.values],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Avg Price: %{x:.2f} DZD<extra></extra>'
            )
        ])
        
        fig_wilaya.update_layout(
            title="Average Price by Wilaya (DZD)",
            template='plotly_dark', 
            height=500, 
            xaxis_title="Avg Price (DZD)", 
            yaxis_title="",
            showlegend=False
        )
        st.plotly_chart(fig_wilaya, use_container_width=True)

with m2:
    st.markdown("### Geographic Distribution")
    st.caption("Visual heatmap of price points across the country.")
    
    if 'latitude' in df_filtered.columns and 'longitude' in df_filtered.columns:
        map_data = df_filtered.groupby(['admin1', 'market', 'latitude', 'longitude'])['price'].mean().reset_index()
        map_data = map_data.dropna(subset=['latitude', 'longitude'])
        
        if not map_data.empty:
            # Use same red-green color scale
            fig_map = px.scatter_mapbox(
                map_data,
                lat="latitude",
                lon="longitude",
                size="price",
                color="price",
                hover_name="market",
                hover_data={"admin1": True, "price": ":.2f"},
                color_continuous_scale=['#00FF00', '#FFFF00', '#FF0000'],  # Green -> Yellow -> Red
                size_max=25,
                zoom=5,
                center={"lat": 34.0, "lon": 3.0}
            )
            fig_map.update_layout(
                mapbox_style="carto-darkmatter", 
                margin={"r":0,"t":0,"l":0,"b":0}, 
                height=500,
                coloraxis_colorbar=dict(
                    title="Price (DZD)",
                    tickformat=".0f"
                )
            )
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

# Forecasting KPIs
forecast_summary = "N/A"
forecast_next_pred = None
forecast_horizon = None
if 'df_future' in locals() and df_future is not None and not df_future.empty:
    forecast_next_pred = df_future['predicted_price'].iloc[0]
    forecast_horizon = len(df_future)
    forecast_summary = f"Next Month Prediction: {forecast_next_pred:,.0f} DZD, Forecast Horizon: {forecast_horizon} months"
elif 'next_pred' in locals() and next_pred is not None:
    forecast_next_pred = next_pred
    forecast_summary = f"Next Month Prediction: {forecast_next_pred:,.0f} DZD"

context = f"""
-- Market Segment: {selected_commodities[0] if selected_commodities else "Mixed"}
-- Current Average Price: {current_mean:,.0f} DZD
-- Median Price: {current_median:,.0f} DZD
-- Price Trend Direction: {trend_desc} ({trend_indicator}, {trend_change_pct:+.2f}%)
-- Volatility Level: {"High" if volatility > 0.5 else "Low"} (CV: {volatility:.2f})
-- Current filtered Commodities: {', '.join(selected_commodities) if selected_commodities else "All"}
-- Total Records: {total_records}
-- Top 3 Categories by Median Price: {', '.join([f'{cat}: {val:,.0f} DZD' for cat, val in df_prices.groupby('category')['price'].median().sort_values(ascending=False).head(3).items()])}
-- Top 3 Commodities by Volume: {', '.join([f'{comm}: {val}' for comm, val in df_prices['commodity'].value_counts().head(3).items()])}
-- Cheapest Wilaya (avg): {df_filtered.groupby('admin1')['price'].mean().idxmin()} ({df_filtered.groupby('admin1')['price'].mean().min():.0f} DZD)
-- Most Expensive Wilaya (avg): {df_filtered.groupby('admin1')['price'].mean().idxmax()} ({df_filtered.groupby('admin1')['price'].mean().max():.0f} DZD)
-- Forecasting Summary: {forecast_summary}
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