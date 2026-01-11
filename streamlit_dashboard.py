import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
import os
import json

warnings.filterwarnings('ignore')

# ============== Color Scheme Constants ==============
COLOR_PRIMARY = '#1f77b4'  # Blue
COLOR_SECONDARY = '#2ca02c'  # Green
COLOR_EXPENSIVE = '#d62728'  # Red
COLOR_CHEAP = '#2ca02c'  # Green
COLOR_NEUTRAL = '#ffbb00'  # Yellow/Orange
COLOR_TREND = '#9467bd'  # Purple
COLORSCALE_HEATMAP = 'RdYlGn_r'  # Red (high) to Green (low)
COLORSCALE_SEQUENTIAL = 'Viridis'  # Standard sequential
COLORSCALE_PRICE = 'YlOrRd'  # Yellow to Red for prices

# Page configuration
st.set_page_config(
    page_title="Food Price Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .header-title {
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .category-header {
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        color: white;
        padding: 10px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .coming-soon {
        background-color: #ffeaa7;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: #2d3436;
    }
    </style>
""", unsafe_allow_html=True)

# Get script and parent directories
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

# ============== SIDEBAR - Dataset Selection ==============
st.sidebar.title("📊 Analytics Hub")
st.sidebar.markdown("---")

# Dataset selector
st.sidebar.subheader("🗂️ Select Dataset")
dataset_options = {
    "🍎 Food Prices": "food",
    "💻 Laptops (Coming Soon)": "laptops",
    "🚗 Cars (Coming Soon)": "cars"
}

selected_dataset = st.sidebar.radio(
    "Choose a dataset to analyze:",
    list(dataset_options.keys()),
    index=0
)

dataset_key = dataset_options[selected_dataset]

st.sidebar.markdown("---")

# ============== Load Data Functions ==============
@st.cache_data
def load_food_data():
    # Price data
    df_prices = pd.read_csv(os.path.join(script_dir, 'wfp_food_prices_dza.csv'), comment='#')
    df_prices['date'] = pd.to_datetime(df_prices['date'])
    
    # Enhanced data (global)
    df_enhanced = pd.read_csv(os.path.join(parent_dir, 'prophet_data_enhanced.csv'))
    df_enhanced['ds'] = pd.to_datetime(df_enhanced['ds'])
    
    # Forecast data
    df_forecast = pd.read_csv(os.path.join(parent_dir, 'outputs/forecast_full.csv'))
    df_forecast['date'] = pd.to_datetime(df_forecast['date'])
    
    df_eval = pd.read_csv(os.path.join(parent_dir, 'outputs/evaluation_results.csv'))
    df_eval['date'] = pd.to_datetime(df_eval['date'])
    
    df_future = pd.read_csv(os.path.join(parent_dir, 'outputs/forecast_future.csv'))
    df_future['date'] = pd.to_datetime(df_future['date'])
    
    # Load model metadata
    model_metadata = None
    metadata_files = [f for f in os.listdir(os.path.join(parent_dir, 'models')) if f.endswith('_metadata.json')]
    if metadata_files:
        with open(os.path.join(parent_dir, 'models', metadata_files[0]), 'r') as f:
            model_metadata = json.load(f)
    
    # Load category data
    category_data = {}
    category_files = [f for f in os.listdir(parent_dir) if f.startswith('prophet_data_category_')]
    for f in category_files:
        category_name = f.replace('prophet_data_category_', '').replace('.csv', '').replace('_', ' ').replace('  ', ' & ')
        filepath = os.path.join(parent_dir, f)
        df_cat = pd.read_csv(filepath)
        df_cat['ds'] = pd.to_datetime(df_cat['ds'])
        category_data[category_name] = df_cat
    
    return df_prices, df_enhanced, df_forecast, df_eval, df_future, model_metadata, category_data

# ============== FOOD PRICES DASHBOARD ==============
if dataset_key == "food":
    # Load data
    df_prices, df_enhanced, df_forecast, df_eval, df_future, model_metadata, category_data = load_food_data()
    
    # Sidebar category selector
    st.sidebar.subheader("📂 View Mode")
    view_options = ["🌍 Global View"] + [f"📁 {cat.title()}" for cat in category_data.keys()]
    selected_view = st.sidebar.selectbox("Select View:", view_options)
    
    st.sidebar.markdown("---")
    
    # ============== SIDEBAR FILTERS (Global) ==============
    st.sidebar.subheader("🔍 Filters")
    
    # Quick filter presets
    st.sidebar.markdown("**Quick Filters:**")
    preset_col1, preset_col2 = st.sidebar.columns(2)
    
    today = df_prices['date'].max()
    if preset_col1.button("📅 Last 30 Days", use_container_width=True):
        start_date = (today - timedelta(days=30)).date()
        end_date = today.date()
    elif preset_col2.button("📅 Last 90 Days", use_container_width=True):
        start_date = (today - timedelta(days=90)).date()
        end_date = today.date()
    else:
        start_date = df_prices['date'].min().date()
        end_date = df_prices['date'].max().date()
    
    preset_col3, preset_col4 = st.sidebar.columns(2)
    if preset_col3.button("📅 Last 6 Months", use_container_width=True):
        start_date = (today - timedelta(days=180)).date()
        end_date = today.date()
    elif preset_col4.button("🌍 All Time", use_container_width=True):
        start_date = df_prices['date'].min().date()
        end_date = df_prices['date'].max().date()
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Custom Date Range:",
        value=(start_date, end_date),
        min_value=df_prices['date'].min().date(),
        max_value=df_prices['date'].max().date(),
        key='global_date_filter'
    )
    
    # Market filter
    markets = sorted(df_prices['market'].unique())
    selected_markets = st.sidebar.multiselect(
        "Select Markets:",
        markets,
        default=markets[:5] if len(markets) > 5 else markets,
        help="Choose specific markets to analyze"
    )
    
    # Commodity filter
    commodities = sorted(df_prices['commodity'].unique())
    selected_commodities = st.sidebar.multiselect(
        "Select Commodities:",
        commodities,
        default=commodities,
        help="Choose specific commodities to analyze"
    )
    
    st.sidebar.markdown("---")
    
    # ============== GLOBAL VIEW ==============
    if selected_view == "🌍 Global View":
        st.markdown('<div class="header-title">🍎 Food Price Analytics - Algeria</div>', unsafe_allow_html=True)
        
        # Apply filters
        mask = (
            (df_prices['date'] >= pd.Timestamp(date_range[0])) &
            (df_prices['date'] <= pd.Timestamp(date_range[1])) &
            (df_prices['market'].isin(selected_markets)) &
            (df_prices['commodity'].isin(selected_commodities))
        )
        filtered_data = df_prices[mask]
        
        # Global KPIs with Insights
        st.markdown("---")
        st.markdown("### 📊 Key Metrics Overview")
        
        # Add insight card
        st.info(""" 
        **📌 What are these metrics?** These KPIs provide a snapshot of the current food price situation in Algeria based on your selected filters. 
        Use them to quickly assess market conditions, price levels, and data coverage.
        """)
        
        kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
        
        avg_price_dzd = filtered_data['price'].mean()
        avg_price_usd = filtered_data['usdprice'].mean()
        
        with kpi1:
            st.metric("Avg Price (DZD)", f"{avg_price_dzd:.2f}", 
                     help="Average price across all selected commodities and markets in Algerian Dinar")
        with kpi2:
            st.metric("Avg Price (USD)", f"${avg_price_usd:.2f}",
                     help="Average price converted to US Dollars for international comparison")
        with kpi3:
            st.metric("Total Records", f"{len(filtered_data):,}",
                     help="Number of price observations in your filtered dataset")
        with kpi4:
            st.metric("Commodities", f"{filtered_data['commodity'].nunique()}",
                     help="Number of unique food items tracked in the selection")
        with kpi5:
            st.metric("Markets", f"{filtered_data['market'].nunique()}",
                     help="Number of unique markets/locations covered in the data")
        with kpi6:
            st.metric("Regions", f"{filtered_data['admin1'].nunique()}",
                     help="Number of provinces (Wilayas) represented in the selection")
        
        st.markdown("---")
        
        # ============== TABS ==============
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "💰 Price Analysis",
            "🛍️ Commodity Analysis",
            "🏪 Market Analysis",
            "🌍 Geographical Analysis",
            "⚙️ Data Quality",
            "🔮 Forecasts",
            "📉 Statistics"
        ])
        
        # ============== TAB 1: PRICE ANALYSIS ==============
        with tab1:
            st.subheader("💰 Price Analysis")
            
            # Price trend over time
            st.markdown("### 📈 Price Trend Over Time")
            
            trend_commodity = st.selectbox(
                "Select commodity for detailed trend:",
                options=filtered_data['commodity'].unique(),
                index=0,
                key='price_trend_commodity'
            )
            
            trend_data = filtered_data[filtered_data['commodity'] == trend_commodity].groupby('date').agg({
                'price': ['mean', 'std', 'min', 'max']
            }).reset_index()
            trend_data.columns = ['date', 'price_mean', 'price_std', 'price_min', 'price_max']
            trend_data = trend_data.sort_values('date')
            trend_data['price_std'] = trend_data['price_std'].fillna(0)
            
            fig_trend = go.Figure()
            
            # Mean line with consistent color
            fig_trend.add_trace(go.Scatter(
                x=trend_data['date'], y=trend_data['price_mean'],
                mode='lines+markers', name='Mean Price',
                line=dict(color=COLOR_PRIMARY, width=3),
                marker=dict(size=4)
            ))
            
            # Std dev band
            fig_trend.add_trace(go.Scatter(
                x=trend_data['date'].tolist() + trend_data['date'].tolist()[::-1],
                y=(trend_data['price_mean'] + trend_data['price_std']).tolist() + 
                  (trend_data['price_mean'] - trend_data['price_std']).tolist()[::-1],
                fill='toself', fillcolor='rgba(31, 119, 180, 0.2)',
                line=dict(color='rgba(255,255,255,0)'), name='±1 Std Dev', hoverinfo='skip'
            ))
            
            # Min/Max with consistent colors
            fig_trend.add_trace(go.Scatter(x=trend_data['date'], y=trend_data['price_max'],
                mode='lines', name='Max Price', line=dict(color=COLOR_EXPENSIVE, width=1, dash='dot')))
            fig_trend.add_trace(go.Scatter(x=trend_data['date'], y=trend_data['price_min'],
                mode='lines', name='Min Price', line=dict(color=COLOR_CHEAP, width=1, dash='dot')))
            
            # Add average line annotation
            overall_avg = trend_data['price_mean'].mean()
            fig_trend.add_hline(y=overall_avg, line_dash="dash", line_color="gray", 
                               annotation_text=f"Overall Avg: {overall_avg:.2f} DZD",
                               annotation_position="right")
            
            fig_trend.update_layout(
                title=f"Price Trend with Volatility - {trend_commodity}<br><sub>Shows mean, min, max, and standard deviation over time</sub>",
                xaxis_title="Date", yaxis_title="Price (DZD)",
                height=450, template='plotly_white', hovermode='x unified'
            )
            st.plotly_chart(fig_trend, use_container_width=True)
            
            st.markdown("---")
            
            # Price volatility
            st.markdown("### 📉 Price Volatility Analysis")
            
            st.info("""  
            **💡 Understanding Volatility:** The Coefficient of Variation (CV) measures price stability. 
            **Low CV** = Stable prices (green) 🟢 | **High CV** = Volatile prices (red) 🔴  
            Volatile commodities may indicate supply issues, seasonal patterns, or market speculation.
            """)
            
            volatility_data = filtered_data.groupby('commodity').agg({
                'price': ['std', 'mean', 'count']
            }).reset_index()
            volatility_data.columns = ['commodity', 'std_dev', 'mean_price', 'count']
            volatility_data['cv'] = (volatility_data['std_dev'] / volatility_data['mean_price'] * 100).round(2)
            
            col1, col2 = st.columns(2)
            
            with col1:
                vol_sorted = volatility_data.nsmallest(15, 'cv')
                fig_stable = go.Figure(data=[go.Bar(
                    x=vol_sorted['cv'], y=vol_sorted['commodity'], orientation='h',
                    marker=dict(color=vol_sorted['cv'], colorscale='Greens_r'),
                    text=vol_sorted['cv'].round(1), textposition='outside', texttemplate='%{text}%'
                )])
                fig_stable.update_layout(title="🟢 Most Stable Prices (Low CV %)<br><sub>Lower is better - indicates consistent pricing</sub>", 
                                        xaxis_title="CV (%)", height=400, template='plotly_white')
                st.plotly_chart(fig_stable, use_container_width=True)
            
            with col2:
                vol_sorted = volatility_data.nlargest(15, 'cv')
                fig_volatile = go.Figure(data=[go.Bar(
                    x=vol_sorted['cv'], y=vol_sorted['commodity'], orientation='h',
                    marker=dict(color=vol_sorted['cv'], colorscale='Reds'),
                    text=vol_sorted['cv'].round(1), textposition='outside', texttemplate='%{text}%'
                )])
                fig_volatile.update_layout(title="🔴 Most Volatile Prices (High CV %)<br><sub>Higher values indicate greater price fluctuation</sub>", 
                                          xaxis_title="CV (%)", height=400, template='plotly_white')
                st.plotly_chart(fig_volatile, use_container_width=True)
            
            st.markdown("---")
            
            # Min/Max/Median Summary
            st.markdown("### 📋 Price Summary by Commodity")
            
            price_summary = filtered_data.groupby('commodity').agg({
                'price': ['min', 'max', 'median', 'mean', 'std'],
                'usdprice': 'mean'
            }).round(2)
            price_summary.columns = ['Min (DZD)', 'Max (DZD)', 'Median (DZD)', 'Mean (DZD)', 'Std (DZD)', 'Avg (USD)']
            price_summary['Range'] = price_summary['Max (DZD)'] - price_summary['Min (DZD)']
            price_summary = price_summary.sort_values('Mean (DZD)', ascending=False)
            
            st.dataframe(price_summary, use_container_width=True, height=400)
        
        # ============== TAB 2: COMMODITY ANALYSIS ==============
        with tab2:
            st.subheader("🛍️ Commodity Analysis")
            
            # Most expensive vs cheapest
            st.markdown("### 💎 Most Expensive vs Cheapest Commodities")
            
            st.info("""  
            **💡 Price Ranking Insight:** This comparison helps identify which commodities are most/least affordable. 
            High prices may indicate scarcity or premium products. Low prices suggest abundance or staple goods.
            """)
            
            commodity_prices = filtered_data.groupby('commodity').agg({
                'price': 'mean', 'usdprice': 'mean'
            }).reset_index().round(2)
            commodity_prices.columns = ['Commodity', 'Avg Price (DZD)', 'Avg Price (USD)']
            commodity_prices = commodity_prices.sort_values('Avg Price (DZD)', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                top_10 = commodity_prices.head(10)
                fig = go.Figure(data=[go.Bar(
                    x=top_10['Avg Price (DZD)'], y=top_10['Commodity'], orientation='h',
                    marker=dict(color=top_10['Avg Price (DZD)'], colorscale=COLORSCALE_PRICE),
                    text=top_10['Avg Price (DZD)'].round(0), textposition='outside', texttemplate='%{text} DZD'
                )])
                fig.update_layout(title="🔴 Top 10 Most Expensive<br><sub>Premium or scarce commodities</sub>", 
                                 xaxis_title="Avg Price (DZD)", height=400, template='plotly_white', 
                                 yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                bottom_10 = commodity_prices.tail(10).sort_values('Avg Price (DZD)')
                fig = go.Figure(data=[go.Bar(
                    x=bottom_10['Avg Price (DZD)'], y=bottom_10['Commodity'], orientation='h',
                    marker=dict(color=bottom_10['Avg Price (DZD)'], colorscale='Greens'),
                    text=bottom_10['Avg Price (DZD)'].round(0), textposition='outside', texttemplate='%{text} DZD'
                )])
                fig.update_layout(title="🟢 Top 10 Cheapest<br><sub>Affordable staple goods</sub>", 
                                 xaxis_title="Avg Price (DZD)", height=400, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Commodity ranking
            st.markdown("### 🏆 Commodity Price Ranking")
            
            ranking_view = st.radio("Ranking by:", ["Category", "Market", "Overall"], horizontal=True, key='comm_ranking')
            
            if ranking_view == "Category":
                selected_cat = st.selectbox("Select Category:", filtered_data['category'].unique(), key='rank_cat')
                cat_data = filtered_data[filtered_data['category'] == selected_cat]
                ranking = cat_data.groupby('commodity')['price'].mean().sort_values(ascending=False).reset_index()
                ranking['Rank'] = range(1, len(ranking) + 1)
                ranking.columns = ['Commodity', 'Avg Price (DZD)', 'Rank']
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = go.Figure(data=[go.Bar(
                        x=ranking['Avg Price (DZD)'], y=ranking['Commodity'], orientation='h',
                        marker=dict(color=ranking['Rank'], colorscale='Viridis_r'),
                        text=ranking['Rank'], textposition='inside'
                    )])
                    fig.update_layout(title=f"Ranking in {selected_cat}", height=450, template='plotly_white', yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.dataframe(ranking[['Rank', 'Commodity', 'Avg Price (DZD)']].round(2), use_container_width=True, height=400, hide_index=True)
            
            elif ranking_view == "Market":
                selected_mkt = st.selectbox("Select Market:", filtered_data['market'].unique(), key='rank_mkt')
                mkt_data = filtered_data[filtered_data['market'] == selected_mkt]
                ranking = mkt_data.groupby('commodity')['price'].mean().sort_values(ascending=False).reset_index()
                ranking['Rank'] = range(1, len(ranking) + 1)
                ranking.columns = ['Commodity', 'Avg Price (DZD)', 'Rank']
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = go.Figure(data=[go.Bar(
                        x=ranking['Avg Price (DZD)'], y=ranking['Commodity'], orientation='h',
                        marker=dict(color=ranking['Rank'], colorscale='Plasma_r'),
                        text=ranking['Rank'], textposition='inside'
                    )])
                    fig.update_layout(title=f"Ranking in {selected_mkt}", height=450, template='plotly_white', yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.dataframe(ranking[['Rank', 'Commodity', 'Avg Price (DZD)']].round(2), use_container_width=True, height=400, hide_index=True)
            
            else:  # Overall
                ranking = commodity_prices.copy()
                ranking['Rank'] = range(1, len(ranking) + 1)
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = go.Figure(data=[go.Bar(
                        x=ranking['Avg Price (DZD)'], y=ranking['Commodity'], orientation='h',
                        marker=dict(color=ranking['Rank'], colorscale='Turbo_r'),
                        text=ranking['Rank'], textposition='inside'
                    )])
                    fig.update_layout(title="Overall Commodity Ranking", height=500, template='plotly_white', yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.dataframe(ranking[['Rank', 'Commodity', 'Avg Price (DZD)', 'Avg Price (USD)']].round(2), use_container_width=True, height=450, hide_index=True)
            
            st.markdown("---")
            
            # Category comparison
            st.markdown("### 📊 Price by Category")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                for cat in filtered_data['category'].unique():
                    fig.add_trace(go.Box(y=filtered_data[filtered_data['category'] == cat]['price'], name=cat))
                fig.update_layout(title="Price Distribution by Category", yaxis_title="Price (DZD)", height=450, template='plotly_white', showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                cat_avg = filtered_data.groupby('category')['price'].mean().sort_values(ascending=True)
                fig = go.Figure(data=[go.Bar(x=cat_avg.values, y=cat_avg.index, orientation='h',
                    marker=dict(color=cat_avg.values, colorscale='Viridis'))])
                fig.update_layout(title="Average Price by Category", xaxis_title="Price (DZD)", height=450, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
        
        # ============== TAB 3: MARKET ANALYSIS ==============
        with tab3:
            st.subheader("🏪 Market Analysis")
            
            # Number of commodities per market
            st.markdown("### 🛒 Commodities per Market")
            
            st.info("""  
            **💡 Market Diversity Insight:** Markets with more commodities offer greater food variety and may indicate 
            larger, more developed trading centers. Low diversity may suggest specialized or rural markets.
            """)
            
            comm_per_market = filtered_data.groupby('market')['commodity'].nunique().sort_values(ascending=False).reset_index()
            comm_per_market.columns = ['Market', 'N Commodities']
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure(data=[go.Bar(
                    x=comm_per_market['N Commodities'], y=comm_per_market['Market'], orientation='h',
                    marker=dict(color=comm_per_market['N Commodities'], colorscale='Viridis')
                )])
                fig.update_layout(title="Commodity Diversity by Market", xaxis_title="Number of Commodities", height=400, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                mcol1.metric("Total Markets", comm_per_market['Market'].nunique())
                mcol2.metric("Avg Commodities", f"{comm_per_market['N Commodities'].mean():.1f}")
                mcol3.metric("Max Commodities", comm_per_market['N Commodities'].max())
                mcol4.metric("Min Commodities", comm_per_market['N Commodities'].min())
                st.dataframe(comm_per_market, use_container_width=True, height=300, hide_index=True)
            
            st.markdown("---")
            
            # Market coverage
            st.markdown("### 🗺️ Market Coverage")
            
            col1, col2 = st.columns(2)
            
            with col1:
                markets_per_region = filtered_data.groupby('admin1')['market'].nunique().sort_values(ascending=False).reset_index()
                markets_per_region.columns = ['Wilaya', 'N Markets']
                
                fig = go.Figure(data=[go.Bar(
                    x=markets_per_region['N Markets'], y=markets_per_region['Wilaya'], orientation='h',
                    marker=dict(color=markets_per_region['N Markets'], colorscale='Teal')
                )])
                fig.update_layout(title="Markets per Wilaya", xaxis_title="Number of Markets", height=400, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                mcol1, mcol2, mcol3 = st.columns(3)
                mcol1.metric("Wilayas Covered", markets_per_region['Wilaya'].nunique())
                mcol2.metric("Total Markets", markets_per_region['N Markets'].sum())
                mcol3.metric("Avg Markets/Wilaya", f"{markets_per_region['N Markets'].mean():.1f}")
                st.dataframe(markets_per_region, use_container_width=True, height=300, hide_index=True)
            
            st.markdown("---")
            
            # Average price per market
            st.markdown("### 💵 Average Price per Market")
            
            st.info("""  
            **💡 Price Distribution Insight:** Markets above the average line (dashed) are more expensive - this could be due to 
            urban location, transportation costs, or higher demand. Below-average markets may offer better value.
            """)
            
            market_stats = filtered_data.groupby('market').agg({
                'price': ['mean', 'std', 'count'],
                'usdprice': 'mean'
            }).reset_index()
            market_stats.columns = ['Market', 'Avg Price (DZD)', 'Std Dev', 'Records', 'Avg Price (USD)']
            market_stats = market_stats.sort_values('Avg Price (DZD)', ascending=False)
            
            overall_avg = filtered_data['price'].mean()
            market_stats['Diff from Avg'] = market_stats['Avg Price (DZD)'] - overall_avg
            market_stats['Category'] = market_stats['Diff from Avg'].apply(
                lambda x: '🔴 Expensive' if x > overall_avg * 0.1 else ('🟢 Cheap' if x < -overall_avg * 0.1 else '🟡 Average')
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                colors = [COLOR_EXPENSIVE if x > overall_avg * 0.1 else (COLOR_CHEAP if x < -overall_avg * 0.1 else COLOR_NEUTRAL) for x in market_stats['Diff from Avg']]
                fig = go.Figure(data=[go.Bar(
                    x=market_stats['Avg Price (DZD)'], y=market_stats['Market'], orientation='h', marker_color=colors,
                    text=market_stats['Avg Price (DZD)'].round(0), textposition='outside', texttemplate='%{text} DZD'
                )])
                fig.add_vline(x=overall_avg, line_dash="dash", line_color="gray", 
                             annotation_text=f"Average: {overall_avg:.0f} DZD",
                             annotation_position="top")
                fig.update_layout(title="Average Price by Market<br><sub>Red=Expensive | Yellow=Average | Green=Cheap</sub>", 
                                 xaxis_title="Avg Price (DZD)", height=500, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("**🟢 Green:** <10% below average | **🟡 Yellow:** Within ±10% | **🔴 Red:** >10% above average")
            
            with col2:
                st.dataframe(market_stats[['Market', 'Avg Price (DZD)', 'Avg Price (USD)', 'Std Dev', 'Records', 'Category']].round(2), 
                            use_container_width=True, height=450, hide_index=True)
                
                qcol1, qcol2, qcol3 = st.columns(3)
                qcol1.metric("Expensive", (market_stats['Category'] == '🔴 Expensive').sum())
                qcol2.metric("Average", (market_stats['Category'] == '🟡 Average').sum())
                qcol3.metric("Cheap", (market_stats['Category'] == '🟢 Cheap').sum())
        
        # ============== TAB 4: GEOGRAPHICAL ANALYSIS ==============
        with tab4:
            st.subheader("🌍 Geographical Analysis")
            
            st.info("""  
            **💡 Geographic Patterns:** Maps reveal regional price disparities. Coastal cities often have higher prices due to demand. 
            Remote areas may show lower diversity but transportation costs can increase prices. Look for clusters and outliers.
            """)
            
            # Prepare geo data
            geo_data = df_prices.groupby(['market', 'admin1', 'admin2', 'latitude', 'longitude']).agg({
                'price': ['mean', 'std', 'count'],
                'commodity': 'nunique'
            }).reset_index()
            geo_data.columns = ['Market', 'Wilaya', 'District', 'Lat', 'Lon', 'Avg Price', 'Std Dev', 'Records', 'N Commodities']
            geo_data = geo_data.dropna(subset=['Lat', 'Lon'])
            
            # Map view selector
            map_view = st.radio("Select Map View:", ["Price Hotspots", "Commodity Diversity", "Price Volatility"], horizontal=True, key='geo_map')
            
            metric_map = {"Price Hotspots": "Avg Price", "Commodity Diversity": "N Commodities", "Price Volatility": "Std Dev"}
            color_map = {"Price Hotspots": "YlOrRd", "Commodity Diversity": "Viridis", "Price Volatility": "Reds"}
            
            fig_map = px.scatter_mapbox(
                geo_data, lat="Lat", lon="Lon",
                color=metric_map[map_view], size="Records",
                hover_name="Market",
                hover_data={'Wilaya': True, 'Avg Price': ':.2f', 'N Commodities': True, 'Records': True, 'Lat': False, 'Lon': False},
                color_continuous_scale=color_map[map_view],
                size_max=25, zoom=4,
                center={"lat": 28.0, "lon": 2.5},
                mapbox_style="carto-positron",
                title=f"Algeria Markets - {map_view}"
            )
            fig_map.update_layout(height=550, margin={"r": 0, "t": 50, "l": 0, "b": 0})
            st.plotly_chart(fig_map, use_container_width=True)
            
            st.markdown("---")
            
            # Regional comparison
            st.markdown("### 🏙️ Regional Price Comparison")
            
            region_level = st.radio("Region Level:", ["Wilaya (Admin1)", "District (Admin2)"], horizontal=True, key='region_lvl')
            
            if region_level == "Wilaya (Admin1)":
                admin_stats = filtered_data.groupby('admin1').agg({
                    'price': ['mean', 'std'],
                    'market': 'nunique',
                    'commodity': 'nunique'
                }).reset_index()
                admin_stats.columns = ['Region', 'Avg Price', 'Std Dev', 'Markets', 'Commodities']
                admin_stats = admin_stats.sort_values('Avg Price', ascending=False)
                
                overall = filtered_data['price'].mean()
                admin_stats['Status'] = admin_stats['Avg Price'].apply(lambda x: '🔴 Above' if x > overall else '🟢 Below')
                
                col1, col2 = st.columns(2)
                with col1:
                    colors = [COLOR_EXPENSIVE if x > overall else COLOR_CHEAP for x in admin_stats['Avg Price']]
                    fig = go.Figure(data=[go.Bar(
                        x=admin_stats['Avg Price'], y=admin_stats['Region'], orientation='h', marker_color=colors,
                        text=admin_stats['Avg Price'].round(0), textposition='outside', texttemplate='%{text} DZD'
                    )])
                    fig.add_vline(x=overall, line_dash="dash", line_color="gray", 
                                 annotation_text=f"National Avg: {overall:.0f} DZD",
                                 annotation_position="top")
                    fig.update_layout(title="Average Price by Wilaya<br><sub>Red=Above National Avg | Green=Below National Avg</sub>", 
                                     xaxis_title="Avg Price (DZD)", height=450, template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.dataframe(admin_stats.round(2), use_container_width=True, height=400, hide_index=True)
            
            else:  # Admin2
                admin_stats = filtered_data.groupby(['admin1', 'admin2']).agg({'price': 'mean'}).reset_index()
                admin_stats.columns = ['Wilaya', 'District', 'Avg Price']
                admin_stats = admin_stats.sort_values('Avg Price', ascending=False)
                
                selected_wilaya = st.selectbox("Filter by Wilaya:", ['All'] + list(admin_stats['Wilaya'].unique()), key='admin2_filter')
                
                if selected_wilaya != 'All':
                    admin_stats = admin_stats[admin_stats['Wilaya'] == selected_wilaya]
                else:
                    admin_stats = admin_stats.head(30)
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = go.Figure(data=[go.Bar(
                        x=admin_stats['Avg Price'], y=admin_stats['District'], orientation='h',
                        marker=dict(color=admin_stats['Avg Price'], colorscale='Viridis')
                    )])
                    fig.update_layout(title="Average Price by District", xaxis_title="Avg Price (DZD)", height=500, template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.dataframe(admin_stats.round(2), use_container_width=True, height=450, hide_index=True)
            
            st.markdown("---")
            
            # Regional heatmap
            st.markdown("### 🔥 Regional Price Heatmap")
            
            regional_cat = filtered_data.groupby(['admin1', 'category'])['price'].mean().reset_index()
            regional_pivot = regional_cat.pivot(index='admin1', columns='category', values='price')
            
            fig = go.Figure(data=go.Heatmap(
                z=regional_pivot.values, x=regional_pivot.columns, y=regional_pivot.index,
                colorscale='RdYlGn_r', hovertemplate='Wilaya: %{y}<br>Category: %{x}<br>Price: DZD %{z:.2f}<extra></extra>'
            ))
            fig.update_layout(title="Price by Wilaya and Category", xaxis_title="Category", yaxis_title="Wilaya", height=400, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        
        # ============== TAB 5: DATA QUALITY ==============
        with tab5:
            st.subheader("⚙️ Data Quality Analysis")
            
            st.info("""  
            **💡 Data Quality Matters:** Understanding data sources helps interpret results. 
            **Retail prices** reflect consumer costs. **Wholesale prices** show bulk/trade values (typically 20-40% lower). 
            **Price flags** indicate data reliability: *actual* (measured) vs *estimated* (interpolated).
            """)
            
            # Price type analysis
            st.markdown("### 🏪 Price Type Analysis (Retail vs Wholesale)")
            
            if 'pricetype' in df_prices.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    ptype_counts = filtered_data['pricetype'].value_counts()
                    fig = go.Figure(data=[go.Pie(labels=ptype_counts.index, values=ptype_counts.values)])
                    fig.update_layout(title="Price Type Distribution", height=350)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    ptype_avg = filtered_data.groupby('pricetype')['price'].mean().reset_index()
                    ptype_avg.columns = ['Price Type', 'Avg Price']
                    fig = go.Figure(data=[go.Bar(x=ptype_avg['Price Type'], y=ptype_avg['Avg Price'], 
                                                  text=ptype_avg['Avg Price'].round(2), textposition='outside')])
                    fig.update_layout(title="Average Price by Type", yaxis_title="Avg Price (DZD)", height=350, template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Price type data not available.")
            
            st.markdown("---")
            
            # Price flag analysis
            st.markdown("### 🚩 Price Flag Analysis (Data Quality)")
            
            if 'priceflag' in df_prices.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    flag_counts = filtered_data['priceflag'].value_counts()
                    flag_colors = {'actual': '#2ca02c', 'estimated': '#ff7f0e', 'missing': '#d62728'}
                    colors = [flag_colors.get(f.lower(), '#7f7f7f') for f in flag_counts.index]
                    
                    fig = go.Figure(data=[go.Pie(labels=flag_counts.index, values=flag_counts.values, marker_colors=colors)])
                    fig.update_layout(title="Price Flag Distribution", height=350)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    total = len(filtered_data)
                    actual_pct = (flag_counts.get('actual', 0) / total * 100) if 'actual' in flag_counts.index else 0
                    mcol1, mcol2, mcol3 = st.columns(3)
                    mcol1.metric("Total Records", total)
                    mcol2.metric("Actual Data %", f"{actual_pct:.1f}%")
                    mcol3.metric("Quality", "🟢 Good" if actual_pct > 80 else ("🟡 Fair" if actual_pct > 50 else "🔴 Poor"))
                
                with col2:
                    flag_avg = filtered_data.groupby('priceflag')['price'].agg(['mean', 'std', 'count']).reset_index()
                    flag_avg.columns = ['Flag', 'Avg Price', 'Std Dev', 'Count']
                    
                    fig = go.Figure(data=[go.Bar(
                        x=flag_avg['Flag'], y=flag_avg['Avg Price'], 
                        marker_color=[flag_colors.get(f.lower(), '#7f7f7f') for f in flag_avg['Flag']],
                        text=flag_avg['Avg Price'].round(2), textposition='outside',
                        error_y=dict(type='data', array=flag_avg['Std Dev'])
                    )])
                    fig.update_layout(title="Average Price by Flag", yaxis_title="Avg Price (DZD)", height=350, template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(flag_avg.round(2), use_container_width=True, hide_index=True)
            else:
                st.info("Price flag data not available.")
            
            st.markdown("---")
            
            # Unit analysis
            st.markdown("### ⚖️ Unit Analysis")
            
            if 'unit' in df_prices.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    unit_counts = filtered_data['unit'].value_counts()
                    fig = go.Figure(data=[go.Bar(
                        x=unit_counts.values, y=unit_counts.index, orientation='h',
                        marker=dict(color=unit_counts.values, colorscale='Viridis')
                    )])
                    fig.update_layout(title="Records by Unit", xaxis_title="Count", height=400, template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    unit_price = filtered_data.groupby('unit')['price'].mean().sort_values(ascending=False).reset_index()
                    unit_price.columns = ['Unit', 'Avg Price']
                    fig = go.Figure(data=[go.Bar(
                        x=unit_price['Avg Price'], y=unit_price['Unit'], orientation='h',
                        marker=dict(color=unit_price['Avg Price'], colorscale='YlOrRd')
                    )])
                    fig.update_layout(title="Average Price by Unit", xaxis_title="Avg Price (DZD)", height=400, template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Unit by category
                st.markdown("#### Unit Usage by Category")
                unit_cat = filtered_data.groupby(['category', 'unit']).size().reset_index(name='count')
                unit_pivot = unit_cat.pivot(index='category', columns='unit', values='count').fillna(0)
                
                fig = go.Figure(data=go.Heatmap(
                    z=unit_pivot.values, x=unit_pivot.columns, y=unit_pivot.index,
                    colorscale='Blues', hovertemplate='Category: %{y}<br>Unit: %{x}<br>Count: %{z}<extra></extra>'
                ))
                fig.update_layout(title="Unit Usage by Category", height=350, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Unit data not available.")
        
        # ============== TAB 6: FORECASTS ==============
        with tab6:
            st.subheader("🔮 Prophet Model Forecasts")
            
            st.info("""  
            **💡 Forecasting Insights:** The Prophet model identifies price trends and seasonal patterns. 
            The **shaded area** represents confidence intervals (prediction uncertainty). Wider bands = less certainty. 
            The **trend line** (green dashed) shows the long-term direction, filtering out seasonal noise.
            """)
            
            # Model info
            if model_metadata:
                col1, col2, col3, col4 = st.columns(4)
                col1.info(f"**Model:** {model_metadata.get('model_name', 'N/A')}")
                col2.info(f"**Version:** {model_metadata.get('version', 'N/A')}")
                col3.info(f"**Seasonality:** {model_metadata.get('config', {}).get('seasonality_mode', 'N/A')}")
                col4.info(f"**Periods:** {model_metadata.get('config', {}).get('horizon', 'N/A')}")
            
            st.markdown("---")
            
            # Historical forecast with confidence bands
            st.markdown("### 📈 Price Forecast with Confidence Intervals")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_forecast['date'], y=df_forecast['predicted_price'], name='Predicted', line=dict(color='#1f77b4', width=2)))
            fig.add_trace(go.Scatter(x=df_forecast['date'], y=df_forecast['trend'], name='Trend', line=dict(color='#2ca02c', width=2, dash='dash')))
            fig.add_trace(go.Scatter(
                x=df_forecast['date'].tolist() + df_forecast['date'].tolist()[::-1],
                y=df_forecast['upper_bound'].tolist() + df_forecast['lower_bound'].tolist()[::-1],
                fill='toself', fillcolor='rgba(31,119,180,0.2)',
                line=dict(color='rgba(255,255,255,0)'), name='Confidence Interval', hoverinfo='skip'
            ))
            fig.update_layout(title="Price Forecast with Trend and Confidence Intervals", xaxis_title="Date", yaxis_title="Price (DZD)", height=450, template='plotly_white', hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Future forecast
            st.markdown("### 🔮 Future Price Forecast")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_future['date'], y=df_future['predicted_price'], name='Forecast', line=dict(color='#2ca02c', width=3)))
                fig.add_trace(go.Scatter(
                    x=df_future['date'].tolist() + df_future['date'].tolist()[::-1],
                    y=df_future['upper_bound'].tolist() + df_future['lower_bound'].tolist()[::-1],
                    fill='toself', fillcolor='rgba(44,160,44,0.2)',
                    line=dict(color='rgba(255,255,255,0)'), name='Confidence Interval', hoverinfo='skip'
                ))
                fig.update_layout(title="12-Month Price Forecast", xaxis_title="Date", yaxis_title="Predicted Price (DZD)", height=400, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Forecast Summary**")
                future_display = df_future[['date', 'predicted_price', 'lower_bound', 'upper_bound']].copy()
                future_display['date'] = future_display['date'].dt.strftime('%b %Y')
                future_display.columns = ['Month', 'Predicted', 'Lower', 'Upper']
                st.dataframe(future_display.round(2), use_container_width=True, hide_index=True)
        
        # ============== TAB 7: STATISTICS ==============
        with tab7:
            st.subheader("📉 Statistical Analysis")
            
            st.info("""  
            **💡 Statistical Insights:** The distribution shows price frequency patterns. 
            **Right-skewed** (long tail right) = few expensive outliers. **Left-skewed** = few cheap outliers. 
            **Correlation** shows how variables relate: values near +1 (strong positive) or -1 (strong negative) indicate meaningful relationships.
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure(data=[go.Histogram(x=filtered_data['price'], nbinsx=50, marker=dict(color='rgba(0,100,200,0.7)'))])
                fig.update_layout(title="Price Distribution", xaxis_title="Price (DZD)", yaxis_title="Frequency", height=400, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure(data=[go.Histogram(x=np.log1p(filtered_data['price']), nbinsx=50, marker=dict(color='rgba(200,0,0,0.7)'))])
                fig.update_layout(title="Log-Scaled Price Distribution", xaxis_title="Log(Price + 1)", yaxis_title="Frequency", height=400, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
            
            # Correlation
            st.markdown("### 📊 Correlation Analysis")
            
            numeric_cols = filtered_data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 1:
                corr = filtered_data[numeric_cols].corr()
                fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale='RdBu_r', zmid=0))
                fig.update_layout(title="Correlation Matrix", height=400, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
            
            # Summary stats
            st.markdown("### 📋 Summary Statistics")
            
            summary = {
                'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q1 (25%)', 'Q3 (75%)', 'IQR', 'Skewness', 'Kurtosis'],
                'Price (DZD)': [
                    f"{filtered_data['price'].mean():.2f}",
                    f"{filtered_data['price'].median():.2f}",
                    f"{filtered_data['price'].std():.2f}",
                    f"{filtered_data['price'].min():.2f}",
                    f"{filtered_data['price'].max():.2f}",
                    f"{filtered_data['price'].quantile(0.25):.2f}",
                    f"{filtered_data['price'].quantile(0.75):.2f}",
                    f"{filtered_data['price'].quantile(0.75) - filtered_data['price'].quantile(0.25):.2f}",
                    f"{filtered_data['price'].skew():.2f}",
                    f"{filtered_data['price'].kurtosis():.2f}"
                ],
                'Price (USD)': [
                    f"{filtered_data['usdprice'].mean():.2f}",
                    f"{filtered_data['usdprice'].median():.2f}",
                    f"{filtered_data['usdprice'].std():.2f}",
                    f"{filtered_data['usdprice'].min():.2f}",
                    f"{filtered_data['usdprice'].max():.2f}",
                    f"{filtered_data['usdprice'].quantile(0.25):.2f}",
                    f"{filtered_data['usdprice'].quantile(0.75):.2f}",
                    f"{filtered_data['usdprice'].quantile(0.75) - filtered_data['usdprice'].quantile(0.25):.2f}",
                    f"{filtered_data['usdprice'].skew():.2f}",
                    f"{filtered_data['usdprice'].kurtosis():.2f}"
                ]
            }
            st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
    
    # ============== CATEGORY VIEW ==============
    else:
        category_key = selected_view.replace("📁 ", "").lower()
        
        if category_key in category_data:
            df_cat = category_data[category_key]
            cat_prices = df_prices[df_prices['category'].str.lower().str.contains(category_key.split()[0])]
            
            st.markdown(f'<div class="header-title">📁 {category_key.title()} Analysis</div>', unsafe_allow_html=True)
            
            # Category filters
            col1, col2 = st.columns(2)
            with col1:
                date_range = st.date_input(
                    "Date Range",
                    value=(df_cat['ds'].min(), df_cat['ds'].max()),
                    key='cat_date'
                )
            with col2:
                if not cat_prices.empty:
                    markets = st.multiselect("Markets", cat_prices['market'].unique(), default=list(cat_prices['market'].unique())[:3], key='cat_mkt')
            
            # Filter data
            mask = (df_cat['ds'] >= pd.Timestamp(date_range[0])) & (df_cat['ds'] <= pd.Timestamp(date_range[1]))
            df_cat = df_cat[mask]
            
            if not cat_prices.empty:
                cat_mask = (
                    (cat_prices['date'] >= pd.Timestamp(date_range[0])) &
                    (cat_prices['date'] <= pd.Timestamp(date_range[1])) &
                    (cat_prices['market'].isin(markets))
                )
                filtered_cat = cat_prices[cat_mask]
            else:
                filtered_cat = cat_prices
            
            # KPIs
            st.markdown("### 📈 Category KPIs")
            kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
            
            if not filtered_cat.empty:
                kpi1.metric("Avg Price (DZD)", f"{filtered_cat['price'].mean():.2f}")
                kpi2.metric("Max Price", f"{filtered_cat['price'].max():.2f}")
                kpi3.metric("Min Price", f"{filtered_cat['price'].min():.2f}")
                kpi4.metric("Volatility", f"{filtered_cat['price'].std():.2f}")
                kpi5.metric("Records", f"{len(filtered_cat):,}")
            
            st.markdown("---")
            
            # Category tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Trends", "🔮 Forecast", "🗺️ Market Analysis", "📉 Statistics"])
            
            with tab1:
                st.subheader(f"Price Trends - {category_key.title()}")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_cat['ds'], y=df_cat['price_dzd'], name='Average Price', mode='lines+markers', line=dict(color='#1f77b4', width=2)))
                fig.update_layout(title=f"{category_key.title()} - Historical Price Trend", xaxis_title="Date", yaxis_title="Price (DZD)", height=450, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
                
                if not filtered_cat.empty:
                    st.subheader("Commodities in This Category")
                    comm_prices = filtered_cat.groupby('commodity')['price'].agg(['mean', 'min', 'max', 'std', 'count']).round(2)
                    comm_prices.columns = ['Avg Price', 'Min', 'Max', 'Std Dev', 'Records']
                    st.dataframe(comm_prices.sort_values('Avg Price', ascending=False), use_container_width=True)
            
            with tab2:
                st.subheader(f"Forecast - {category_key.title()}")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_cat['ds'], y=df_cat['y'], name='Price Index (USD)', mode='lines+markers', line=dict(color='#2ca02c', width=2)))
                fig.update_layout(title=f"{category_key.title()} - Price Index Trend", xaxis_title="Date", yaxis_title="Price Index (USD)", height=400, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Latest Price", f"{df_cat['price_dzd'].iloc[-1]:.2f} DZD")
                price_change = df_cat['price_dzd'].iloc[-1] - df_cat['price_dzd'].iloc[0]
                col2.metric("Price Change", f"{price_change:+.2f} DZD")
                pct_change = ((df_cat['price_dzd'].iloc[-1] / df_cat['price_dzd'].iloc[0]) - 1) * 100
                col3.metric("% Change", f"{pct_change:+.1f}%")
                col4.metric("Observations", f"{df_cat['n_observations'].sum():,}")
            
            with tab3:
                st.subheader(f"Market Analysis - {category_key.title()}")
                
                if not filtered_cat.empty:
                    market_avg = filtered_cat.groupby('market')['price'].mean().sort_values(ascending=False)
                    fig = go.Figure(data=[go.Bar(x=market_avg.values, y=market_avg.index, orientation='h', marker=dict(color=market_avg.values, colorscale='Viridis'))])
                    fig.update_layout(title=f"Average {category_key.title()} Prices by Market", xaxis_title="Price (DZD)", height=400, template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Map
                    cat_geo = filtered_cat.groupby(['market', 'latitude', 'longitude']).agg({'price': 'mean'}).reset_index().dropna()
                    if not cat_geo.empty:
                        fig_map = px.scatter_mapbox(cat_geo, lat="latitude", lon="longitude", color="price", size="price", hover_name="market",
                            color_continuous_scale="YlOrRd", size_max=20, zoom=4, center={"lat": 28.0, "lon": 2.5}, mapbox_style="carto-positron",
                            title=f"{category_key.title()} Prices by Location")
                        fig_map.update_layout(height=500)
                        st.plotly_chart(fig_map, use_container_width=True)
                else:
                    st.info("No market data available for this category.")
            
            with tab4:
                st.subheader(f"Statistics - {category_key.title()}")
                
                if not filtered_cat.empty:
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = go.Figure(data=[go.Histogram(x=filtered_cat['price'], nbinsx=30, marker=dict(color='#1f77b4'))])
                        fig.update_layout(title="Price Distribution", xaxis_title="Price (DZD)", height=350, template='plotly_white')
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        fig = go.Figure(data=[go.Box(y=filtered_cat['price'], marker=dict(color='#2ca02c'))])
                        fig.update_layout(title="Price Box Plot", yaxis_title="Price (DZD)", height=350, template='plotly_white')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    summary = {
                        'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q1', 'Q3', 'Skewness'],
                        'Value': [
                            f"{filtered_cat['price'].mean():.2f}",
                            f"{filtered_cat['price'].median():.2f}",
                            f"{filtered_cat['price'].std():.2f}",
                            f"{filtered_cat['price'].min():.2f}",
                            f"{filtered_cat['price'].max():.2f}",
                            f"{filtered_cat['price'].quantile(0.25):.2f}",
                            f"{filtered_cat['price'].quantile(0.75):.2f}",
                            f"{filtered_cat['price'].skew():.2f}"
                        ]
                    }
                    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
        else:
            st.warning("Category data not available.")

elif dataset_key == "laptops":
    st.markdown('<div class="header-title">💻 Laptop Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="coming-soon">
            <h2>🚧 Coming Soon!</h2>
            <p>The Laptop Analytics module is under development.</p>
        </div>
    """, unsafe_allow_html=True)

elif dataset_key == "cars":
    st.markdown('<div class="header-title">🚗 Car Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="coming-soon">
            <h2>🚧 Coming Soon!</h2>
            <p>The Car Analytics module is under development.</p>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(f"""
    <div style='text-align: center; color: #888;'>
    <p>📊 Food Price Analytics Dashboard | Algeria</p>
    <p>Last Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
""", unsafe_allow_html=True)
