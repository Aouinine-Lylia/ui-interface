import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from hijri_converter import Hijri
from datetime import date, datetime
from dateutil.relativedelta import relativedelta


# ======================
# EVENT GENERATORS
# ======================
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
EVENT_COLORS = {
    "religious": "rgba(44,160,44,0.25)",
    "cultural": "rgba(152,223,138,0.25)",
    "season": "rgba(255,187,120,0.25)",
    "institutional": "rgba(214,39,40,0.25)",
    "covid": "rgba(127,127,127,0.3)"
}

# ======================
# MAIN GRAPH FUNCTION
# ======================
def plot_prices(df, date_col, price_col, events, title):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[price_col],
        mode="lines",
        name="Price Trend",
        line=dict(color="#1f77b4", width=3)
    ))

    for name, start, end, category in events:
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor=EVENT_COLORS[category],
            opacity=1,
            layer="below",
            line_width=0,
            annotation_text=name,
            annotation_position="top left"
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        height=600,
        legend_title="Legend"
    )

    return fig

