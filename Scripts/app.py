import streamlit as st
import pandas as pd
import numpy as np


# ---------- Load data ----------
@st.cache_data
def load_data():
    segments = pd.read_csv("Data/processed/customer_segments.csv")
    clv_bgnbd = pd.read_csv("Data/processed/customer_clv_bgnbd.csv")
    clv_opt = pd.read_csv("Data/processed/customer_clv_optimized.csv")
    monthly_forecast = pd.read_csv("Data/processed/monthly_clv_forecast.csv")
    monthly_forecast["month"] = pd.to_datetime(monthly_forecast["month"])

    # No rename needed – CLV is already 'pred_clv'
    return segments, clv_bgnbd, clv_opt, monthly_forecast


segments, clv_bgnbd, clv_opt, monthly_forecast = load_data()

# Merge CLV + optimized spend (no suffix on pred_clv)
clv_full = clv_bgnbd.merge(clv_opt, on="customer_id", how="left")
if "pred_clv_x" in clv_full.columns:
    clv_full = clv_full.rename(columns={"pred_clv_x": "pred_clv"})

# ---------- Sidebar ----------
st.sidebar.title("CLV Dashboard")
segment_filter = st.sidebar.selectbox(
    "Select segment",
    options=sorted(segments["Segment"].unique())
)

# ---------- Header ----------
st.title("Optimizing Marketing ROI with CLV")
st.markdown(
    "This app summarizes the CLV modeling, budget optimization, and forecasting pipeline."
)

# ---------- Segment overview ----------
st.subheader("Customer Segments")

seg_counts = segments["Segment"].value_counts().sort_index()
st.bar_chart(seg_counts)

st.write("Selected segment details:")
seg_ids = segments.loc[segments["Segment"] == segment_filter, "CustomerID"].astype(str)
seg_clv = clv_full[clv_full["customer_id"].astype(str).isin(seg_ids)]

st.write("Debug – rows in seg_clv:", len(seg_clv))
st.write("Columns:", list(seg_clv.columns))

if len(seg_clv):
    # Prefer the original CLV from BG/NBD
    if "pred_clv_x" in seg_clv.columns:
        avg_clv_value = f"${seg_clv['pred_clv_x'].mean():,.0f}"
    elif "pred_clv" in seg_clv.columns:
        avg_clv_value = f"${seg_clv['pred_clv'].mean():,.0f}"
    else:
        avg_clv_value = "N/A"
else:
    avg_clv_value = "N/A"


st.metric(
    label="Avg probabilistic CLV (selected segment)",
    value=avg_clv_value
)

# ---------- Optimization impact ----------
st.subheader("Budget Optimization Impact")

budget = 100000
uniform_clv = 2636038.92
optimized_clv = 2724476.24
delta = optimized_clv - uniform_clv
improvement = delta / uniform_clv * 100

col1, col2, col3 = st.columns(3)
col1.metric("Budget", f"${budget:,.0f}")
col2.metric("Uniform CLV", f"${uniform_clv:,.0f}")
col3.metric("Optimized CLV", f"${optimized_clv:,.0f}", f"+{improvement:.1f}%")

st.write(
    f"Reallocating the existing budget toward high‑value customers yields about "
    f"${delta:,.0f} additional CLV (+{improvement:.1f}%)."
)

# ---------- Monthly CLV forecast ----------
st.subheader("Monthly Total CLV Forecast")

monthly_forecast = monthly_forecast.set_index("month")
st.line_chart(monthly_forecast["forecast_clv"])

st.caption("ARIMA(1,1,0) forecast of aggregate monthly CLV over the next 6 months.")
