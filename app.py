import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import create_engine, text

st.set_page_config(page_title="Multi-Asset ML Forecasting", layout="wide")

st.title("📈 Multi-Asset Multi-Model ML Forecasting Dashboard")
st.markdown("Automated Machine Learning Forecast System supporting GBRT, AdaBoost, SVR, and BiLSTM.")

# SQLite Engine
engine = create_engine("sqlite:///./forecasting_system.db")

# Fetch unique symbols
try:
    symbols_df = pd.read_sql("SELECT DISTINCT symbol FROM actual_prices", con=engine)
    symbols = symbols_df["symbol"].tolist()
except Exception:
    symbols = ["CRUDE_OIL", "GOLD", "BITCOIN"]

selected_symbol = st.sidebar.selectbox("Select Asset", symbols if symbols else ["CRUDE_OIL", "GOLD", "BITCOIN"])

# Load Actual Data
try:
    prices_df = pd.read_sql(
        text("SELECT date, price FROM actual_prices WHERE symbol = :s ORDER BY date ASC"),
        con=engine, params={"s": selected_symbol}
    )
    prices_df["date"] = pd.to_datetime(prices_df["date"])
except Exception as e:
    st.error(f"Error loading actual prices: {e}")
    prices_df = pd.DataFrame()

# Fetch predictions across all models from db logs if available
try:
    preds_df = pd.read_sql(
        text("SELECT target_date as date, predicted_price, model_version FROM prediction_logs WHERE symbol = :s ORDER BY target_date ASC"),
        con=engine, params={"s": selected_symbol}
    )
    preds_df["date"] = pd.to_datetime(preds_df["date"])
except Exception as e:
    preds_df = pd.DataFrame()

if not prices_df.empty:
    st.subheader(f"{selected_symbol} Price Visualization & Model Forecasts")

    # Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices_df["date"].tail(60), y=prices_df["price"].tail(60), name="Actual History", line=dict(color="white", width=2)))

    if not preds_df.empty:
        # Plot each model's prediction if existing
        models = preds_df["model_version"].unique()
        colors = {"GBRT_v1": "orange", "AdaBoost": "cyan", "SVR": "magenta", "BiLSTM": "yellow"}
        for m in models:
            m_df = preds_df[preds_df["model_version"] == m].tail(7)
            fig.add_trace(go.Scatter(
                x=m_df["date"],
                y=m_df["predicted_price"],
                name=f"7-Day {m} Forecast",
                line=dict(color=colors.get(m, "gray"), width=3, dash="dash")
            ))

    fig.update_layout(template="plotly_dark", hovermode="x unified", margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Display predictions table
    if not preds_df.empty:
        st.subheader("📅 7-Day Forecast Values Comparison")
        pivot_df = preds_df.tail(28).pivot(index="date", columns="model_version", values="predicted_price").tail(7)
        st.dataframe(pivot_df, use_container_width=True)
else:
    st.warning("No database data found. Make sure forecasting_system.db is in the same directory.")
