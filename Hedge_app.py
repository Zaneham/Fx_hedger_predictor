# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 19:47:25 2025
@author: GGPC
"""
from pathlib import Path
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests

# --- .env loader (works with or without python-dotenv) ---
try:
    from dotenv import load_dotenv  # optional dependency
    _HAS_DOTENV = True
except Exception:
    _HAS_DOTENV = False

ENV_PATH = Path(r"C:\Users\GGPC\OneDrive\Documents\New folder") / ".env"

if ENV_PATH.exists():
    if _HAS_DOTENV:
        load_dotenv(dotenv_path=ENV_PATH, override=False)
    else:
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            if not line or line.strip().startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
else:
    # only warn once early; replace with st.info if noisy in deploy
    st.warning(".env not found at expected path; check path or move .env to project folder.")

# Debug (temporary) — remove after confirming
st.write("DEBUG: .env path:", str(ENV_PATH))
st.write("DEBUG: .env exists:", ENV_PATH.exists())
st.write("DEBUG: FX_API_KEY present after load:", bool(os.getenv("FX_API_KEY")))

# --- App config ---
st.set_page_config(page_title="FX Hedging Dashboard", layout="wide")
st.title("📊 FX Hedging Dashboard 📊")

# --- Helper functions used by get_live_rate ---
def _get_env_key():
    return os.getenv("FX_API_KEY")

def _call_exchangerateapi(base, api_key, timeout=8):
    url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/{base}"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

def _call_exchangerate_host(base, quote, timeout=6):
    url = f"https://api.exchangerate.host/latest?base={base}&symbols={quote}"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

# --- Live rate (cached) ---
@st.cache_data(ttl=60)
def get_live_rate(base="NZD", quote="USD"):
    """
    Return float rate base -> quote.
    Tries primary exchangerate-api (requires FX_API_KEY), then fallback exchangerate.host.
    Returns float or None.
    """
    api_key = _get_env_key()
    if api_key:
        try:
            data = _call_exchangerateapi(base, api_key)
            rate = data.get("conversion_rates", {}).get(quote)
            if rate is not None:
                return float(rate)
            else:
                st.warning(f"Primary API returned no rate for {base}->{quote}.")
        except Exception as e:
            st.warning(f"Primary API error: {e}")
    else:
        st.info("FX_API_KEY not set; skipping primary provider.")

    # fallback
    try:
        data2 = _call_exchangerate_host(base, quote)
        rate2 = data2.get("rates", {}).get(quote)
        if rate2 is not None:
            st.info("Using fallback provider exchangerate.host")
            return float(rate2)
        else:
            st.warning(f"Fallback provider returned no rate for {base}->{quote}.")
    except Exception as e:
        st.warning(f"Fallback provider error: {e}")

    return None

# --- Hedge log loader ---
def load_hedge_log_for_pair(base, quote):
    filename = f"hedge_log_{base.lower()}{quote.lower()}.csv"
    base_dir = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    return df

# --- Sidebar controls ---
with st.sidebar:
    st.markdown("## How It Works")
    st.markdown("Kia ora/Hello — this app demonstrates a proof-of-concept LSTM-based FX hedging workflow.")
    st.header("🔧 Controls")
    currency_pairs = ["NZD/USD", "USD/NZD", "AUD/NZD", "NZD/AUD"]
    selected_pair = st.selectbox("Select currency pair", currency_pairs)
    base_currency, quote_currency = selected_pair.split("/")
    use_sentiment = st.checkbox("Include sentiment features - To be added")

# --- Main app flow (unchanged) ---
log = load_hedge_log_for_pair(base_currency, quote_currency)
if log.empty:
    st.warning(
        f"No hedge log found for {selected_pair}. Expected file: "
        f"hedge_log_{base_currency.lower()}{quote_currency.lower()}.csv"
    )
    st.info("Create the hedge log CSV or switch to a pair with an existing log.")
    st.stop()

log.columns = log.columns.str.strip().str.replace(" ", "_")
st.subheader("🔮 Latest Hedge Decision")
latest = log.iloc[-1]

col1, col2, col3 = st.columns(3)
live_rate = get_live_rate(base_currency, quote_currency)
if live_rate is not None:
    col1.metric(f"Live {selected_pair} Rate", f"{live_rate:.5f}")
else:
    col1.warning("Live rate unavailable")

pred_val = latest.Predicted_Rate if "Predicted_Rate" in latest else None
decision_val = latest.Decision if "Decision" in latest else None

col2.metric("Predicted Rate", f"{float(pred_val):.5f}" if pred_val is not None else "N/A")
col3.metric("Decision", decision_val or "N/A")

st.subheader("🔍 Filter Hedge Log by Decision Type")
decision_types = log.Decision.dropna().unique().tolist() if "Decision" in log.columns else []
selected_decision = st.selectbox("Select decision type:", ["All"] + decision_types)

filtered_log = log[log.Decision == selected_decision] if (selected_decision != "All" and "Decision" in log.columns) else log
st.dataframe(filtered_log.tail(20), use_container_width=True)

st.download_button(
    label="📥 Download filtered hedge log",
    data=filtered_log.to_csv(index=False),
    file_name=f"filtered_hedge_log_{base_currency.lower()}{quote_currency.lower()}.csv",
    mime="text/csv"
)

st.subheader("📉 Performance Summary 📈")
rmse = log.Error.dropna().pow(2).mean() ** 0.5 if "Error" in log.columns else float("nan")
mae = log.Error.dropna().abs().mean() if "Error" in log.columns else float("nan")
dir_acc = log.CorrectDirection.dropna().mean() * 100 if "CorrectDirection" in log.columns else float("nan")

m1, m2, m3 = st.columns(3)
m1.metric("RMSE", f"{rmse:.5f}")
m2.metric("MAE", f"{mae:.5f}")
m3.metric("Directional Accuracy", f"{dir_acc:.2f}%")

st.subheader("📊 Hedge Outcome Over Time")
if {"Decision", "HedgeOutcome"}.issubset(log.columns):
    outcome_by_decision = log.groupby(["Decision", "HedgeOutcome"]).size().unstack(fill_value=0)
    st.bar_chart(outcome_by_decision)
else:
    st.info("Outcome breakdown not available (missing columns).")

# Prediction vs Actual chart
st.subheader("📉 Predicted vs Actual Rates 📈")
if {"Timestamp", "Live_Rate", "Predicted_Rate", "Actual"}.issubset(log.columns):
    fig, ax = plt.subplots(figsize=(10, 4))
    log.plot(x="Timestamp", y=["Live_Rate", "Predicted_Rate", "Actual"], ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)
else:
    st.info("Prediction vs Actual chart requires columns: Timestamp, Live_Rate, Predicted_Rate, Actual.")
