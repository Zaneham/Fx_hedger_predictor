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


base_dir = Path(__file__).resolve().parent
st.write("DEBUG: app folder:", str(base_dir))
st.write("DEBUG: csv files:", [p.name for p in base_dir.glob("hedge_log_*.csv")])

# --- Securely obtain API key (prefer st.secrets on Cloud, then env var, then .env file) ---
def _load_api_key():
    #We use secrets first
    try:
        key = st.secrets.get("FX_API_KEY")
        if key:
            return key
    except Exception:
        pass

   #then env 
    key = os.getenv("FX_API_KEY")
    if key:
        return key

   
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=env_path, override=False)
            key = os.getenv("FX_API_KEY")
            if key:
                return key
        except Exception:
            # minimal fallback loader if python-dotenv not installed
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if not line or line.strip().startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
            key = os.getenv("FX_API_KEY")
            if key:
                return key

    return None

_API_KEY = _load_api_key()

# --- App config ---
st.set_page_config(page_title="FX Hedging Dashboard", layout="wide")
st.title("📊 FX Hedging Dashboard 📊")

# --- Helper functions ---
def _get_env_key():
    return _API_KEY

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

@st.cache_data(ttl=60)
def get_live_rate(base="NZD", quote="USD"):
    api_key = _get_env_key()
    if api_key:
        try:
            data = _call_exchangerateapi(base, api_key)
            rate = data.get("conversion_rates", {}).get(quote)
            if rate is not None:
                return float(rate)
        except Exception:
            # primary failed or key invalid — fall through to fallback
            pass

    try:
        data2 = _call_exchangerate_host(base, quote)
        rate2 = data2.get("rates", {}).get(quote)
        if rate2 is not None:
            return float(rate2)
    except Exception:
        pass

    return None

def load_hedge_log_for_pair(base, quote):
    filename = f"hedge_log_{base.lower()}{quote.lower()}.csv"
    base_dir = Path(__file__).resolve().parent
    path = base_dir / filename

    st.write("DEBUG: trying to load:", str(path))
    if not path.exists():
        st.write(f"DEBUG: file not found: {filename}")
        return pd.DataFrame()

    try:
        stat = path.stat()
        st.write("DEBUG: file size (bytes):", stat.st_size)
    except Exception as e:
        st.write("DEBUG: stat error:", e)

    try:
        raw = "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[:10])
        st.code(raw, language="text")
    except Exception as e:
        st.write("DEBUG: raw read error:", e)

    try:
        df = pd.read_csv(path)
        if df.shape[0] == 0 and df.shape[1] > 1:
            df = pd.read_csv(path, header=None)
            df.columns = [f"col_{i}" for i in range(df.shape[1])]
        df.columns = df.columns.str.strip().str.replace(" ", "_")
        st.write("DEBUG: pd.read_csv shape:", df.shape)
        st.write("DEBUG: columns:", df.columns.tolist())
        st.dataframe(df.head(5), use_container_width=True)
        return df
    except Exception as e:
        st.write("DEBUG: pd.read_csv failed:", e)

    for enc in ("utf-8-sig", "latin1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            df.columns = df.columns.str.strip().str.replace(" ", "_")
            st.write(f"DEBUG: read with encoding={enc}; shape:", df.shape)
            st.dataframe(df.head(5), use_container_width=True)
            return df
        except Exception as e:
            st.write(f"DEBUG: encoding={enc} failed:", e)

    try:
        import csv
        sample = path.read_text(encoding="utf-8", errors="replace").splitlines()[:5]
        sniff = csv.Sniffer()
        delimiter = sniff.sniff("\n".join(sample)).delimiter
        st.write("DEBUG: sniffed delimiter:", delimiter)
        df = pd.read_csv(path, sep=delimiter, encoding="utf-8", engine="python")
        df.columns = df.columns.str.strip().str.replace(" ", "_")
        st.write("DEBUG: read with sniffed delimiter; shape:", df.shape)
        st.dataframe(df.head(5), use_container_width=True)
        return df
    except Exception as e:
        st.write("DEBUG: delimiter-sniff read failed:", e)

    st.write("DEBUG: returning empty DataFrame")
    return pd.DataFrame()
# --- Sidebar controls ---
with st.sidebar:
    st.markdown("## How It Works")
    st.markdown(
        "Kia ora/Hello — this app demonstrates a proof-of-concept LSTM-based FX hedging workflow."
    )
    st.header("🔧 Controls")
    currency_pairs = ["NZD/USD", "USD/NZD", "AUD/NZD", "NZD/AUD"]
    selected_pair = st.selectbox("Select currency pair", currency_pairs)
    base_currency, quote_currency = selected_pair.split("/")
    use_sentiment = st.checkbox("Include sentiment features - To be added")

# --- Main app flow ---
log = load_hedge_log_for_pair(base_currency, quote_currency)
if log.empty:
    st.warning(
        f"No hedge log found for {selected_pair}. Expected file: "
        f"hedge_log_{base_currency.lower()}{quote_currency.lower()}.csv"
    )
    st.info("Create the hedge log CSV in the app folder or switch to a pair with an existing log.")
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

# --- Filter hedge log by decision ---
st.subheader("🔍 Filter Hedge Log by Decision Type")
decision_types = log.Decision.dropna().unique().tolist() if "Decision" in log.columns else []
selected_decision = st.selectbox("Select decision type:", ["All"] + decision_types)

filtered_log = log[log.Decision == selected_decision] if (selected_decision != "All" and "Decision" in log.columns) else log
st.dataframe(filtered_log.tail(20), use_container_width=True)

# --- Download button ---
st.download_button(
    label="📥 Download filtered hedge log",
    data=filtered_log.to_csv(index=False),
    file_name=f"filtered_hedge_log_{base_currency.lower()}{quote_currency.lower()}.csv",
    mime="text/csv"
)

# --- Performance metrics ---
st.subheader("📉 Performance Summary 📈")
rmse = log.Error.dropna().pow(2).mean() ** 0.5 if "Error" in log.columns else float("nan")
mae = log.Error.dropna().abs().mean() if "Error" in log.columns else float("nan")
dir_acc = log.CorrectDirection.dropna().mean() * 100 if "CorrectDirection" in log.columns else float("nan")

m1, m2, m3 = st.columns(3)
m1.metric("RMSE", f"{rmse:.5f}")
m2.metric("MAE", f"{mae:.5f}")
m3.metric("Directional Accuracy", f"{dir_acc:.2f}%")

# --- Outcome charts ---
st.subheader("📊 Hedge Outcome Over Time")
if {"Decision", "HedgeOutcome"}.issubset(log.columns):
    outcome_by_decision = log.groupby(["Decision", "HedgeOutcome"]).size().unstack(fill_value=0)
    st.bar_chart(outcome_by_decision)
else:
    st.info("Outcome breakdown not available (missing columns).")

st.subheader("📊 Hedge Outcome Breakdown")
if "HedgeOutcome" in log.columns:
    outcome_counts = log.HedgeOutcome.value_counts()
    st.bar_chart(outcome_counts)
else:
    st.info("No HedgeOutcome column in log.")

# --- Prediction vs Actual chart ---
st.subheader("📉 Predicted vs Actual Rates 📈")
if {"Timestamp", "Live_Rate", "Predicted_Rate", "Actual"}.issubset(log.columns):
    fig, ax = plt.subplots(figsize=(10, 4))
    log.plot(x="Timestamp", y=["Live_Rate", "Predicted_Rate", "Actual"], ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)
else:
    st.info("Prediction vs Actual chart requires columns: Timestamp, Live_Rate, Predicted_Rate, Actual.")

# --- Simulator ---
st.subheader(f"🧪 Hedge Decision Simulator for {selected_pair} 🧪")
default_hyp = latest.Live_Rate if "Live_Rate" in latest else (live_rate or 0.0)
hypothetical_rate = st.number_input(f"Enter hypothetical live {selected_pair} rate:", value=float(default_hyp))
predicted_rate = float(latest.Predicted_Rate) if "Predicted_Rate" in latest else None

if st.button("Simulate Decision"):
    if predicted_rate is None:
        st.warning("No predicted rate available in the log to run simulation.")
    else:
        if predicted_rate > hypothetical_rate:
            st.success("Model would recommend: Hedge now")
        elif predicted_rate < hypothetical_rate:
            st.info("Model would recommend: Wait")
        else:
            st.warning("Model is neutral — no clear signal")

st.markdown("DISCLAIMER: This is not financial advice nor is it intended to be.")
