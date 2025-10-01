# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 19:47:25 2025
@author: Zane Hambly
"""
from pathlib import Path
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests




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

# --- Main app flow ---
if selected_pair == "AUD/NZD":
    # Coming soon landing page
    st.subheader("🚧 AUD/NZD Coming Soon 🚧")
    st.markdown(
        """
        G'day! We're currently working on adding support for the **AUD/NZD** pair.  
        Stay tuned — live rates, hedge logs, and simulation features will be available here soon.
        """
    )
    st.info("In the meantime, explore other pairs from the sidebar.")

elif selected_pair == "NZD/AUD":
    st.subheader("🚧NZD/AUD Coming Soon🚧")
    st.markdown(
        """ G'day! We're currently working on adding support for the **AUD/NZD** pair.  
        Stay tuned — live rates, hedge logs, and simulation features will be available here soon""" 
        )
    st.info("While you're here, explore the other pairs from the sidebar.")
else:
    # --- Normal flow for supported pairs ---
    try:
        log = load_hedge_log_for_pair(base_currency, quote_currency)
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

        filtered_log = (
            log[log.Decision == selected_decision]
            if (selected_decision != "All" and "Decision" in log.columns)
            else log
        )
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
        st.markdown(
        """
        💡 **What this does:**  
        The simulator is a *what‑if sandbox*. Enter a hypothetical live rate and see how the model
        **would have responded** — hedge now, wait, or stay neutral.  
        It’s designed to help you understand the model’s decision logic under different scenarios.

        ⚠️ **Important:**  
        This is **not financial advice**. It’s a demonstration tool only, and should not be relied on
        for trading or investment decisions.
        """
        )


    

        default_hyp = latest.Live_Rate if "Live_Rate" in latest else (live_rate or 0.0)
        hypothetical_rate = st.number_input(
            f"Enter hypothetical live {selected_pair} rate:", value=float(default_hyp)
        )
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

    except FileNotFoundError:
        st.error(f"No hedge log found for {selected_pair}.")
    except Exception as e:
        st.error(f"Unexpected error loading {selected_pair}: {e}")





