# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 22:48:43 2025

@author: GGPC
"""


import pandas as pd
import numpy as np
import requests
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from dotenv import load_dotenv
import os
from jb_news.news import CJBNews
from textblob import TextBlob
import yfinance as yf

# --- Load API keys ---
load_dotenv()
JB_API_KEY = os.getenv("JB_API_KEY")      # jblanked.com key - You can use another news gpt api or create your own
NEWS_API_KEY = os.getenv("NEWS_API_KEY")  # newsapi.org key - Pulls headlines, use this with news gpt to create sentiment
FX_API_KEY = os.getenv("FX_API_KEY")      # exchangerate-api key - 

# --- Helper functions ---
def fetch_jb_calendar(api_key, offset=15):
    jb = CJBNews()
    jb.offset = offset
    if jb.calendar(api_key, today=False):
        return pd.DataFrame([
            {
                "Date": pd.to_datetime(event.date).date(),
                "Event": event.name,
                "Currency": event.currency,
                "Forecast": event.forecast,
                "Actual": event.actual,
                "Previous": event.previous,
                "Outcome": event.outcome,
                "Strength": event.strength,
                "Quality": event.quality,
                "Projection": event.projection
            }
            for event in jb.calendar_info
        ])
    return pd.DataFrame()

def engineer_jb_features(jb_df):
    if jb_df.empty:
        return jb_df
    jb_df["Forecast"] = pd.to_numeric(jb_df["Forecast"], errors="coerce")
    jb_df["Actual"] = pd.to_numeric(jb_df["Actual"], errors="coerce")
    jb_df["Surprise"] = jb_df["Actual"] - jb_df["Forecast"]
    outcome_map = {"Better": 1, "Worse": -1, "In-Line": 0}
    jb_df["Outcome_num"] = jb_df["Outcome"].map(outcome_map).fillna(0)
    strength_map = {"Low": 1, "Medium": 2, "High": 3}
    jb_df["Strength_num"] = jb_df["Strength"].map(strength_map).fillna(0)
    quality_map = {"Low": 1, "Medium": 2, "High": 3}
    jb_df["Quality_num"] = jb_df["Quality"].map(quality_map).fillna(0)
    return jb_df[["Date", "Surprise", "Outcome_num", "Strength_num", "Quality_num"]]

def fetch_news_sentiment(api_key):
    url = "https://newsapi.org/v2/everything?q=forex&language=en"
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        news_data = response.json()
        rows = []
        for article in news_data.get("articles", []):
            title = article.get("title") or ""          ### coerce None -> ""
            desc = article.get("description") or ""     ### coerce None -> ""
            text = title + " " + desc
            polarity = TextBlob(text).sentiment.polarity
            sentiment = 1 if polarity > 0.05 else -1 if polarity < -0.05 else 0
            rows.append({
                "Date": pd.to_datetime(article.get("publishedAt")).date(),
                "Sentiment": sentiment
            })
        return pd.DataFrame(rows)
    else:
        print(f"NewsAPI error: {response.status_code}")
        return pd.DataFrame(columns=["Date", "Sentiment"])


# --- Load FX price data ---
# "NZDUSD=X" is the ticker for NZD/USD spot rate
df = yf.download("NZDUSD=X", start="2015-01-01", end="2025-01-01", interval="1d")

if isinstance(df.columns, pd.MultiIndex):
    df = df['Close']['NZDUSD=X']

df = df.reset_index().rename(columns={'NZDUSD=X': 'NZDUSD'})
df['Price'] = 1 / df['NZDUSD']  # Invert to get USD/NZD
df['Date_only'] = df['Date'].dt.date

print("Data range:", df['Date'].min(), "to", df['Date'].max(), "(", len(df), "rows )")

# If MultiIndex (as with single ticker downloads), select just the Close column
if isinstance(df.columns, pd.MultiIndex):
    df = df['Close']['NZDUSD=X']

# Reset index and rename
df = df.reset_index().rename(columns={'NZDUSD=X': 'Price'})
df['Date_only'] = df['Date'].dt.date

print("Data range:", df['Date'].min(), "to", df['Date'].max(), "(", len(df), "rows )")


# --- Fetch jb-news features ---
try:
    jb_calendar_df = fetch_jb_calendar(JB_API_KEY, offset=15)
    jb_features = engineer_jb_features(jb_calendar_df)

    if not jb_features.empty:
        df = pd.merge(df, jb_features, left_on="Date_only", right_on="Date", how="left")
        df.drop(columns=["Date_y"], inplace=True)
        df.rename(columns={"Date_x": "Date"}, inplace=True)
        print("jb-news features merged successfully.")
    else:
        print("jb-news returned no data, skipping.")
except Exception as e:
    print(f"Skipping jb-news features due to error: {e}")
# --- Fetch NewsAPI sentiment ---
sentiment_df = fetch_news_sentiment(NEWS_API_KEY)

# --- Merge sentiment ---
df = pd.merge(df, sentiment_df, left_on="Date_only", right_on="Date", how="left")
df['Sentiment'] = df['Sentiment'].fillna(0)
df.drop(columns=["Date_only", "Date_y"], inplace=True, errors="ignore")
df.rename(columns={"Date_x": "Date"}, inplace=True)

# --- Diagnostics ---
plt.figure(figsize=(8, 4))
plt.hist(df['Price'], bins=50, color='skyblue', edgecolor='black')
plt.title("Distribution of NZD/USD Prices in Training Data")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# --- Prepare data for LSTM ---
full_df = df.copy()

features = ['Price', 'Sentiment',]#'Surprise', 'Outcome_num', 'Strength_num', 'Quality_num' #Use this for later##
train_size = int(len(full_df) * 0.8)
train_df = full_df.iloc[:train_size]
test_df  = full_df.iloc[train_size:]

scaler = MinMaxScaler()
scaler.fit(train_df[features])

train_scaled = scaler.transform(train_df[features])
test_scaled  = scaler.transform(test_df[features])

def create_sequences(data, look_back):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

look_back = 60
X_train, y_train = create_sequences(train_scaled, look_back)
X_test,  y_test  = create_sequences(test_scaled,  look_back)


# --- Build and train LSTM 

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=100, batch_size=32,
          validation_data=(X_test, y_test),
          callbacks=[early_stop])
y_pred_test = model.predict(X_test)

# Invert scaling
y_pred_padded = np.hstack([y_pred_test, np.zeros_like(y_pred_test)])
y_test_padded = np.hstack([y_test.reshape(-1,1), np.zeros_like(y_test.reshape(-1,1))])

y_pred_unscaled = scaler.inverse_transform(y_pred_padded)[:, 0]
y_test_unscaled = scaler.inverse_transform(y_test_padded)[:, 0]

mse = mean_squared_error(y_test_unscaled, y_pred_unscaled)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)

direction_actual = np.sign(np.diff(y_test_unscaled))
direction_pred   = np.sign(np.diff(y_pred_unscaled))
directional_hits = np.mean(direction_actual == direction_pred) * 100

print(f"Test RMSE: {rmse:.6f}")
print(f"Test MAE: {mae:.6f}")
print(f"Directional Accuracy: {directional_hits:.2f}%")

#Current Fx api

url = f"https://v6.exchangerate-api.com/v6/{FX_API_KEY}/latest/NZD"
response = requests.get(url)
data_api = response.json()
live_rate_nzdusd = data_api['conversion_rates']['USD']
live_rate_usdnzd = 1 / live_rate_nzdusd
print(f"Live USD/NZD rate: {live_rate_usdnzd:.5f}")

# Predict next rate

recent_scaled = test_scaled[-(look_back - 1):]
new_point = scaler.transform([[live_rate_usdnzd, 0]])  # neutral sentiment 
updated_sequence = np.vstack([recent_scaled, new_point])
X_live = updated_sequence.reshape(1, look_back, 2)

predicted_scaled = model.predict(X_live)
predicted_scaled_padded = np.array([[predicted_scaled[0][0], 0]])
predicted_rate = scaler.inverse_transform(predicted_scaled_padded)[0][0]
print(f"Predicted next rate: {predicted_rate:.5f}")

# Sensitivity analysis 
print("\nSentiment sensitivity:")
for s in [-1, 0, 1]:
    test_point = scaler.transform([[live_rate_usdnzd, s]])
    updated_seq = np.vstack([recent_scaled, test_point])
    X_sent = updated_seq.reshape(1, look_back, 2)
    pred_scaled = model.predict(X_sent)
    pred_padded = np.array([[pred_scaled[0][0], 0]])
    pred_unscaled = scaler.inverse_transform(pred_padded)[0][0]
    print(f"Sentiment {s}: Predicted Rate = {pred_unscaled:.5f}")


decision = "Hedge now" if predicted_rate < live_rate_usdnzd else "Wait"
print(f"Recommendation: {decision}")

with open('hedge_log_usdnzd.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        live_rate_usdnzd,
        predicted_rate,
        decision,
        f"{rmse:.6f}",
        f"{mae:.6f}",
        f"{directional_hits:.2f}%"
    ])


#Plot predictions vs actual 
predicted_scaled = model.predict(X_test)
predicted_padded = np.hstack([predicted_scaled, np.zeros_like(predicted_scaled)])
predicted_unscaled = scaler.inverse_transform(predicted_padded)[:, 0]

plt.figure(figsize=(10, 6))
test_dates = test_df['Date'].values[look_back:]
test_prices = test_df['Price'].values[look_back:]
plt.plot(test_dates, predicted_unscaled, label='Predicted')
plt.plot(test_dates, test_prices, label='Actual')
plt.legend()
plt.title('USD/NZD Prediction vs Actual')
plt.xlabel('Date')
plt.ylabel('Rate')
plt.show()



