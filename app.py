import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

from modeldef import NBeatsModel, TimesNetModel, ResidualEnsemble, RegimeDetector

st.set_page_config(page_title="Time Series Forecasting Ensemble", layout="wide")

st.sidebar.title("Time Series Forecasting Ensemble")
st.sidebar.write("Ensemble of N-BEATS & TimesNet")
st.sidebar.write("Regime detection with HMM")

st.subheader("data")
ticker = "BTC-USD"
btc_data = yf.download(ticker, start="2014-01-01", end="2024-01-01")
btc_data.reset_index(inplace=True)

st.write(f"loaded {len(btc_data)} rows of {ticker} data")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nbeats_path = os.path.join("models", "nbeats.pth")
timesnet_path = os.path.join("models", "timesnet.pth")

# Configs
nbeats_configs = {
    "input_size": 14,
    "hidden_size": 64,
    "output_size": 1,
    "n_layers": 4,
    "n_stacks": 3,
}
timesnet_configs = type("Configs", (), {})()
timesnet_configs.seq_len = 14
timesnet_configs.pred_len = 1
timesnet_configs.enc_in = 1
timesnet_configs.c_out = 1
timesnet_configs.e_layers = 1
timesnet_configs.d_model = 32
timesnet_configs.d_ff = 64
timesnet_configs.num_kernels = 3
timesnet_configs.embed = "timeF"
timesnet_configs.freq = "d"
timesnet_configs.dropout = 0.05
timesnet_configs.top_k = 3

ensemble = ResidualEnsemble(
    nbeats_configs, nbeats_path, timesnet_configs, timesnet_path
).to(device)

# Inference
close_prices = btc_data["Close"].values
X = torch.tensor(close_prices[:-1], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
x_mark = torch.zeros_like(X).to(device)  # placeholder if you have time features

with torch.inference_mode():
    preds = []
    for i in range(len(close_prices) - timesnet_configs.seq_len):
        x_enc = X[:, i : i + timesnet_configs.seq_len, :]
        x_mark_enc = x_mark[:, i : i + timesnet_configs.seq_len, :]
        y_pred = ensemble(x_enc, x_mark_enc)
        preds.append(y_pred.cpu().numpy().flatten()[0])

# Align predictions
preds = np.array(preds)
btc_data = btc_data.iloc[timesnet_configs.seq_len:].copy()
btc_data["Prediction"] = preds
btc_data["Residual"] = btc_data["Close"] - btc_data["Prediction"]

# Regime Detection
detector = RegimeDetector(n_states=3)
detector.fit(btc_data)
btc_data["Regime"] = [detector.detect(np.array([r, v, res]))[0]
                      for r, v, res in zip(
                          np.log(btc_data["Close"]).diff().fillna(0),
                          btc_data["Close"].pct_change().rolling(14).std().fillna(0),
                          btc_data["Residual"]
                      )]

# Plotting
st.subheader("BTC Forecast vs Actual")

fig = go.Figure()

# Actual prices
fig.add_trace(go.Scatter(
    x=btc_data["Date"], y=btc_data["Close"],
    mode="lines", name="Actual Price"
))

# Predictions
fig.add_trace(go.Scatter(
    x=btc_data["Date"], y=btc_data["Prediction"],
    mode="lines", name="Ensemble Prediction"
))

# Regime shading
colors = {0: "green", 1: "red", 2: "orange"}
for state, color in colors.items():
    regime_dates = btc_data[btc_data["Regime"] == state]
    if not regime_dates.empty:
        fig.add_trace(go.Scatter(
            x=regime_dates["Date"], y=regime_dates["Close"],
            mode="markers", marker=dict(color=color, size=4),
            name=f"Regime {state}"
        ))

fig.update_layout(
    title="BTC Price Forecast with Regimes",
    xaxis_title="Date", yaxis_title="Price (USD)",
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

# Residuals
st.subheader("Residuals")
st.line_chart(btc_data.set_index("Date")["Residual"])

# Metrics
st.subheader("Metrics")
rmse = np.sqrt(mean_squared_error(btc_data["Close"], btc_data["Prediction"]))
mape = mean_absolute_percentage_error(btc_data["Close"], btc_data["Prediction"])

col1, col2, col3, col4 = st.columns(4)
col1.metric("RMSE", f"{rmse:.2f}")
col2.metric("MAPE", f"{mape*100:.2f}%")
col3.metric("Latest Price", f"{btc_data['Close'].iloc[-1]:,.2f}")
col4.metric("Latest Regime", f"{btc_data['Regime'].iloc[-1]}")