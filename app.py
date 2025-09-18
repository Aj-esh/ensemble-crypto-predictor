import os
import numpy as np

import torch

import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from modeldef import ResidualEnsemble, RegimeDetector, create_time_features, create_sequences

st.set_page_config(page_title="Time Series Forecasting Ensemble", layout="wide")

st.sidebar.title("Time Series Forecasting Ensemble")
st.sidebar.write("Ensemble of N-BEATS & TimesNet")
st.sidebar.write("Regime detection with HMM")

st.subheader("data")
ticker = "BTC-USD"
btc_data = yf.download(ticker, start="2014-01-01", end="2024-01-01", auto_adjust=False)
btc_data.reset_index(inplace=True)

st.write(f"loaded {len(btc_data)} rows of {ticker} data")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nbeats_path = os.path.join("weights", "best_nbeats_model.pth")
timesnet_path = os.path.join("weights", "best_timesnet_model.pth")

# Configs
nbeats_configs = {
    "input_size": 14,
    "hidden_size": 64,
    "output_size": 1,
    "n_layers": 4,
    "n_stacks": 3,
}
timesnet_configs = type("Configs", (), {})()
timesnet_configs.task_name = 'long_term_forecast'
timesnet_configs.seq_len = 14
timesnet_configs.label_len = 0
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

scaler = StandardScaler()

price_data = btc_data[['Close']].values
# Create time features
time_features = create_time_features(btc_data, time_col='Date', freq='d')

# Fit only on the training data to avoid data leakage
train_size = int(len(price_data) * 0.8)
scaler.fit(price_data[:train_size])
scaled_data = scaler.transform(price_data)

X, y, X_mark = create_sequences(scaled_data, time_features, timesnet_configs.seq_len, timesnet_configs.pred_len)
# Split Data
X_train, X_test, y_train, y_test, X_mark_train, X_mark_test = train_test_split(
    X, y, X_mark, test_size=0.2, shuffle=False # Time series data should not be shuffled
)

# Convert all data to tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
X_mark = torch.tensor(X_mark, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
X_mark_test = torch.tensor(X_mark_test, dtype=torch.float32)

ensemble = ResidualEnsemble(
    nbeats_configs, nbeats_path, timesnet_configs, timesnet_path
).to(device)

criterion = torch.nn.MSELoss()

ensemble.eval()
with torch.inference_mode():
    # Use the full dataset for generating predictions for the plot
    full_X = X.to(device)
    full_X_mark = X_mark.to(device)

    outputs = ensemble(full_X, full_X_mark)

    # Use the test set for calculating loss/metrics
    test_outputs = ensemble(X_test.to(device), X_mark_test.to(device))
    loss_test = criterion(test_outputs, y_test.to(device))
    test_loss = loss_test.item()

# ADD: inverse transform full-sequence predictions for predictions_df
preds_scaled = outputs.cpu().numpy()
preds_original_scale = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

# NEW: inverse transform test set actual & predictions
actual_original_scale = scaler.inverse_transform(
    y_test.cpu().numpy().reshape(-1, 1)
)
predicted_original_scale = scaler.inverse_transform(
    test_outputs.cpu().numpy().reshape(-1, 1)
)

# Align predictions with the corresponding dates in the original dataframe
prediction_start_index = timesnet_configs.seq_len
predictions_df = btc_data.iloc[prediction_start_index : prediction_start_index + len(preds_original_scale)].copy()
predictions_df["Prediction"] = preds_original_scale

# Calculate residuals
predictions_df["Residual"] = predictions_df["Close"].values.flatten() - predictions_df["Prediction"].values.flatten()

# Calculate log returns for Regime Detection
predictions_df['logreturns'] = np.log(predictions_df['Close'] / predictions_df['Close'].shift(1))
# Use 'Volume' from yfinance, which is named 'Volume' not 'volume'
predictions_df['volume'] = predictions_df['Volume']


# Regime Detection
detector = RegimeDetector(n_states=3)
fittable_df = predictions_df.dropna()
detector.fit(fittable_df)
predictions_df["Regime"] = [detector.detect(np.array([r, v, res]))[0]
                      for r, v, res in zip(
                          predictions_df['logreturns'].fillna(0),
                          predictions_df['volume'].fillna(0),
                          predictions_df["Residual"].fillna(0)
                      )]

# REPLACED PLOT 1
st.subheader("Bitcoin Price Prediction (Test Set)")
fig = go.Figure()
fig.add_trace(go.Scatter(
    y=actual_original_scale.flatten(),
    mode='lines',
    name='Actual',
    line=dict(color='blue')
))
fig.add_trace(go.Scatter(
    y=predicted_original_scale.flatten(),
    mode='lines',
    name='Predicted',
    line=dict(color='orange')
))
fig.update_layout(
    title='Bitcoin Price Prediction (Test Set)',
    xaxis_title='Time (days)',
    yaxis_title='Price (USD)',
    legend=dict(x=0, y=1),
    template='plotly_white',
    height=600,
    width=1000
)
st.plotly_chart(fig, use_container_width=True)

# REPLACED PLOT 2 (only regime change points as dots)
st.subheader("Regime Changes")
# detect change points
regime_change_mask = predictions_df['Regime'].ne(predictions_df['Regime'].shift(1))
regime_change_df = predictions_df[regime_change_mask]

color_map = {0: "lime", 1: "red", 2: "orange"}

plot2 = go.Figure()
for state, df_state in regime_change_df.groupby("Regime"):
    plot2.add_trace(go.Scatter(
        x=df_state["Date"],
        y=df_state["Regime"],
        mode="markers",
        marker=dict(size=10, symbol="circle", color=color_map.get(state, "purple"), line=dict(width=1, color="#333")),
        name=f"Regime {state}"
    ))

plot2.update_layout(
    title="Regime Change Points",
    xaxis_title="Date",
    yaxis_title="Regime",
    template="plotly_dark",
    yaxis=dict(tickmode="array", tickvals=sorted(predictions_df['Regime'].unique())),
    legend=dict(orientation="h", y=1.1)
)
st.plotly_chart(plot2, use_container_width=True)

# Residuals
st.subheader("Residuals")
st.line_chart(predictions_df.set_index("Date")["Residual"])

# Metrics
st.subheader("Metrics")
rmse = np.sqrt(mean_squared_error(predictions_df["Close"], predictions_df["Prediction"]))
mape = mean_absolute_percentage_error(predictions_df["Close"], predictions_df["Prediction"])

col1, col2, col3, col4 = st.columns(4)
col1.metric("RMSE", f"{rmse:.2f}")
col2.metric("MAPE", f"{mape*100:.2f}%")
col3.metric("Latest Price", predictions_df['Close'].iloc[-1])
col4.metric("Latest Regime", predictions_df['Regime'].iloc[-1])