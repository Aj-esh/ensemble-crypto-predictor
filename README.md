# Bitcoin Price Prediction Ensemble Model

This project implements a time-series forecasting model to predict Bitcoin (BTC-USD) prices. It uses an ensemble approach combining N-BEATS and TimesNet, along with a regime detection mechanism to adapt to market conditions.

## Model Architecture

The forecasting system is built on a residual ensemble model, which combines two different neural network architectures:

1.  **N-BEATS**: A deep neural network architecture based on backward and forward residual connections. It serves as the primary forecasting model. The implementation can be found in the [`Nbeats`](https://github.com/Aj-esh/ensemble-crypto-predictor/blob/main/time_series_Nbeats%20(1).ipynb) class.

2.  **TimesNet**: A model designed to capture multi-periodicity in time series data. It analyzes the residuals (the errors from the N-BEATS model's predictions) to provide a corrective forecast. The core components are [`TimesNetModel`](https://github.com/Aj-esh/ensemble-crypto-predictor/blob/main/timesnet_bc.ipynb) and [`TimesBlock`](https://github.com/Aj-esh/ensemble-crypto-predictor/blob/main/timesnet_bc.ipynb).

The final prediction is the sum of the N-BEATS forecast and the TimesNet residual forecast. This is handled by the [`ResidualEnsemble`](https://github.com/Aj-esh/ensemble-crypto-predictor/blob/main/ensemble_timeseres.ipynb) class.

### Regime Detection

A **Hidden Markov Model (HMM)** is used to detect different market regimes (e.g., bull, bear, volatile) based on log returns, volume, and model residuals. This allows for dynamic adjustments to the forecasting strategy. The [`RegimeDetector`](https://github.com/Aj-esh/ensemble-crypto-predictor/blob/main/modeldef.py) class implements this functionality.
