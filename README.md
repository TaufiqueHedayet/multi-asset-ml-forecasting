# 📈 Multi-Asset ML Forecasting

An automated machine learning system that pulls live market prices, trains multiple models, stores 7-day forecasts in a database, and displays them in an interactive dashboard.

## 🔴 Live Demo

**[Open the Streamlit dashboard](https://multi-asset-ml-forecasting-cggtuegkr2aiha2pl5ghmj.streamlit.app/)**



## What it does

- Fetches 5 years of daily prices from Yahoo Finance (`yfinance`) for three assets:
  - Crude Oil (`CL=F`)
  - Gold (`GC=F`)
  - Bitcoin (`BTC-USD`)
- Stores actual prices and model predictions in SQLite via SQLAlchemy
- Trains four models per asset and produces a recursive 7-day forecast from each:
  - Gradient Boosting Regressor (GBRT)
  - AdaBoost Regressor
  - Support Vector Regression (SVR)
  - Bidirectional LSTM (BiLSTM)
- Shows actual history and all model forecasts side by side in a Streamlit + Plotly dashboard

## How it works

```
Yahoo Finance ──► actual_prices (SQLite) ──► feature engineering ──► 4 models
                                                                        │
Streamlit dashboard ◄────────── prediction_logs (SQLite) ◄──────────────┘
```

**Features** (all relative to the previous day's price, so they are comparable across assets):
`lag2_ratio`, `lag3_ratio`, `mean3_ratio`, `std3_ratio`, `day_of_week`, `is_weekend`

**Target:** next-day return. Forecasts are built recursively: each predicted price is fed back in to predict the following day, for 7 days ahead.

**Preprocessing:** median imputation and robust scaling in a scikit-learn `Pipeline`.

## Tech stack

Python · pandas · NumPy · scikit-learn · TensorFlow/Keras · yfinance · SQLAlchemy · SQLite · Streamlit · Plotly

## Repository structure

```
├── Gbrt.ipynb               # Data sync, feature engineering, model training, forecast generation
├── app.py                   # Streamlit dashboard
├── forecasting_system.db    # SQLite database (prices + predictions)
├── requirements.txt         # Dashboard dependencies
└── README.md
```

## Run locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

pip install streamlit pandas numpy plotly sqlalchemy
streamlit run app.py
```

To regenerate forecasts, install the training dependencies and run the notebook:

```bash
pip install yfinance scikit-learn tensorflow
jupyter notebook Gbrt.ipynb
```

## Limitations

- Models are trained on all available data and are not yet evaluated against a hold-out set or a naive baseline ("tomorrow = today"). Short-horizon price forecasting is hard, so treat the outputs as a demonstration of the pipeline, not as reliable predictions.
- Forecasts are recursive, so errors compound over the 7-day horizon.
- The BiLSTM currently receives a sequence length of 1.
- The deployed dashboard shows the data in the committed database and does not refresh automatically yet.

## Roadmap

- [ ] Keep prediction history and fill in actual prices and errors once target dates pass
- [ ] Walk-forward validation with a naive baseline and per-model error metrics (MAE, RMSE, MAPE)
- [ ] Scheduled daily updates with GitHub Actions
- [ ] Move code from the notebook into a modular package with typed config (Pydantic)
- [ ] FastAPI service (`/predict`, `/log-actual`)
- [ ] Drift tracking, automatic retraining and alerts
- [ ] Docker, docker-compose and CI with pytest

## Disclaimer

This project is for learning and portfolio purposes only. It is not financial advice.

## Author

**<Taufique Hedayet>** · [LinkedIn](https://linkedin.com/in/TaufiqueHedayet) · [GitHub](https://github.com/TaufiqueHedayet)
