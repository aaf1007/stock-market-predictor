# Stock Market Predictor

CLI-based stock prediction project using `yfinance` + `scikit-learn`.

The current backend architecture trains and stores **one linear-regression model per ticker** using `backend/model/train/linear_regression_factory.py`. Models are cached as `.joblib` files and loaded on demand by `backend/main.py`.

## Current Model Design

- Model type: `LinearRegression` inside a `Pipeline(StandardScaler -> LinearRegression)`
- Data source: `yfinance`
- Forecast target: next-day close (`Close[t+1]`)
- Features:
  - `Close`
  - `Volume`
  - `Moving_Average` (20-day rolling mean)
  - `Volatility` (20-day rolling std)
- Validation: `TimeSeriesSplit`
- Stored artifact per ticker: `backend/model/<TICKER>.joblib`

## Project Structure

```text
backend/
  main.py                               # CLI app
  requirements.txt
  model/
    AAPL.joblib                         # Example trained model artifact
    T.TO.joblib                         # Example trained model artifact
    train/
      linear_regression_factory.py      # Per-ticker training + persistence
      linear_regression_pred.py         # Older/general training script
      ticker_universe.tsv
      stock_info.json
```

## Setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

From `backend/`:

```bash
python main.py
```

CLI menu:

1. Predict stock price
2. Get popular stocks
3. Exit

If a ticker model does not exist yet, the app trains it first via the factory and then saves it under `backend/model/`.

## Performance Snapshot

Metrics below are read from current saved artifacts in this repo (`backend/model/*.joblib`).  
Metric definition in the factory is based on **next-day return error** computed from predicted vs actual close.

| Ticker | MSE | RMSE | RMSE (%) |
|---|---:|---:|---:|
| `AAPL` | 0.0005352936 | 0.0209927145 | 2.099% |
| `T.TO` | 0.0001424114 | 0.0117423143 | 1.174% |
| **Average (2 models)** | **0.0003388525** | **0.0163675144** | **1.637%** |

## Pros and Cons (Current Approach)

### Pros

- Simple and fast to train/infer.
- Per-ticker model storage makes repeated predictions efficient.
- Time-series-aware split avoids random-shuffle leakage.
- Pipeline keeps preprocessing consistent with training.

### Cons

- Linear model can underfit non-linear market behavior.
- Feature set is small (only price/volume + 2 rolling stats).
- Limited robustness to market regime shifts and sudden events.
- Metrics are from short local samples unless you retrain broadly.
- No confidence intervals, only point predictions.

## Improvements

See `MODEL_IMPROVEMENTS.md` for a prioritized improvement roadmap.

