import joblib
import pandas as pd
from pathlib import Path
import yfinance as yf

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error

# Creates features from DataFram from yf.Ticker().history()
def make_features(ticker: str, days_ahead: int = 1, pred_period: int = 365, max_window: int = 20):
    # Gets Stock Data from yf
    df = yf.Ticker(ticker).history(period=f"{pred_period}d")

    df["Target"] = df["Close"].shift(-days_ahead) 

    # Feature Engineering
    df["Moving_Average"] = df["Close"].rolling(window=max_window).mean()
    df["Volatility"] = df["Close"].rolling(window=max_window).std()

    df = df.dropna() # Drop NaN Values

    y = df["Target"]
    features = ["Close", "Volume", "Moving_Average", "Volatility"]
    X = df[features]

    # Returns X, y, test_features
    return X, y, features

def make_model(ticker: str, days_ahead: int = 1, pred_period: int = 365, max_window: int = 20) -> bool:
    # Get X and y vectors
    X, y, features = make_features(ticker, days_ahead, pred_period, max_window)

    # Make pipeline for training
    # Handles scaling features
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])
    mse = []
    rmse =[]
    # Training model
    split = TimeSeriesSplit()
    for train_idx, test_idx in split.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Fit and predict model
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Calculate mse as a return instead of predicted price
        # Formula = Close(t) / Close (t-1) - 1 , t = days_ahead
        close_t = X_test["Close"].to_numpy() # today's close for each row
        r_true = (y_test.to_numpy() / close_t) - 1 # true next-day return
        r_pred = (y_pred / close_t) - 1     

        mse.append(mean_squared_error(r_true, r_pred))
        rmse.append(root_mean_squared_error(r_true, r_pred))
    
    # Metrics
    mse = sum(mse) / len(mse)
    rmse = sum(rmse) / len(rmse)

    # Fit model and all data
    pipe.fit(X, y)

    # Object to hold information
    info = {
        "pipeline": pipe,
        "features" : features,
        "mse_metric" : mse,
        "rmse_metric" : rmse
    }

    # Make a joblib file to store model
    model_dir = Path(__file__).resolve().parents[1]
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(info, model_dir / f"{ticker}.joblib")

    return True
