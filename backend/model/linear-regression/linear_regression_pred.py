
import yfinance as yf
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error

# This is as a general model that is trained on pre-determined stocks
def predict_price(ticker_symbol: str, days_ahead: int = 1, period_num: int = 60, max_period: int = 20) -> float:
    # Get DataFrame dataset
    tsv_path = Path(__file__).resolve().parent / "ticker_universe.tsv"
    df = pd.read_csv(tsv_path, sep='\t')
    training_stocks = df["ticker"].dropna().tolist()

    stocks = build_data_set(training_stocks, days_ahead, period_num, max_period)
    stocks.to_csv("current_data_set_sample.tsv", sep='\t', index=False)
    # Error Handling: empty dataset
    if stocks.empty:
        raise ValueError("Data set is empty")

    # Build Linear Regression Params
    X = stocks[["Close", "Volume", "Moving_Average", "Volatility"]]
    y = stocks["Target"]

    # Split data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Test the training model
    y_pred = model.predict(X_test)

    # Calculate the mean squared error for training
    mse = mean_squared_error(y_test, y_pred)

    # Actual Pred
    # Get returned value from model from current days data
    latest_price_df = yf.Ticker(ticker_symbol).history(period=f"{period_num}d")
    latest_price_df["Target"] = latest_price_df["Close"].shift(-days_ahead) # Predicted Days

    # Features Engineering
    latest_price_df["Moving_Average"] = latest_price_df["Close"].rolling(window=max_period).mean()
    latest_price_df["Volatility"] = latest_price_df["Close"].rolling(window=max_period).std()

    latest_price_df = latest_price_df.dropna() # Drop NaN vals

    latest_price_features = latest_price_df[["Close", "Volume", "Moving_Average", "Volatility"]].iloc[[-1]]
    ret_prediction = model.predict(latest_price_features)

    return round(ret_prediction[0], 2), round(mse, 2)


def get_last_price(ticker_symbol) -> float:
    stock_symbol = ticker_symbol
    stock_data = yf.Ticker(stock_symbol)

    # Fetch price data for training the model
    price = stock_data.history(period="365d") # gets 365 days of data

    return round(price["Close"].iloc[-1], 2), price.index[-1].date()

# Returns a dataset of TRAINING_STOCKS for training
def build_data_set(stocks: list, pred_days: int, period: int = 60, max_window: int = 20) -> pd.DataFrame:
    data_set = []

    for ticker in stocks:
        ticker_data = yf.Ticker(ticker)

        price = ticker_data.history(period=f"{period}d") # Get n days

        price["Target"] = price["Close"].shift(-pred_days) / price["Close"] - 1# Predicted Days

        # Features Engineering
        # TODO implement more features
        price["Moving_Average"] = price["Close"].rolling(window=max_window).mean() # 60 day moving average
        price["Volatility"] = price["Close"].rolling(window=max_window).std() # 60 day rolling standard deviation

        price = price.dropna() # Drop NaN vals
        
        data_set.append(price) # Append to data_set

    # Makes list of DataFrames into one combined
    data_set = pd.concat(data_set, axis=0, ignore_index=False)

    return data_set


try:
    period, window = 365*3, 20
    pred, err = predict_price("KO",1, period, window)
    print(f"Prediction: {pred} Training Error: {err} Period: {period}")
except Exception as err:
    print(err)
