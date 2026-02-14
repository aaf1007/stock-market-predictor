from pathlib import Path
import yfinance as yf
from model.train import linear_regression_factory as factory
import joblib

POPULAR_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V"]

def get_popular_stocks():
    """Display popular stocks with their names."""
    for i, ticker in enumerate(POPULAR_STOCKS):
        print(f"{i+1}.) {ticker} - {yf.Ticker(ticker).info.get('longName')}")
    
def main():
    print("Stock Price Predictor")
    print("=====================")

    while True:
        print("1. Predict Stock Price")
        print("2. Get Popular Stocks")
        print("3. Exit")
        print("=====================")
        choice = input("Enter your choice: ")
        match choice:
            case "1":
            # Ask for stock symbol
                stock_symbol = input("Enter stock symbol or name (e.g. AAPL or Apple): ").upper()

                ticker = yf.Ticker(stock_symbol) # Ticker object

                while (yf.Ticker(stock_symbol).info.get("regularMarketPrice") is None):
                    # Stock symbol / Ticker is invalid
                    stock_symbol = input("Enter stock symbol or name (e.g. AAPL or Apple): ").upper()
                    ticker = yf.Ticker(stock_symbol)

                # Path for joblib file
                model_path = Path(__file__).resolve().parent / "model" / f"{stock_symbol}.joblib"

                # If model is not live then create
                if not model_path.exists():
                    train = factory.make_model(stock_symbol)
                    print(f"Is training? {train}")

                job = joblib.load(model_path)
                model = job["pipeline"]
                mse = job["mse_metric"]
                rmse = job["rmse_metric"]
                features = job["features"]

                stock = ticker.history(period="365d")
                stock["Target"] = stock["Close"].shift(-1) / stock["Close"]
                stock["Moving_Average"] = stock["Close"].rolling(window=20).mean()
                stock["Volatility"] = stock["Close"].rolling(window=20).std()

                stock = stock.dropna() # Drop NaN values

                stock_pred = stock[features]
                # Make prediction
                pred = model.predict(stock_pred)

                # Stock Metrics
                future_pred = float(pred[-1])
                net_change = float(future_pred - pred[-2])
                stock_return = float((net_change / pred[-2]) * 100)

                print(f"Predicted Price: {future_pred:.2f} | " + 
                    f"Change: ${net_change:.2f} | " +
                    f"Return: {stock_return:.2f}% | " +
                    f"Model MSE: {mse:.5f} | " +
                    f"Model RMSE: {rmse*100:.3f}%"
                    )


            case "2":
                get_popular_stocks()
            case "3":
                print('Thank you!')
                exit()
            case _:
                print("Invalid choice")
                continue

        print("\n")

main()