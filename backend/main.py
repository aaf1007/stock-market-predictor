from pathlib import Path
import yfinance as yf
from model.train import linear_regression_factory as factory
import joblib

POPULAR_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V"]


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[38;5;153m"     # pastel sky
    BLUE = "\033[38;5;147m"     # pastel lavender
    GREEN = "\033[38;5;121m"    # pastel mint
    YELLOW = "\033[38;5;223m"   # pastel peach
    RED = "\033[38;5;217m"      # pastel rose
    MAGENTA = "\033[38;5;183m"  # pastel lilac


def style(text: str, color: str) -> str:
    return f"{color}{text}{C.RESET}"


def print_header():
    width = 54
    print(style(f"╔{'═' * width}╗", C.BLUE))
    print(style(f"║{'Welcome'.center(width)}║", C.MAGENTA))
    print(style(f"║{'Stock Market Predictor'.center(width)}║", C.BOLD + C.CYAN))
    print(style(f"║{'Command-line interface'.center(width)}║", C.GREEN))
    print(style(f"╚{'═' * width}╝", C.BLUE))
    print()

def get_popular_stocks():
    """Display popular stocks with their names."""
    print(style("Popular Stocks", C.BOLD + C.CYAN))
    for i, ticker in enumerate(POPULAR_STOCKS):
        print(f"{style(f'{i+1}.) {ticker}', C.YELLOW)} - {yf.Ticker(ticker).info.get('longName')}")
    
def main():
    print_header()

    while True:
        print(style("======================", C.BLUE))
        print(style("1. Predict Stock Price", C.GREEN))
        print(style("2. Get Popular Stocks", C.GREEN))
        print(style("3. Exit", C.GREEN))
        print(style("======================", C.BLUE))
        choice = input(style("Enter your choice: ", C.YELLOW))
        match choice:
            case "1":
            # Ask for stock symbol
                stock_symbol = input(style("Enter stock/ticker symbol(e.g. AAPL or META): ", C.YELLOW)).upper()

                ticker = yf.Ticker(stock_symbol) # Ticker object

                while (yf.Ticker(stock_symbol).info.get("regularMarketPrice") is None):
                    # Stock symbol / Ticker is invalid
                    print(style("Invalid ticker. Try again.", C.RED))
                    stock_symbol = input(style("Enter stock symbol or name (e.g. AAPL or Apple): ", C.YELLOW)).upper()
                    ticker = yf.Ticker(stock_symbol)

                # Path for joblib file
                model_path = Path(__file__).resolve().parent / "model" / f"{stock_symbol}.joblib"

                # If model is not live then create
                if not model_path.exists():
                    train = factory.make_model(stock_symbol)
                    if train:
                        print(style("Model currently training...", C.MAGENTA))
                else:
                    print(style("Model loaded.", C.MAGENTA))

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

                print(
                    style(f"Predicted Price: {future_pred:.2f}", C.CYAN)
                    + " | "
                    + style(f"Change: ${net_change:.2f}", C.YELLOW)
                    + " | "
                    + style(f"Return: {stock_return:.2f}%", C.GREEN)
                    + " | "
                    + style(f"Model MSE: {mse:.5f}", C.BLUE)
                    + " | "
                    + style(f"Model RMSE: {rmse*100:.3f}%", C.MAGENTA)
                    + "\n"
                )

            case "2":
                get_popular_stocks()
            case "3":
                print(style('Thank you!', C.CYAN))
                exit()
            case _:
                print(style("Invalid choice", C.RED))
                continue


if __name__ == "__main__":
    main()