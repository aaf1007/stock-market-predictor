import json
import yfinance as yf

stock_symbol = "AAPL"
stock_data = yf.Ticker(stock_symbol)
info = stock_data.info

# Pretty-print in terminal (indented JSON)
print(json.dumps(info, indent=2, default=str))

# Optional: save to file and open in browser
# Uncomment the next 2 lines, then run and open backend/stock_info.json in your browser
with open("stock_info.json", "w") as f:
    json.dump(info, f, indent=2, default=str)

