import yfinance as yf
import pandas as pd

# Our five Indian stocks
stocks = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "M&M.NS"
]

# Function to retrieve stock data
def fetch_stock_data(symbol):

    stock = yf.Ticker(symbol)

    data = stock.history(period="1y")

    return data


# Retrieve data for all five stocks
for symbol in stocks:

    data = fetch_stock_data(symbol)

    if data.empty:
        print("No data found:", symbol)
        continue

    data.to_csv(f"data/{symbol}.csv")

    print("\nStock:", symbol)
    print(data.head())