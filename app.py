
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

# ---------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------

st.set_page_config(
    page_title="MarketMinds",
    page_icon="📈",
    layout="wide"
)

# ---------------------------------
# 2. APPLICATION TITLE
# ---------------------------------

st.title("📈 MarketMinds")
st.subheader("AI-Powered Stock Market Advisor")

st.write(
    "Analyze Indian stocks and predict "
    "the next trading day's closing price."
)

st.divider()

# ---------------------------------
# 3. STOCK SELECTION
# ---------------------------------

stocks = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
    "Infosys": "INFY.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Mahindra & Mahindra": "M&M.NS"
}

selected_stock = st.selectbox(
    "Select a Stock",
    list(stocks.keys())
)

symbol = stocks[selected_stock]

# ---------------------------------
# 4. ANALYZE STOCK
# ---------------------------------

if st.button("Analyze Stock"):

    with st.spinner("Fetching stock data..."):

        stock = yf.Ticker(symbol)
        df = stock.history(period="1y")

    if df.empty:

        st.error("Unable to fetch stock data. Please try again.")

    else:

        st.success("Stock data retrieved successfully!")

        # ---------------------------------
        # 5. CALCULATE TECHNICAL INDICATORS
        # ---------------------------------

        # Simple Moving Averages
        df["SMA_5"] = df["Close"].rolling(window=5).mean()
        df["SMA_20"] = df["Close"].rolling(window=20).mean()

        # Relative Strength Index (RSI)
        delta = df["Close"].diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        rs = avg_gain / avg_loss

        df["RSI"] = 100 - (100 / (1 + rs))

        # ---------------------------------
        # 6. LATEST CLOSING PRICE
        # ---------------------------------

        latest_price = df["Close"].iloc[-1]

        # ---------------------------------
        # 7. LINEAR REGRESSION PREDICTION
        # ---------------------------------

        model_path = f"models/{symbol}.joblib"

        col1, col2 = st.columns(2)

        col1.metric(
            "Latest Closing Price",
            f"₹{latest_price:.2f}"
        )

        if os.path.exists(model_path):

            model_data = joblib.load(model_path)

            model = model_data["model"]
            features = model_data["features"]

            # Get the latest Open, High, Low and Close
            latest_data = df[features].tail(1)

            # Predict next trading day's closing price
            prediction = model.predict(latest_data)[0]

            col2.metric(
                "Predicted Next Closing Price",
                f"₹{prediction:.2f}",
                delta=f"₹{prediction - latest_price:.2f}"
            )

            st.info(
                "Prediction generated using Linear Regression."
            )

        else:

            col2.warning("Model not found.")

            st.warning(
                "Please run train_models.py to create the stock models."
            )

        # ---------------------------------
        # 8. DISPLAY TECHNICAL INDICATORS
        # ---------------------------------

        st.divider()
        st.subheader("Technical Indicators")

        latest_sma5 = df["SMA_5"].iloc[-1]
        latest_sma20 = df["SMA_20"].iloc[-1]
        latest_rsi = df["RSI"].iloc[-1]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "SMA 5",
            f"₹{latest_sma5:.2f}"
            if pd.notna(latest_sma5) else "N/A"
        )

        col2.metric(
            "SMA 20",
            f"₹{latest_sma20:.2f}"
            if pd.notna(latest_sma20) else "N/A"
        )

        col3.metric(
            "RSI",
            f"{latest_rsi:.2f}"
            if pd.notna(latest_rsi) else "N/A"
        )

        # RSI interpretation
        if pd.notna(latest_rsi):

            if latest_rsi > 70:
                st.warning("RSI indicates a potentially overbought condition.")

            elif latest_rsi < 30:
                st.warning("RSI indicates a potentially oversold condition.")

            else:
                st.info("RSI is currently in the neutral range.")

        # ---------------------------------
        # 9. HISTORICAL STOCK PRICE CHART
        # ---------------------------------

        st.divider()
        st.subheader("Historical Stock Price")

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(
            df.index,
            df["Close"],
            label="Closing Price"
        )

        ax.plot(
            df.index,
            df["SMA_5"],
            label="SMA 5"
        )

        ax.plot(
            df.index,
            df["SMA_20"],
            label="SMA 20"
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Price (₹)")
        ax.set_title(selected_stock + " Stock Price")

        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

        # ---------------------------------
        # 10. HISTORICAL DATA TABLE
        # ---------------------------------

        st.divider()
        st.subheader("Historical Stock Data")

        st.dataframe(
            df.tail(10),
            use_container_width=True
        )

# ---------------------------------
# 11. DISCLAIMER
# ---------------------------------

st.divider()

st.caption(
    "MarketMinds is an educational college project. "
    "Predictions are estimates and are not financial advice."
)
