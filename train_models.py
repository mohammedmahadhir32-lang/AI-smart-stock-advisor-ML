
import pandas as pd
import joblib
import os

from sklearn.linear_model import LinearRegression

# Create models folder if it does not exist
os.makedirs("models", exist_ok=True)

# Five stocks
stocks = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "M&M.NS"
]

# Features used by Linear Regression
features = ["Open", "High", "Low", "Close"]

# Train a separate model for each stock
for symbol in stocks:

    print("\nTraining model for:", symbol)

    # Load stock data
    df = pd.read_csv(f"data/{symbol}.csv")

    # Create next trading day's closing price as target
    df["Target"] = df["Close"].shift(-1)

    # Prepare features and target
    df = df.dropna(subset=features + ["Target"])

    X = df[features]
    y = df["Target"]

    # Create and train Linear Regression model
    model = LinearRegression()
    model.fit(X, y)

    # Store model and feature names
    model_data = {
        "model": model,
        "features": features
    }

    # Save model
    joblib.dump(model_data, f"models/{symbol}.joblib")

    print("Model saved successfully:", symbol)

print("\nAll stock models trained successfully!")
