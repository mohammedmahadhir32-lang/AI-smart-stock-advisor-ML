
import pandas as pd
import matplotlib.pyplot as plt

# =====================================
# 1. LOAD STOCK DATASET
# =====================================

df = pd.read_csv("data/RELIANCE.NS.csv")

print("Original Dataset:")
print(df.head())


# =====================================
# 2. DATASET INFORMATION
# =====================================

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())


# =====================================
# 3. DATA TYPE CONVERSION
# =====================================

# Convert Date column into datetime format
df["Date"] = pd.to_datetime(df["Date"])


# =====================================
# 4. DATA QUALITY CHECKS
# =====================================

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Check zero-volume records
print("\nZero Volume Records:")
print((df["Volume"] == 0).sum())

# Display zero-volume records
print("\nZero Volume Details:")
print(df[df["Volume"] == 0][["Date", "Volume"]])


# =====================================
# 5. DATA CLEANING
# =====================================

# Remove records with zero volume
df = df[df["Volume"] > 0]

# Sort records by date
df = df.sort_values("Date")

# Reset row index
df = df.reset_index(drop=True)


# =====================================
# 6. FINAL DATASET CHECK
# =====================================

print("\nCleaned Dataset:")
print(df.head())

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nZero Volume Records Remaining:")
print((df["Volume"] == 0).sum())

# Save cleaned dataset separately
df.to_csv("data/RELIANCE_cleaned.csv", index=False)

print("Cleaned dataset saved successfully!")

# =====================================
# 7. CLOSING PRICE VISUALIZATION
# =====================================

plt.figure(figsize=(12, 5))

plt.plot(df["Date"], df["Close"], label="Closing Price")

plt.title("Reliance Stock Closing Price - 1 Year")
plt.xlabel("Date")
plt.ylabel("Closing Price (INR)")

plt.legend()
plt.grid(True)

plt.show()


# =====================================
# 8. TRADING VOLUME VISUALIZATION
# =====================================

plt.figure(figsize=(12, 5))

plt.bar(df["Date"], df["Volume"], label="Trading Volume")

plt.title("Reliance Stock Trading Volume - 1 Year")
plt.xlabel("Date")
plt.ylabel("Number of Shares Traded")

plt.legend()
plt.grid(True)

plt.show()

# =====================================
# 9. DAILY RETURNS ANALYSIS
# =====================================

# Calculate daily percentage change
df["Daily_Return"] = df["Close"].pct_change() * 100

print("\nDaily Returns:")
print(df[["Date", "Close", "Daily_Return"]].head(10))

# =====================================
# 10. DAILY RETURNS VISUALIZATION
# =====================================

plt.figure(figsize=(12, 5))

plt.plot(df["Date"], df["Daily_Return"],
         label="Daily Returns")

plt.axhline(y=0, linestyle="--")

plt.title("Reliance Daily Returns")
plt.xlabel("Date")
plt.ylabel("Daily Return (%)")

plt.legend()
plt.grid(True)

plt.show()


# =====================================
# 11. FEATURE ENGINEERING - SMA
# =====================================

# Calculate 5-day Simple Moving Average
df["SMA_5"] = df["Close"].rolling(window=5).mean()

# Calculate 20-day Simple Moving Average
df["SMA_20"] = df["Close"].rolling(window=20).mean()

# Display SMA results
print("\nMoving Average Results:")
print(df[["Date", "Close", "SMA_5", "SMA_20"]].tail(10))


# =====================================
# 12. FEATURE ENGINEERING - RSI
# =====================================

# Calculate price difference
delta = df["Close"].diff()

# Separate gains and losses
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

# Calculate average gain and loss over 14 days
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()

# Calculate Relative Strength
rs = avg_gain / avg_loss

# Calculate RSI
df["RSI"] = 100 - (100 / (1 + rs))

# Handle periods with no losses
df.loc[avg_loss == 0, "RSI"] = 100

# Display RSI results
print("\nRSI Results:")
print(df[["Date", "Close", "RSI"]].tail(10))


# =====================================
# 13. RSI VISUALIZATION
# =====================================

plt.figure(figsize=(12, 5))

plt.plot(df["Date"], df["RSI"], label="RSI")

# Overbought level
plt.axhline(y=70, linestyle="--", label="Overbought (70)")

# Oversold level
plt.axhline(y=30, linestyle="--", label="Oversold (30)")

plt.title("Reliance RSI Indicator")
plt.xlabel("Date")
plt.ylabel("RSI Value")

plt.legend()
plt.grid(True)

plt.show()

# VOLUME CHANGE
df["Volume_Change"] = df["Volume"].pct_change() * 100

print("\nVolume Change:")
print(df[["Date", "Volume", "Volume_Change"]].tail(10))

import seaborn as sns

# 14. VOLUME CHANGE
df["Volume_Change"] = df["Volume"].pct_change() * 100

# 15. SAVE FINAL DATASET
df.to_csv("data/RELIANCE_cleaned.csv", index=False)

print("\nFinal dataset saved successfully!")
print(df.tail())

# CORRELATION MATRIX
correlation = df.corr(numeric_only=True)

print("\nCorrelation Matrix:")
print(correlation)


# CORRELATION HEATMAP
plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Stock Data Correlation Heatmap")
plt.show()
