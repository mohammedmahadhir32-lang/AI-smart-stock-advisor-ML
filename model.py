
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


# 1. LOAD DATASET

df = pd.read_csv("data/RELIANCE_cleaned.csv")

print("\nDataset Loaded Successfully!")


# 2. INSPECT STOCK DATA AROUND NOVEMBER 2

print("\nStock Data Around November 2:")

print(
    df.iloc[238:247][
        [
            "Date", "Open", "High", "Low",
            "Close", "Volume", "Dividends", "Stock Splits"
        ]
    ].to_string(index=False)
)


# 3. CHECK UNUSUAL DAILY PRICE MOVEMENTS

df["Daily_Return_Percent"] = df["Close"].pct_change() * 100

print("\nLargest Daily Price Movements:")

unusual_days = df.loc[
    df["Daily_Return_Percent"].abs().nlargest(10).index,
    ["Date", "Close", "Daily_Return_Percent"]
]

print(unusual_days)


# 4. CREATE NEXT-DAY TARGET

df["Target"] = df["Close"].shift(-1)


# 5. SELECT FOUR IMPORTANT FEATURES

features = [
    "Open",
    "High",
    "Low",
    "Close"
]


# 6. PREPARE X AND y

dataset = df[features + ["Target"]].dropna()

X = dataset[features]
y = dataset["Target"]

print("\nFinal ML Dataset:")
print(dataset.tail())

print("\nDataset Shape:", dataset.shape)


# 7. CHRONOLOGICAL TRAIN-TEST SPLIT

split_index = int(len(X) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining Features:", X_train.shape)
print("Testing Features:", X_test.shape)


# 8. CREATE MODELS

models = {

    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=3,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )

}


# 9. TRAIN AND EVALUATE MODELS

comparison_results = []
predictions_dict = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    print(f"\n{name} Model Trained Successfully!")

    train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)

    predictions_dict[name] = y_pred

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, y_pred)

    mae = mean_absolute_error(y_test, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )

    print(f"\n{name} Evaluation:")

    print("Training R2:", round(train_r2, 4))
    print("Testing R2:", round(test_r2, 4))
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))

    comparison_results.append({
        "Model": name,
        "Training R2": round(train_r2, 4),
        "Testing R2": round(test_r2, 4),
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2)
    })


# 10. FEATURE IMPORTANCE

lr_model = models["Linear Regression"]

importance = permutation_importance(
    lr_model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="neg_mean_absolute_error"
)

feature_importance = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": importance.importances_mean
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE:")
print(feature_importance.to_string(index=False))


# FEATURE IMPORTANCE GRAPH

plt.figure(figsize=(9, 5))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Permutation Importance")
plt.ylabel("Features")
plt.title("Linear Regression - Feature Importance")

plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# 11. BASELINE MODEL

baseline_pred = X_test["Close"].values

predictions_dict["Baseline"] = baseline_pred

baseline_r2 = r2_score(y_test, baseline_pred)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_pred
)

baseline_rmse = np.sqrt(
    mean_squared_error(y_test, baseline_pred)
)

print("\nBaseline Evaluation:")

print("Baseline R2:", round(baseline_r2, 4))
print("Baseline MAE:", round(baseline_mae, 2))
print("Baseline RMSE:", round(baseline_rmse, 2))

comparison_results.append({
    "Model": "Baseline",
    "Training R2": np.nan,
    "Testing R2": round(baseline_r2, 4),
    "MAE": round(baseline_mae, 2),
    "RMSE": round(baseline_rmse, 2)
})


# 12. FINAL MODEL COMPARISON

comparison_df = pd.DataFrame(comparison_results)

print("\n====================================")
print("FINAL MODEL COMPARISON")
print("====================================")

print(comparison_df.to_string(index=False))


# 13. SELECT BEST MODEL BASED ON MAE

best_model = comparison_df.loc[
    comparison_df["MAE"].idxmin()
]

print("\nBest Model Based on MAE:")
print(best_model["Model"])

print("Best MAE:", best_model["MAE"])


# 14. ACTUAL VS PREDICTED GRAPH

plt.figure(figsize=(12, 6))

plt.plot(
    y_test.values,
    label="Actual Price",
    linewidth=2
)

for name, prediction in predictions_dict.items():

    plt.plot(
        prediction,
        label=name,
        linestyle="--"
    )

plt.title("Reliance - Four Feature Model Comparison")

plt.xlabel("Test Data Observations")
plt.ylabel("Stock Price (₹)")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()


# 15. FINAL ACTUAL VS PREDICTED TABLE

final_results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Linear Regression": predictions_dict["Linear Regression"],
    "Random Forest": predictions_dict["Random Forest"],
    "Baseline": predictions_dict["Baseline"]
})

print("\nFINAL ACTUAL VS PREDICTED PRICES:")

print(final_results.head(15).round(2))


import joblib

# Create the final Linear Regression model
final_model = LinearRegression()

# Train using all available data
final_model.fit(X, y)

# Store model and feature names together
model_data = {
    "model": final_model,
    "features": features
}

# Save the model
joblib.dump(model_data, "marketminds_model.joblib")

print("\nLinear Regression Model Saved Successfully!")

