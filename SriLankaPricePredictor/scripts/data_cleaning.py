import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

if not os.path.exists("plots"):
    os.makedirs("plots")

df = pd.read_csv(r"data\commodity_prices.csv")
df["Date"] = pd.to_datetime(df["Date"])           
df.fillna(method="ffill", inplace=True)           

print("Dataset Preview:")
print(df.head())

commodities = ["Rice (Rs/kg)", "Dhal (Rs/kg)", "Wheat Flour (Rs/kg)"]

for commodity in commodities:
    df[f"{commodity}_MA3"] = df[commodity].rolling(window=3).mean()
    df[f"{commodity}_pct_change"] = df[commodity].pct_change() * 100

print("\nMonthly % Change in Rice:")
print(df[["Date", "Rice (Rs/kg)_pct_change"]])

corr_matrix = df[["Rice (Rs/kg)", "Dhal (Rs/kg)", "Wheat Flour (Rs/kg)", "Petrol (Rs/litre)", "Diesel (Rs/litre)"]].corr()
print("\nCorrelation Matrix:")
print(corr_matrix)

def safe_filename(name):
    return "".join(c for c in name if c.isalnum() or c == "_").lower()

plt.figure(figsize=(10,6))
for commodity in commodities:
    plt.plot(df["Date"], df[commodity], marker="o", label=commodity)

plt.title("Sri Lanka Commodity Prices Trend")
plt.xlabel("Date")
plt.ylabel("Price (Rs)")
plt.legend()
plt.grid(True)
plt.savefig(f"plots/commodity_trends.png")
plt.show()

plt.figure(figsize=(10,6))
colors = ["green", "purple", "orange"]
for commodity, color in zip(commodities, colors):
    plt.plot(df["Date"], df[f"{commodity}_MA3"], linestyle="--", marker="o", color=color, label=f"{commodity} 3-Month MA")

plt.title("Rolling Average (3-Month) for Commodities")
plt.xlabel("Date")
plt.ylabel("Price (Rs)")
plt.legend()
plt.grid(True)
plt.savefig("plots/rolling_avg.png")
plt.show()

df["Date_Ordinal"] = df["Date"].map(pd.Timestamp.toordinal)

def predict_linear(df, commodity, months_ahead=2):
    X = df["Date_Ordinal"].values
    y = df[commodity].values
    m, c = np.polyfit(X, y, 1)

    future_dates = pd.date_range(start=df["Date"].max(), periods=months_ahead+1, freq="M")[1:]
    future_ordinals = future_dates.map(pd.Timestamp.toordinal)
    future_preds = m * future_ordinals + c

    print(f"\n Linear Predicted {commodity}:")
    for date, price in zip(future_dates, future_preds):
        print(f"{date.strftime('%Y-%m-%d')}: Rs. {price:.2f}")

    plt.figure(figsize=(10,6))
    plt.scatter(df["Date"], y, color="blue", label="Actual")
    plt.plot(df["Date"], m*X + c, color="red", label="Linear Trend")
    plt.scatter(future_dates, future_preds, color="green", marker="x", s=100, label="Predicted Future")
    plt.title(f"{commodity} Price Prediction (Linear Regression)")
    plt.xlabel("Date")
    plt.ylabel("Price (Rs/kg)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/{safe_filename(commodity)}_linear_prediction.png")
    plt.show()

for commodity in commodities:
    predict_linear(df, commodity)

def predict_polynomial(df, commodity, degree=2, months_ahead=2):
    X = df["Date_Ordinal"].values
    y = df[commodity].values
    coeffs = np.polyfit(X, y, degree)
    poly_func = np.poly1d(coeffs)

    future_dates = pd.date_range(start=df["Date"].max(), periods=months_ahead+1, freq="M")[1:]
    future_ordinals = future_dates.map(pd.Timestamp.toordinal)
    preds = poly_func(future_ordinals)

    print(f"\n Polynomial Predicted {commodity}:")
    for date, price in zip(future_dates, preds):
        print(f"{date.strftime('%Y-%m-%d')}: Rs. {price:.2f}")

    plt.figure(figsize=(10,6))
    plt.scatter(df["Date"], y, color="blue", label="Actual")
    plt.plot(df["Date"], poly_func(X), color="orange", linestyle="--", label=f"Polynomial Trend (deg={degree})")
    plt.scatter(future_dates, preds, color="green", marker="x", s=100, label="Predicted Future")
    plt.title(f"{commodity} Price Prediction (Polynomial Regression)")
    plt.xlabel("Date")
    plt.ylabel("Price (Rs/kg)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/{safe_filename(commodity)}_poly_prediction.png")
    plt.show()

for commodity in commodities:
    predict_polynomial(df, commodity)

for commodity in commodities:
    spikes = df[df[f"{commodity}_pct_change"] > 5]
    if not spikes.empty:
        print(f"\n {commodity} Price Spikes (>5% month-to-month):")
        print(spikes[["Date", commodity, f"{commodity}_pct_change"]])

plt.figure(figsize=(10,6))
plt.plot(df["Date"], df["Rice (Rs/kg)"], marker="o", label="Rice")
plt.plot(df["Date"], df["Dhal (Rs/kg)"], marker="s", label="Dhal")
plt.plot(df["Date"], df["Petrol (Rs/litre)"], marker="^", label="Petrol")
plt.title("Food Prices vs Petrol in Sri Lanka")
plt.xlabel("Date")
plt.ylabel("Price (Rs)")
plt.legend()
plt.grid(True)
plt.savefig("plots/food_vs_petrol.png")
plt.show()
