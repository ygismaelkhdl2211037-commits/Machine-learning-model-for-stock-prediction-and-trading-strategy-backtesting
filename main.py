import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
ticker = "AAPL"
data = yf.download(ticker, start="2020-01-01", end="2024-01-01")

# Feature Engineering
data['Return'] = data['Close'].pct_change()
data['MA10'] = data['Close'].rolling(10).mean()
data['MA20'] = data['Close'].rolling(20).mean()

delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

data = data.dropna()

# Target
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
data = data.dropna()

# Train
features = ['Return', 'MA10', 'MA20', 'RSI']
X = data[features]
y = data['Target']

split = int(len(data) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Backtest
data['Prediction'] = model.predict(X)
data['Strategy_Return'] = data['Return'] * data['Prediction']
data['BuyHold_Return'] = data['Return']

data['Strategy_Cum'] = (1 + data['Strategy_Return']).cumprod()
data['BuyHold_Cum'] = (1 + data['BuyHold_Return']).cumprod()

# Plot
plt.figure(figsize=(10,5))
plt.plot(data['Strategy_Cum'], label='Strategy')
plt.plot(data['BuyHold_Cum'], label='Buy & Hold')
plt.legend()
plt.title("Trading Strategy vs Buy & Hold")
plt.grid()

plt.savefig("result.png") 
plt.show()

print("Strategy Return:", data['Strategy_Cum'].iloc[-1])
print("Buy & Hold Return:", data['BuyHold_Cum'].iloc[-1])
