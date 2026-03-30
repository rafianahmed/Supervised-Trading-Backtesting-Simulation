import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    df.dropna(inplace=True)
    return df

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
def create_features(df):
    df = df.copy()

    df['return_1'] = df['Close'].pct_change()
    df['return_5'] = df['Close'].pct_change(5)

    df['volatility_5'] = df['return_1'].rolling(5).std()
    df['volatility_10'] = df['return_1'].rolling(10).std()

    df['sma_5'] = df['Close'].rolling(5).mean()
    df['sma_10'] = df['Close'].rolling(10).mean()

    df['sma_ratio'] = df['Close'] / df['sma_5']

    df.dropna(inplace=True)
    return df

# -------------------------------
# TARGETS
# -------------------------------
def create_targets(df):
    df = df.copy()

    df['future_return'] = df['Close'].pct_change().shift(-1)

    # multiclass labels
    def label(x):
        if x > 0.004:
            return 2  # BUY
        elif x < -0.004:
            return 0  # SELL
        else:
            return 1  # HOLD

    df['class'] = df['future_return'].apply(label)

    df.dropna(inplace=True)
    return df

# -------------------------------
# MAIN APP
# -------------------------------
st.title("Minimal ML Prediction App")

ticker = st.text_input("Ticker", "AAPL")

data = load_data(ticker, "2018-01-01", "2024-01-01")

df = create_features(data)
df = create_targets(df)

features = [
    'return_1', 'return_5',
    'volatility_5', 'volatility_10',
    'sma_ratio'
]

X = df[features]
y_reg = df['future_return']
y_clf = df['class']

# split
split = int(len(df) * 0.8)

X_train, X_test = X[:split], X[split:]
y_reg_train, y_reg_test = y_reg[:split], y_reg[split:]
y_clf_train, y_clf_test = y_clf[:split], y_clf[split:]

# scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# MODELS
# -------------------------------

# Softmax classifier
clf = LogisticRegression(
    solver='lbfgs',
    max_iter=2000,
    class_weight='balanced'
)
clf.fit(X_train_scaled, y_clf_train)

# Gradient descent regressor
reg = SGDRegressor(max_iter=2000, tol=1e-3)
reg.fit(X_train_scaled, y_reg_train)

# -------------------------------
# PREDICTIONS
# -------------------------------
y_clf_pred = clf.predict(X_test_scaled)
y_reg_pred = reg.predict(X_test_scaled)

# -------------------------------
# METRICS
# -------------------------------
accuracy = accuracy_score(y_clf_test, y_clf_pred)
mae = mean_absolute_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))

# -------------------------------
# OUTPUT
# -------------------------------

st.subheader("Model Performance")

st.write(f"Classifier Accuracy: {accuracy:.4f}")
st.write(f"Regression MAE: {mae:.6f}")
st.write(f"Regression RMSE: {rmse:.6f}")

st.subheader("Predictions vs Actual")

results = pd.DataFrame({
    "Actual_Return": y_reg_test.values,
    "Predicted_Return": y_reg_pred,
    "Actual_Class": y_clf_test.values,
    "Predicted_Class": y_clf_pred
})

st.dataframe(results.tail(100))
