# Dual-Model Trading Simulator

This Streamlit app builds a trading simulator inspired by the uploaded `Practical-Python-for-Algorithmic-Trading` materials.

## What it does
- Downloads market data from Yahoo Finance
- Engineers OHLCV-based features
- Trains **two prediction models**:
  - **Classification model**: predicts `UP` or `DOWN`
  - **Regression model**: predicts next-period return in percent
- Optionally **optimizes model hyperparameters** with `GridSearchCV`
- Optionally **optimizes trading thresholds** for the regression signal using a Sharpe-based search
- Simulates buy/sell decisions using:
  - Classification only
  - Regression only
  - Combined strategy using both models
- Compares strategy equity against buy-and-hold

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Strategy logic
### Combined mode
- **BUY** when:
  - classifier predicts `UP`
  - and regressor predicted return is above the buy threshold
- **SELL** when:
  - classifier predicts `DOWN`
  - or regressor predicted return is below the sell threshold

## Files
- `app.py` - main Streamlit simulator
- `requirements.txt` - dependencies
- `README.md` - overview
