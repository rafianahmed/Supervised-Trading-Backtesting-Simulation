# Softmax Dual-Model Trading Simulator

This Streamlit app upgrades the earlier tree-based simulator into a pipeline with:

- feature engineering
- `StandardScaler`
- OLS p-value screening for regression features
- multinomial logistic regression with softmax for `SELL / HOLD / BUY`
- gradient-descent regression with `SGDRegressor`
- validation-based threshold optimization
- backtest metrics, confusion matrix, and trade log

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Main ideas

- **Classification target**: `SELL / HOLD / BUY` using a return threshold
- **Softmax model**: `LogisticRegression(multi_class="multinomial")`
- **Regression model**: `SGDRegressor`
- **Trading engine**:
  - buy when buy probability and predicted return are strong enough
  - sell when sell probability is high or predicted return is weak enough

## Notes

- OLS significance filtering is used as a practical feature screen for the regression target.
- For the softmax model, confidence thresholds are more important than raw accuracy.
- This is a research simulator, not live trading advice.
