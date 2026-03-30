# Trading Simulator

This Streamlit app upgrades the earlier tree-based simulator into a pipeline with:

- feature engineering
- `StandardScaler`
- OLS p-value screening for regression features
- multinomial logistic regression with softmax for `SELL / HOLD / BUY`
- gradient-descent regression with `SGDRegressor`
- validation-based threshold optimization
- backtest metrics, confusion matrix, and trade log

## Main ideas

- **Classification target**: `SELL / HOLD / BUY` using a return threshold
- **Softmax model**: `LogisticRegression(multi_class="multinomial")`
- **Regression model**: `SGDRegressor`
- **Trading engine**:
  - buy when buy probability and predicted return are strong enough
  - sell when sell probability is high or predicted return is weak enough

