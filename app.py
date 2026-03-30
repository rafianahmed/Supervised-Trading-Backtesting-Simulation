import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title='Dual-Model Trading Simulator', layout='wide')


def load_data(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    keep = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
    df = df[keep].copy()
    df = df.dropna()
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['Return_1d'] = out['Close'].pct_change() * 100
    out['Return_3d'] = out['Close'].pct_change(3) * 100
    out['Return_5d'] = out['Close'].pct_change(5) * 100
    out['Range_Pct'] = ((out['High'] - out['Low']) / out['Close'].replace(0, np.nan)) * 100
    out['Gap_Pct'] = ((out['Open'] - out['Close'].shift(1)) / out['Close'].shift(1).replace(0, np.nan)) * 100
    out['Volume_Change_Pct'] = out['Volume'].pct_change() * 100
    out['SMA_5'] = out['Close'].rolling(5).mean()
    out['SMA_10'] = out['Close'].rolling(10).mean()
    out['SMA_20'] = out['Close'].rolling(20).mean()
    out['SMA5_Ratio'] = (out['Close'] / out['SMA_5'] - 1) * 100
    out['SMA10_Ratio'] = (out['Close'] / out['SMA_10'] - 1) * 100
    out['SMA20_Ratio'] = (out['Close'] / out['SMA_20'] - 1) * 100
    out['Volatility_5'] = out['Return_1d'].rolling(5).std()
    out['Volatility_10'] = out['Return_1d'].rolling(10).std()

    out['Target_Reg'] = out['Close'].shift(-1) / out['Close'] - 1
    out['Target_Reg'] = out['Target_Reg'] * 100
    out['Target_Cls'] = np.where(out['Target_Reg'] > 0, 'UP', 'DOWN')

    out = out.dropna().copy()
    return out


def split_data(df: pd.DataFrame, train_ratio: float):
    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


def get_feature_columns(df: pd.DataFrame):
    return [c for c in df.columns if c not in ['Target_Reg', 'Target_Cls']]


def train_models(train_df: pd.DataFrame, optimize_models: bool):
    features = get_feature_columns(train_df)
    X_train = train_df[features]
    y_cls = train_df['Target_Cls']
    y_reg = train_df['Target_Reg']

    if optimize_models and len(train_df) >= 120:
        cls = GridSearchCV(
            DecisionTreeClassifier(random_state=42),
            param_grid={
                'max_depth': [3, 5, 8, 12, 15],
                'min_samples_leaf': [1, 3, 5, 10],
            },
            cv=3,
            n_jobs=-1,
            scoring='accuracy'
        )
        reg = GridSearchCV(
            DecisionTreeRegressor(random_state=42),
            param_grid={
                'max_depth': [3, 5, 8, 12, 15],
                'min_samples_leaf': [1, 3, 5, 10],
            },
            cv=3,
            n_jobs=-1,
            scoring='neg_mean_absolute_error'
        )
        cls.fit(X_train, y_cls)
        reg.fit(X_train, y_reg)
        cls_model = cls.best_estimator_
        reg_model = reg.best_estimator_
        cls_params = cls.best_params_
        reg_params = reg.best_params_
    else:
        cls_model = DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=42)
        reg_model = DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, random_state=42)
        cls_model.fit(X_train, y_cls)
        reg_model.fit(X_train, y_reg)
        cls_params = {'max_depth': 8, 'min_samples_leaf': 5}
        reg_params = {'max_depth': 8, 'min_samples_leaf': 5}

    if optimize_models and len(train_df) >= 120:
        # GridSearchCV already fitted, but best_estimator_ may be already fit; fit again for clarity.
        cls_model.fit(X_train, y_cls)
        reg_model.fit(X_train, y_reg)

    return cls_model, reg_model, cls_params, reg_params, features


def add_predictions(df: pd.DataFrame, cls_model, reg_model, features: list[str]) -> pd.DataFrame:
    out = df.copy()
    X = out[features]
    out['Pred_Cls'] = cls_model.predict(X)
    out['Pred_Reg'] = reg_model.predict(X)
    return out


def simulate_strategy(df: pd.DataFrame, buy_threshold: float, sell_threshold: float,
                      strategy_mode: str, initial_cash: float, fee: float) -> tuple[pd.DataFrame, dict]:
    sim = df.copy()
    cash = float(initial_cash)
    shares = 0.0
    position = 0
    equity_curve = []
    actions = []
    positions = []

    for _, row in sim.iterrows():
        price = float(row['Close'])
        pred_cls = row['Pred_Cls']
        pred_reg = float(row['Pred_Reg'])

        buy_signal = False
        sell_signal = False

        if strategy_mode == 'Classification only':
            buy_signal = pred_cls == 'UP' and position == 0
            sell_signal = pred_cls == 'DOWN' and position == 1
        elif strategy_mode == 'Regression only':
            buy_signal = pred_reg >= buy_threshold and position == 0
            sell_signal = pred_reg <= sell_threshold and position == 1
        else:
            buy_signal = pred_cls == 'UP' and pred_reg >= buy_threshold and position == 0
            sell_signal = (pred_cls == 'DOWN' or pred_reg <= sell_threshold) and position == 1

        action = 'HOLD'
        if buy_signal:
            shares = (cash * (1 - fee)) / price if price > 0 else 0.0
            cash = 0.0
            position = 1
            action = 'BUY'
        elif sell_signal:
            cash = shares * price * (1 - fee)
            shares = 0.0
            position = 0
            action = 'SELL'

        equity = cash + shares * price
        equity_curve.append(equity)
        actions.append(action)
        positions.append(position)

    sim['Action'] = actions
    sim['Position'] = positions
    sim['Equity'] = equity_curve
    sim['Strategy_Return'] = sim['Equity'].pct_change().fillna(0)
    sim['BuyHold_Equity'] = initial_cash * (sim['Close'] / sim['Close'].iloc[0])
    sim['BuyHold_Return'] = sim['BuyHold_Equity'].pct_change().fillna(0)

    total_return = (sim['Equity'].iloc[-1] / initial_cash - 1) * 100
    bh_return = (sim['BuyHold_Equity'].iloc[-1] / initial_cash - 1) * 100
    vol = sim['Strategy_Return'].std()
    sharpe = (sim['Strategy_Return'].mean() / vol * np.sqrt(252)) if vol not in [0, np.nan] and pd.notna(vol) else 0.0
    max_dd = ((sim['Equity'] / sim['Equity'].cummax()) - 1).min() * 100
    trades = int((sim['Action'].isin(['BUY', 'SELL'])).sum())

    stats = {
        'Final Equity': float(sim['Equity'].iloc[-1]),
        'Strategy Return [%]': float(total_return),
        'Buy & Hold Return [%]': float(bh_return),
        'Sharpe': float(sharpe),
        'Max Drawdown [%]': float(max_dd),
        'Trades': trades,
    }
    return sim, stats


def optimize_signal_thresholds(train_pred_df: pd.DataFrame, strategy_mode: str,
                               initial_cash: float, fee: float):
    if strategy_mode == 'Classification only':
        return None, None, {'best_score': None}

    buy_grid = np.arange(0.0, 2.6, 0.25)
    sell_grid = np.arange(-2.5, 0.1, 0.25)
    best_score = -np.inf
    best_pair = (0.5, -0.5)

    for buy_t in buy_grid:
        for sell_t in sell_grid:
            if sell_t >= buy_t:
                continue
            _, stats = simulate_strategy(train_pred_df, buy_t, sell_t, strategy_mode, initial_cash, fee)
            score = stats['Sharpe']
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_pair = (float(buy_t), float(sell_t))

    return best_pair[0], best_pair[1], {'best_score': float(best_score)}


def plot_equity(sim: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sim.index, y=sim['Equity'], mode='lines', name='Strategy Equity'))
    fig.add_trace(go.Scatter(x=sim.index, y=sim['BuyHold_Equity'], mode='lines', name='Buy & Hold'))
    buys = sim[sim['Action'] == 'BUY']
    sells = sim[sim['Action'] == 'SELL']
    fig.add_trace(go.Scatter(x=buys.index, y=buys['Equity'], mode='markers', name='BUY', marker_symbol='triangle-up', marker_size=10))
    fig.add_trace(go.Scatter(x=sells.index, y=sells['Equity'], mode='markers', name='SELL', marker_symbol='triangle-down', marker_size=10))
    fig.update_layout(title='Equity Curve', xaxis_title='Date', yaxis_title='Portfolio Value', height=500)
    return fig


def plot_price_signals(sim: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sim.index, y=sim['Close'], mode='lines', name='Close'))
    buys = sim[sim['Action'] == 'BUY']
    sells = sim[sim['Action'] == 'SELL']
    fig.add_trace(go.Scatter(x=buys.index, y=buys['Close'], mode='markers', name='BUY', marker_symbol='triangle-up', marker_size=11))
    fig.add_trace(go.Scatter(x=sells.index, y=sells['Close'], mode='markers', name='SELL', marker_symbol='triangle-down', marker_size=11))
    fig.update_layout(title='Price With Trading Signals', xaxis_title='Date', yaxis_title='Price', height=500)
    return fig


st.title('Dual-Model Trading Simulator')
st.caption('Trades using a classification model, a regression model, and optimized signal thresholds inspired by your uploaded algorithmic trading materials.')

with st.sidebar:
    st.header('Settings')
    ticker = st.text_input('Ticker', value='AAPL')
    start = st.date_input('Start date', value=pd.Timestamp('2020-01-01'))
    end = st.date_input('End date', value=pd.Timestamp.today())
    interval = st.selectbox('Interval', ['1d', '1wk', '1mo'], index=0)
    train_ratio = st.slider('Train split', min_value=0.50, max_value=0.90, value=0.75, step=0.05)
    initial_cash = st.number_input('Initial cash', min_value=1000.0, value=10000.0, step=1000.0)
    fee = st.number_input('Transaction fee (decimal)', min_value=0.0, value=0.002, step=0.001, format='%.4f')
    strategy_mode = st.selectbox('Trading mode', ['Combined (both models)', 'Classification only', 'Regression only'])
    optimize_models = st.checkbox('Optimize model hyperparameters', value=True)
    optimize_signals = st.checkbox('Optimize buy/sell thresholds', value=True)
    manual_buy = st.number_input('Manual buy threshold (%)', value=0.50, step=0.25)
    manual_sell = st.number_input('Manual sell threshold (%)', value=-0.50, step=0.25)
    run = st.button('Run simulator', type='primary')

if run:
    df_raw = load_data(ticker, str(start), str(end), interval)
    if df_raw.empty or len(df_raw) < 120:
        st.error('Not enough price data. Try a longer date range or a different ticker.')
        st.stop()

    df = engineer_features(df_raw)
    if len(df) < 100:
        st.error('Not enough engineered rows after feature creation. Try a longer history.')
        st.stop()

    train_df, test_df = split_data(df, train_ratio)
    cls_model, reg_model, cls_params, reg_params, features = train_models(train_df, optimize_models)

    train_pred = add_predictions(train_df, cls_model, reg_model, features)
    test_pred = add_predictions(test_df, cls_model, reg_model, features)

    if optimize_signals:
        buy_threshold, sell_threshold, opt_meta = optimize_signal_thresholds(
            train_pred, strategy_mode, initial_cash, fee
        )
    else:
        buy_threshold, sell_threshold = manual_buy, manual_sell
        opt_meta = {'best_score': None}

    sim_test, stats = simulate_strategy(test_pred, buy_threshold or manual_buy, sell_threshold or manual_sell,
                                        strategy_mode, initial_cash, fee)

    cls_acc = accuracy_score(test_pred['Target_Cls'], test_pred['Pred_Cls'])
    reg_mae = mean_absolute_error(test_pred['Target_Reg'], test_pred['Pred_Reg'])
    reg_rmse = np.sqrt(mean_squared_error(test_pred['Target_Reg'], test_pred['Pred_Reg']))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Classifier Accuracy', f'{cls_acc:.2%}')
    c2.metric('Regressor MAE', f'{reg_mae:.3f}%')
    c3.metric('Regressor RMSE', f'{reg_rmse:.3f}%')
    c4.metric('Final Equity', f"${stats['Final Equity']:,.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric('Strategy Return', f"{stats['Strategy Return [%]']:.2f}%")
    c6.metric('Buy & Hold', f"{stats['Buy & Hold Return [%]']:.2f}%")
    c7.metric('Sharpe', f"{stats['Sharpe']:.2f}")
    c8.metric('Max Drawdown', f"{stats['Max Drawdown [%]']:.2f}%")

    st.subheader('Chosen Model Settings')
    p1, p2, p3 = st.columns(3)
    p1.write({'classification_model': cls_params})
    p2.write({'regression_model': reg_params})
    p3.write({
        'buy_threshold_%': buy_threshold if buy_threshold is not None else 'not used',
        'sell_threshold_%': sell_threshold if sell_threshold is not None else 'not used',
        'signal_optimizer_score': opt_meta.get('best_score')
    })

    st.plotly_chart(plot_price_signals(sim_test), use_container_width=True)
    st.plotly_chart(plot_equity(sim_test), use_container_width=True)

    st.subheader('Latest predictions and trades')
    show_cols = ['Close', 'Target_Cls', 'Pred_Cls', 'Target_Reg', 'Pred_Reg', 'Action', 'Position', 'Equity']
    st.dataframe(sim_test[show_cols].tail(30), use_container_width=True)

    csv = sim_test.reset_index().to_csv(index=False).encode('utf-8')
    st.download_button('Download simulation results CSV', csv, file_name=f'{ticker}_simulation_results.csv', mime='text/csv')
else:
    st.info('Set the ticker and settings in the sidebar, then click **Run simulator**.')
