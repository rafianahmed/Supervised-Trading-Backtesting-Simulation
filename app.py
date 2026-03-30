import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
import yfinance as yf
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title='Softmax Dual-Model Trading Simulator', layout='wide')

SEED = 42
LABEL_MAP = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}


@st.cache_data(show_spinner=False, ttl=3600)
def load_yahoo_data(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    last_err = None
    for attempt in range(4):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [str(c).title() for c in df.columns]
            keep = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
            df = df[keep].copy().dropna()
            if not df.empty:
                return df
        except Exception as e:
            last_err = e
        time.sleep(2 * (attempt + 1))

    raise RuntimeError(
        f"Failed to download data for {ticker}. Yahoo Finance may be rate-limiting requests. "
        f"Try again later or switch to CSV upload. Last error: {last_err}"
    )


@st.cache_data(show_spinner=False)
def load_csv_data(file_bytes: bytes, filename: str) -> pd.DataFrame:
    from io import BytesIO
    df = pd.read_csv(BytesIO(file_bytes))
    df.columns = [str(c).strip() for c in df.columns]
    rename_map = {}
    for col in df.columns:
        lc = col.lower()
        if lc == 'date':
            rename_map[col] = 'Date'
        elif lc == 'open':
            rename_map[col] = 'Open'
        elif lc == 'high':
            rename_map[col] = 'High'
        elif lc == 'low':
            rename_map[col] = 'Low'
        elif lc == 'close':
            rename_map[col] = 'Close'
        elif lc == 'volume':
            rename_map[col] = 'Volume'
    df = df.rename(columns=rename_map)

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. CSV must include Date, Open, High, Low, Close, Volume.")

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date']).sort_values('Date').set_index('Date')
    else:
        df.index = pd.RangeIndex(start=0, stop=len(df), step=1)

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df[required_cols].dropna().copy()


def get_market_data(data_mode: str, ticker: str, start: str, end: str, interval: str, uploaded_file):
    if 'market_data_cache' not in st.session_state:
        st.session_state['market_data_cache'] = {}

    if data_mode == 'Yahoo Finance':
        cache_key = f"yf::{ticker}::{start}::{end}::{interval}"
        if cache_key not in st.session_state['market_data_cache']:
            st.session_state['market_data_cache'][cache_key] = load_yahoo_data(ticker, start, end, interval)
        return st.session_state['market_data_cache'][cache_key]

    if uploaded_file is None:
        raise ValueError('Please upload a CSV file with Date, Open, High, Low, Close, Volume columns.')

    file_bytes = uploaded_file.getvalue()
    cache_key = f"csv::{uploaded_file.name}::{len(file_bytes)}"
    if cache_key not in st.session_state['market_data_cache']:
        st.session_state['market_data_cache'][cache_key] = load_csv_data(file_bytes, uploaded_file.name)
    return st.session_state['market_data_cache'][cache_key]


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(close: pd.Series):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def engineer_features(df: pd.DataFrame, move_threshold_pct: float, horizon: int) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    close = out['Close']
    open_ = out['Open']
    high = out['High']
    low = out['Low']
    volume = out['Volume'].replace(0, np.nan)

    out['ret_1'] = close.pct_change() * 100
    for lag in [2, 3, 5, 10]:
        out[f'ret_{lag}'] = close.pct_change(lag) * 100

    for w in [5, 10, 20]:
        sma = close.rolling(w).mean()
        out[f'price_sma_{w}'] = (close / sma - 1) * 100
        out[f'sma_slope_{w}'] = sma.pct_change(3) * 100
        out[f'vol_{w}'] = out['ret_1'].rolling(w).std()
        out[f'breakout_up_{w}'] = (close / high.rolling(w).max() - 1) * 100
        out[f'breakout_dn_{w}'] = (close / low.rolling(w).min() - 1) * 100
        out[f'mom_{w}'] = close.pct_change(w) * 100

    out['range_pct'] = ((high - low) / close.replace(0, np.nan)) * 100
    out['body_pct'] = ((close - open_) / open_.replace(0, np.nan)) * 100
    out['gap_pct'] = ((open_ - close.shift(1)) / close.shift(1).replace(0, np.nan)) * 100
    out['hl_spread'] = ((high - low) / open_.replace(0, np.nan)) * 100
    out['vol_chg_1'] = volume.pct_change() * 100
    out['vol_chg_5'] = volume.pct_change(5) * 100
    out['vol_ratio_5'] = volume / volume.rolling(5).mean()
    out['vol_ratio_20'] = volume / volume.rolling(20).mean()

    out['rsi_14'] = compute_rsi(close, 14)
    macd, macd_signal, macd_hist = compute_macd(close)
    out['macd'] = macd
    out['macd_signal'] = macd_signal
    out['macd_hist'] = macd_hist

    future_return = (close.shift(-horizon) / close - 1) * 100
    out['target_reg'] = future_return
    threshold = float(move_threshold_pct)
    out['target_cls'] = np.select(
        [future_return > threshold, future_return < -threshold],
        [1, -1],
        default=0,
    )

    feature_cols = [
        c for c in out.columns
        if c not in ['target_reg', 'target_cls'] and pd.api.types.is_numeric_dtype(out[c])
    ]
    out = out.replace([np.inf, -np.inf], np.nan).dropna().copy()
    return out, feature_cols


def split_data(df: pd.DataFrame, train_ratio: float, valid_ratio: float):
    n = len(df)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))
    train_df = df.iloc[:train_end].copy()
    valid_df = df.iloc[train_end:valid_end].copy()
    test_df = df.iloc[valid_end:].copy()
    return train_df, valid_df, test_df


def select_significant_features(train_df: pd.DataFrame, feature_cols: list[str], pval_cutoff: float, max_features: int):
    X = train_df[feature_cols].copy().replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    y = train_df.loc[X.index, 'target_reg']
    usable = list(X.columns)
    if len(usable) == 0:
        return feature_cols[: min(len(feature_cols), max_features)], pd.DataFrame()

    try:
        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()
        pvals = model.pvalues.drop('const', errors='ignore').sort_values()
        summary = pd.DataFrame({'feature': pvals.index, 'p_value': pvals.values})
        selected = summary.loc[summary['p_value'] <= pval_cutoff, 'feature'].tolist()
        if len(selected) == 0:
            selected = summary.head(min(max_features, len(summary)))['feature'].tolist()
        selected = selected[:max_features]
        return selected, summary
    except Exception:
        fallback = usable[: min(len(usable), max_features)]
        summary = pd.DataFrame({'feature': fallback, 'p_value': np.nan})
        return fallback, summary


def fit_models(train_df: pd.DataFrame, selected_features: list[str], alpha_reg: float, clf_c: float):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[selected_features])
    y_reg = train_df['target_reg'].values
    y_cls = train_df['target_cls'].values

    clf = LogisticRegression(
        solver='lbfgs',
        C=clf_c,
        class_weight='balanced',
        max_iter=2500,
        random_state=SEED,
    )
    reg = SGDRegressor(
        loss='squared_error',
        penalty='l2',
        alpha=alpha_reg,
        max_iter=2500,
        tol=1e-4,
        random_state=SEED,
    )

    clf.fit(X_train, y_cls)
    reg.fit(X_train, y_reg)
    return scaler, clf, reg


def add_predictions(df: pd.DataFrame, scaler: StandardScaler, clf, reg, selected_features: list[str]) -> pd.DataFrame:
    out = df.copy()
    X_scaled = scaler.transform(out[selected_features])
    proba = clf.predict_proba(X_scaled)
    classes = clf.classes_
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    out['prob_sell'] = proba[:, class_to_idx.get(-1, 0)] if -1 in class_to_idx else 0.0
    out['prob_hold'] = proba[:, class_to_idx.get(0, 0)] if 0 in class_to_idx else 0.0
    out['prob_buy'] = proba[:, class_to_idx.get(1, 0)] if 1 in class_to_idx else 0.0
    out['pred_cls_num'] = clf.predict(X_scaled)
    out['pred_cls'] = out['pred_cls_num'].map(LABEL_MAP)
    out['pred_reg'] = reg.predict(X_scaled)
    return out


def simulate_strategy(
    df: pd.DataFrame,
    trading_mode: str,
    buy_prob_th: float,
    sell_prob_th: float,
    buy_ret_th: float,
    sell_ret_th: float,
    initial_cash: float,
    fee: float,
    cooldown_bars: int,
    min_holding_bars: int,
):
    sim = df.copy()
    cash = float(initial_cash)
    shares = 0.0
    position = 0
    bars_in_position = 0
    cooldown_left = 0

    equity_curve, actions, positions = [], [], []

    for _, row in sim.iterrows():
        price = float(row['Close'])
        buy_prob = float(row['prob_buy'])
        sell_prob = float(row['prob_sell'])
        pred_ret = float(row['pred_reg'])

        if position == 1:
            bars_in_position += 1
        if cooldown_left > 0:
            cooldown_left -= 1

        buy_signal = False
        sell_signal = False

        if trading_mode == 'Softmax only':
            buy_signal = (buy_prob >= buy_prob_th) and position == 0 and cooldown_left == 0
            sell_signal = (sell_prob >= sell_prob_th) and position == 1 and bars_in_position >= min_holding_bars
        elif trading_mode == 'Regression only':
            buy_signal = (pred_ret >= buy_ret_th) and position == 0 and cooldown_left == 0
            sell_signal = (pred_ret <= sell_ret_th) and position == 1 and bars_in_position >= min_holding_bars
        else:
            buy_signal = (
                buy_prob >= buy_prob_th and pred_ret >= buy_ret_th and position == 0 and cooldown_left == 0
            )
            sell_signal = (
                (sell_prob >= sell_prob_th or pred_ret <= sell_ret_th)
                and position == 1
                and bars_in_position >= min_holding_bars
            )

        action = 'HOLD'
        if buy_signal and price > 0:
            shares = (cash * (1 - fee)) / price
            cash = 0.0
            position = 1
            bars_in_position = 0
            action = 'BUY'
        elif sell_signal and price > 0:
            cash = shares * price * (1 - fee)
            shares = 0.0
            position = 0
            cooldown_left = cooldown_bars
            bars_in_position = 0
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
    sharpe = (sim['Strategy_Return'].mean() / vol * np.sqrt(252)) if pd.notna(vol) and vol > 0 else 0.0
    max_dd = ((sim['Equity'] / sim['Equity'].cummax()) - 1).min() * 100
    trades = int(sim['Action'].isin(['BUY', 'SELL']).sum())

    stats = {
        'Final Equity': float(sim['Equity'].iloc[-1]),
        'Strategy Return [%]': float(total_return),
        'Buy & Hold Return [%]': float(bh_return),
        'Sharpe': float(sharpe),
        'Max Drawdown [%]': float(max_dd),
        'Trades': trades,
    }
    return sim, stats


def optimize_thresholds(valid_pred: pd.DataFrame, trading_mode: str, initial_cash: float, fee: float, cooldown_bars: int, min_holding_bars: int):
    best = {
        'buy_prob_th': 0.55,
        'sell_prob_th': 0.55,
        'buy_ret_th': 0.25,
        'sell_ret_th': -0.25,
        'score': -np.inf,
    }

    prob_grid = [0.45, 0.50, 0.55, 0.60, 0.65]
    buy_grid = [0.10, 0.20, 0.30, 0.40, 0.50, 0.75]
    sell_grid = [-0.10, -0.20, -0.30, -0.40, -0.50, -0.75]

    for buy_prob_th in prob_grid:
        for sell_prob_th in prob_grid:
            for buy_ret_th in buy_grid:
                for sell_ret_th in sell_grid:
                    _, stats = simulate_strategy(
                        valid_pred,
                        trading_mode,
                        buy_prob_th,
                        sell_prob_th,
                        buy_ret_th,
                        sell_ret_th,
                        initial_cash,
                        fee,
                        cooldown_bars,
                        min_holding_bars,
                    )
                    score = stats['Sharpe'] - 0.05 * max(0, -stats['Max Drawdown [%]']) / 10 - 0.001 * stats['Trades']
                    if np.isfinite(score) and score > best['score']:
                        best = {
                            'buy_prob_th': buy_prob_th,
                            'sell_prob_th': sell_prob_th,
                            'buy_ret_th': buy_ret_th,
                            'sell_ret_th': sell_ret_th,
                            'score': float(score),
                        }
    return best


def plot_equity(sim: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sim.index, y=sim['Equity'], mode='lines', name='Strategy Equity'))
    fig.add_trace(go.Scatter(x=sim.index, y=sim['BuyHold_Equity'], mode='lines', name='Buy & Hold'))
    buys = sim[sim['Action'] == 'BUY']
    sells = sim[sim['Action'] == 'SELL']
    fig.add_trace(go.Scatter(x=buys.index, y=buys['Equity'], mode='markers', name='BUY', marker_symbol='triangle-up', marker_size=10))
    fig.add_trace(go.Scatter(x=sells.index, y=sells['Equity'], mode='markers', name='SELL', marker_symbol='triangle-down', marker_size=10))
    fig.update_layout(title='Equity Curve', xaxis_title='Date', yaxis_title='Portfolio Value', height=480)
    return fig


def plot_price_signals(sim: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sim.index, y=sim['Close'], mode='lines', name='Close'))
    buys = sim[sim['Action'] == 'BUY']
    sells = sim[sim['Action'] == 'SELL']
    fig.add_trace(go.Scatter(x=buys.index, y=buys['Close'], mode='markers', name='BUY', marker_symbol='triangle-up', marker_size=11))
    fig.add_trace(go.Scatter(x=sells.index, y=sells['Close'], mode='markers', name='SELL', marker_symbol='triangle-down', marker_size=11))
    fig.update_layout(title='Price With Trading Signals', xaxis_title='Date', yaxis_title='Price', height=480)
    return fig


def pretty_confusion_matrix(y_true: pd.Series, y_pred: pd.Series):
    labels = [-1, 0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=['True SELL', 'True HOLD', 'True BUY'], columns=['Pred SELL', 'Pred HOLD', 'Pred BUY'])
    return cm_df


st.title('Softmax Dual-Model Trading Simulator')
st.caption('Uses scaled features, OLS significance filtering, softmax probabilities, a gradient-descent regressor, and Yahoo/CSV data input.')

with st.sidebar:
    st.header('Market Data')
    data_mode = st.radio('Choose data source', ['Yahoo Finance', 'Upload CSV'], index=0)
    ticker = st.text_input('Ticker', value='AAPL')
    start = st.date_input('Start date', value=pd.Timestamp('2018-01-01'))
    end = st.date_input('End date', value=pd.Timestamp.today())
    interval = st.selectbox('Interval', ['1d', '1wk', '1mo'], index=0)
    uploaded_file = None
    if data_mode == 'Upload CSV':
        uploaded_file = st.file_uploader('Upload OHLCV CSV', type=['csv'])
        st.caption('Required columns: Date, Open, High, Low, Close, Volume')

    st.header('Target Design')
    horizon = st.slider('Prediction horizon (bars ahead)', min_value=1, max_value=10, value=1)
    move_threshold_pct = st.slider('Buy/Sell label threshold (%)', min_value=0.10, max_value=2.00, value=0.40, step=0.05)

    st.header('Data Split')
    train_ratio = st.slider('Train ratio', min_value=0.50, max_value=0.75, value=0.60, step=0.05)
    valid_ratio = st.slider('Validation ratio', min_value=0.10, max_value=0.25, value=0.20, step=0.05)

    st.header('Feature Selection')
    pval_cutoff = st.select_slider('OLS p-value cutoff', options=[0.01, 0.03, 0.05, 0.10, 0.15], value=0.10)
    max_features = st.slider('Max selected features', min_value=5, max_value=20, value=12)

    st.header('Models')
    clf_c = st.select_slider('Softmax inverse regularization C', options=[0.05, 0.10, 0.25, 0.50, 1.0, 2.0], value=0.50)
    alpha_reg = st.select_slider('Regressor L2 alpha', options=[0.0001, 0.001, 0.005, 0.01, 0.05], value=0.001)

    st.header('Trading Rules')
    trading_mode = st.selectbox('Trading mode', ['Combined', 'Softmax only', 'Regression only'])
    optimize_thresholds_flag = st.checkbox('Optimize thresholds on validation set', value=True)
    buy_prob_th_manual = st.slider('Manual buy probability threshold', min_value=0.33, max_value=0.90, value=0.60, step=0.01)
    sell_prob_th_manual = st.slider('Manual sell probability threshold', min_value=0.33, max_value=0.90, value=0.60, step=0.01)
    buy_ret_th_manual = st.slider('Manual buy return threshold (%)', min_value=0.05, max_value=1.50, value=0.30, step=0.05)
    sell_ret_th_manual = st.slider('Manual sell return threshold (%)', min_value=-1.50, max_value=-0.05, value=-0.30, step=0.05)
    cooldown_bars = st.slider('Cooldown bars after sell', min_value=0, max_value=10, value=1)
    min_holding_bars = st.slider('Minimum holding bars', min_value=0, max_value=10, value=1)
    initial_cash = st.number_input('Initial cash', min_value=1000.0, value=10000.0, step=1000.0)
    fee = st.number_input('Transaction fee (decimal)', min_value=0.0, value=0.002, step=0.001, format='%.4f')

    run = st.button('Run simulator', type='primary')


if run:
    if train_ratio + valid_ratio >= 0.95:
        st.error('Train ratio + validation ratio must leave enough room for the test set.')
        st.stop()

    try:
        raw = get_market_data(data_mode, ticker, str(start), str(end), interval, uploaded_file)
    except Exception as e:
        st.error(str(e))
        if data_mode == 'Yahoo Finance':
            st.info('Yahoo Finance is temporarily rate-limiting requests. Switch to Upload CSV in the sidebar for a stable run.')
        st.stop()

    if raw is None or raw.empty:
        st.error('No market data available. Try another ticker/date range or use CSV upload.')
        st.stop()

    if len(raw) < 250:
        st.error('Not enough price data. Use a longer date range or another ticker.')
        st.stop()

    df, feature_cols = engineer_features(raw, move_threshold_pct, horizon)
    if len(df) < 180:
        st.error('Not enough rows after feature engineering. Try more history or a lower horizon.')
        st.stop()

    train_df, valid_df, test_df = split_data(df, train_ratio, valid_ratio)
    if min(len(train_df), len(valid_df), len(test_df)) < 40:
        st.error('One split is too small. Adjust the ratios or use more data.')
        st.stop()

    selected_features, pval_summary = select_significant_features(train_df, feature_cols, pval_cutoff, max_features)
    scaler, clf, reg = fit_models(train_df, selected_features, alpha_reg, clf_c)

    valid_pred = add_predictions(valid_df, scaler, clf, reg, selected_features)
    test_pred = add_predictions(test_df, scaler, clf, reg, selected_features)

    if optimize_thresholds_flag:
        best = optimize_thresholds(valid_pred, trading_mode, initial_cash, fee, cooldown_bars, min_holding_bars)
    else:
        best = {
            'buy_prob_th': buy_prob_th_manual,
            'sell_prob_th': sell_prob_th_manual,
            'buy_ret_th': buy_ret_th_manual,
            'sell_ret_th': sell_ret_th_manual,
            'score': np.nan,
        }

    sim_test, stats = simulate_strategy(
        test_pred,
        trading_mode,
        best['buy_prob_th'],
        best['sell_prob_th'],
        best['buy_ret_th'],
        best['sell_ret_th'],
        initial_cash,
        fee,
        cooldown_bars,
        min_holding_bars,
    )

    cls_acc = accuracy_score(test_pred['target_cls'], test_pred['pred_cls_num'])
    reg_mae = mean_absolute_error(test_pred['target_reg'], test_pred['pred_reg'])
    reg_rmse = np.sqrt(mean_squared_error(test_pred['target_reg'], test_pred['pred_reg']))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Softmax Accuracy', f'{cls_acc:.2%}')
    c2.metric('Regressor MAE', f'{reg_mae:.3f}%')
    c3.metric('Regressor RMSE', f'{reg_rmse:.3f}%')
    c4.metric('Final Equity', f"${stats['Final Equity']:,.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric('Strategy Return', f"{stats['Strategy Return [%]']:.2f}%")
    c6.metric('Buy & Hold', f"{stats['Buy & Hold Return [%]']:.2f}%")
    c7.metric('Sharpe', f"{stats['Sharpe']:.2f}")
    c8.metric('Max Drawdown', f"{stats['Max Drawdown [%]']:.2f}%")

    st.subheader('Selected Features and Model Setup')
    m1, m2, m3 = st.columns(3)
    m1.write({'selected_features': selected_features})
    m2.write({
        'softmax_model': {
            'type': 'Logistic Regression with Softmax probabilities',
            'solver': 'lbfgs',
            'C': clf_c,
            'class_weight': 'balanced',
        },
        'regressor_model': {
            'type': 'SGDRegressor',
            'loss': 'squared_error',
            'penalty': 'l2',
            'alpha': alpha_reg,
        },
        'data_source': data_mode,
    })
    m3.write({
        'buy_prob_threshold': best['buy_prob_th'],
        'sell_prob_threshold': best['sell_prob_th'],
        'buy_return_threshold_%': best['buy_ret_th'],
        'sell_return_threshold_%': best['sell_ret_th'],
        'threshold_score': best['score'],
    })

    st.plotly_chart(plot_price_signals(sim_test), use_container_width=True)
    st.plotly_chart(plot_equity(sim_test), use_container_width=True)

    st.subheader('Softmax Confusion Matrix')
    st.dataframe(pretty_confusion_matrix(test_pred['target_cls'], test_pred['pred_cls_num']), use_container_width=True)

    st.subheader('OLS Feature Significance Summary')
    st.dataframe(pval_summary.head(20), use_container_width=True)

    st.subheader('Latest predictions and trades')
    view_cols = [
        'Close', 'target_reg', 'pred_reg', 'target_cls', 'pred_cls',
        'prob_sell', 'prob_hold', 'prob_buy', 'Action', 'Position', 'Equity'
    ]
    st.dataframe(sim_test[view_cols].tail(40), use_container_width=True)

    csv = sim_test.reset_index().to_csv(index=False).encode('utf-8')
    base_name = ticker if data_mode == 'Yahoo Finance' else 'uploaded_data'
    st.download_button('Download simulation results CSV', csv, file_name=f'{base_name}_softmax_simulation_results.csv', mime='text/csv')
else:
    st.info('Choose the settings in the sidebar and click Run simulator.')
