import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from trendline_automation import fit_trendlines_single, fit_upper_trendline


def trendline_breakout_dataset(
        ohlcv: pd.DataFrame, lookback: int,
        hold_period: int = 12, tp_mult: float = 3.0, sl_mult: float = 3.0,
        atr_lookback: int = 168
):
    assert(atr_lookback >= lookback)

    close = np.log(ohlcv['close'].to_numpy())

    # ── Indicateurs de base ──────────────────────────────────────────────────
    atr = ta.atr(np.log(ohlcv['high']), np.log(ohlcv['low']),
                 np.log(ohlcv['close']), atr_lookback)
    atr_arr = atr.to_numpy()

    # ATR long terme (2x atr_lookback) pour mesurer le régime de volatilité
    atr_slow = ta.atr(np.log(ohlcv['high']), np.log(ohlcv['low']),
                      np.log(ohlcv['close']), atr_lookback * 2)
    atr_slow_arr = atr_slow.to_numpy()

    # Volume normalisé
    vol_arr = (ohlcv['volume'] / ohlcv['volume'].rolling(atr_lookback).median()).to_numpy()

    # ADX
    adx = ta.adx(ohlcv['high'], ohlcv['low'], ohlcv['close'], lookback)
    adx_arr = adx['ADX_' + str(lookback)].to_numpy()

    # ── NOUVELLES features de contexte ──────────────────────────────────────

    # Tendance court terme : prix vs MA50
    ma_fast = ohlcv['close'].rolling(50).mean().to_numpy()
    # Tendance long terme : prix vs MA200
    ma_slow = ohlcv['close'].rolling(200).mean().to_numpy()
    close_raw = ohlcv['close'].to_numpy()

    # Heure et jour (index datetime)
    hour_arr        = np.array(ohlcv.index.hour)
    day_arr         = np.array(ohlcv.index.dayofweek)  # 0=lundi, 6=dimanche

    # ── Boucle principale ────────────────────────────────────────────────────
    trades   = pd.DataFrame()
    trade_i  = 0
    in_trade = False
    tp_price = sl_price = hp_i = None

    for i in range(atr_lookback, len(ohlcv)):
        window = close[i - lookback: i]

        s_coefs, r_coefs = fit_trendlines_single(window)
        r_val = r_coefs[1] + lookback * r_coefs[0]

        # ── Entrée ───────────────────────────────────────────────────────────
        if not in_trade and close[i] > r_val:

            tp_price = close[i] + atr_arr[i] * tp_mult
            sl_price = close[i] - atr_arr[i] * sl_mult
            hp_i     = i + hold_period
            in_trade = True

            trades.loc[trade_i, 'entry_i'] = i
            trades.loc[trade_i, 'entry_p'] = close[i]
            trades.loc[trade_i, 'atr']     = atr_arr[i]
            trades.loc[trade_i, 'sl']      = sl_price
            trades.loc[trade_i, 'tp']      = tp_price
            trades.loc[trade_i, 'hp_i']    = i + hold_period
            trades.loc[trade_i, 'slope']   = r_coefs[0]
            trades.loc[trade_i, 'intercept'] = r_coefs[1]

            # ── Features existantes ──────────────────────────────────────────
            line_vals = r_coefs[1] + np.arange(lookback) * r_coefs[0]
            diff      = line_vals - window

            trades.loc[trade_i, 'resist_s'] = r_coefs[0] / atr_arr[i]
            trades.loc[trade_i, 'tl_err']   = (diff.sum() / lookback) / atr_arr[i]
            trades.loc[trade_i, 'max_dist'] = diff.max() / atr_arr[i]
            trades.loc[trade_i, 'vol']      = vol_arr[i]
            trades.loc[trade_i, 'adx']      = adx_arr[i]

            # ── NOUVELLES features ───────────────────────────────────────────

            # 1. Heure de la bougie (0-23)
            #    Les cassures en session asiatique (0-8h) sont souvent fausses
            trades.loc[trade_i, 'hour'] = hour_arr[i]

            # 2. Jour de la semaine (0=lundi … 6=dimanche)
            #    Le weekend BTC est souvent moins fiable
            trades.loc[trade_i, 'day_of_week'] = day_arr[i]

            # 3. Tendance court terme : prix au-dessus ou en-dessous de MA50
            #    > 0 = tendance haussière → cassure plus fiable
            if ma_fast[i] > 0:
                trades.loc[trade_i, 'trend_fast'] = (close_raw[i] - ma_fast[i]) / ma_fast[i]
            else:
                trades.loc[trade_i, 'trend_fast'] = 0.0

            # 4. Tendance long terme : prix vs MA200
            #    Cassure dans le sens de la tendance long terme = plus fiable
            if ma_slow[i] > 0:
                trades.loc[trade_i, 'trend_slow'] = (close_raw[i] - ma_slow[i]) / ma_slow[i]
            else:
                trades.loc[trade_i, 'trend_slow'] = 0.0

            # 5. Régime de volatilité : ATR court / ATR long
            #    > 1 = marché agité (breakouts moins fiables)
            #    < 1 = marché calme (breakouts plus propres)
            if atr_slow_arr[i] > 0:
                trades.loc[trade_i, 'vol_regime'] = atr_arr[i] / atr_slow_arr[i]
            else:
                trades.loc[trade_i, 'vol_regime'] = 1.0

            # 6. Nombre de touches de la résistance avant la cassure
            #    Plus la ligne a été touchée, plus la cassure est significative
            tolerance   = atr_arr[i] * 0.5
            r_line_vals = r_coefs[1] + np.arange(lookback) * r_coefs[0]
            n_touches   = np.sum(np.abs(window - r_line_vals) < tolerance)
            trades.loc[trade_i, 'n_touches'] = n_touches

        # ── Sortie ───────────────────────────────────────────────────────────
        if in_trade:
            if close[i] >= tp_price or close[i] <= sl_price or i >= hp_i:
                trades.loc[trade_i, 'exit_i'] = i
                trades.loc[trade_i, 'exit_p'] = close[i]
                in_trade = False
                trade_i += 1

    trades['return'] = trades['exit_p'] - trades['entry_p']

    # Features complètes : 5 originales + 6 nouvelles = 11 features
    feature_cols = [
        'resist_s', 'tl_err', 'max_dist', 'vol', 'adx',  # originales
        'hour', 'day_of_week', 'trend_fast', 'trend_slow',  # contexte temps
        'vol_regime', 'n_touches'                            # contexte marché
    ]
    data_x = trades[feature_cols]
    data_y = pd.Series(0, index=trades.index)
    data_y.loc[trades['return'] > 0] = 1

    return trades, data_x, data_y


# ════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    data = pd.read_csv('BTCUSDT3600.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')
    data = data.dropna()

    trades, data_x, data_y = trendline_breakout_dataset(data, 72)
    trades = trades.dropna()

    print(f"Total trades    : {len(trades)}")
    print(f"Features        : {list(data_x.columns)}")
    print(f"Win rate base   : {round(len(trades[trades['return'] > 0]) / len(trades), 4)}")
    print(f"\nAperçu features :")
    print(data_x.describe().round(4))