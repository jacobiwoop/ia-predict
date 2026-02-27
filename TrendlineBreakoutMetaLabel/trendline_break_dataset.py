import numpy as np
import pandas as pd
import pandas_ta as ta
from trendline_automation import fit_trendlines_single


def trendline_breakout_dataset(
        ohlcv: pd.DataFrame, lookback: int,
        hold_period: int = 12, tp_mult: float = 3.0, sl_mult: float = 3.0,
        atr_lookback: int = 168
):
    assert atr_lookback >= lookback

    close = np.log(ohlcv['close'].to_numpy())

    # ── Indicateurs de base ─────────────────────────────────────────────────
    atr = ta.atr(np.log(ohlcv['high']), np.log(ohlcv['low']),
                 np.log(ohlcv['close']), atr_lookback)
    atr_arr = atr.to_numpy()

    vol_arr = (ohlcv['volume'] / ohlcv['volume'].rolling(atr_lookback).median()).to_numpy()

    adx     = ta.adx(ohlcv['high'], ohlcv['low'], ohlcv['close'], lookback)
    adx_arr = adx['ADX_' + str(lookback)].to_numpy()

    close_raw = ohlcv['close'].to_numpy()

    # ── Features contextuelles corrigées ───────────────────────────────────

    # 1. HEURE → encodée en sinus/cosinus (relation circulaire)
    #    sin(0h) ≈ sin(24h), les heures proches restent proches
    hour_arr     = np.array(ohlcv.index.hour, dtype=float)
    hour_sin     = np.sin(2 * np.pi * hour_arr / 24)
    hour_cos     = np.cos(2 * np.pi * hour_arr / 24)

    # 2. JOUR → encodé en sinus/cosinus (relation circulaire)
    dow_arr      = np.array(ohlcv.index.dayofweek, dtype=float)
    dow_sin      = np.sin(2 * np.pi * dow_arr / 7)
    dow_cos      = np.cos(2 * np.pi * dow_arr / 7)

    # 3. TENDANCE : retour sur X bougies normalisé par ATR
    #    Mesure si on est dans un trend fort ou faible, pas juste la direction
    ret_24  = pd.Series(close_raw).pct_change(24).to_numpy()   # 24h
    ret_168 = pd.Series(close_raw).pct_change(168).to_numpy()  # 1 semaine

    # 4. REGIME DE VOLATILITE : ATR / ATR_long
    atr_slow     = ta.atr(np.log(ohlcv['high']), np.log(ohlcv['low']),
                          np.log(ohlcv['close']), atr_lookback * 2)
    atr_slow_arr = atr_slow.to_numpy()
    vol_regime   = np.where(atr_slow_arr > 0, atr_arr / atr_slow_arr, 1.0)

    # 5. POSITION RELATIVE dans le range récent (0=bas, 1=haut)
    #    Si le prix est déjà au sommet du range → cassure moins fiable
    high_168 = pd.Series(close_raw).rolling(168).max().to_numpy()
    low_168  = pd.Series(close_raw).rolling(168).min().to_numpy()
    range_168 = high_168 - low_168
    price_position = np.where(range_168 > 0,
                               (close_raw - low_168) / range_168,
                               0.5)

    # ── Boucle principale ───────────────────────────────────────────────────
    trades   = pd.DataFrame()
    trade_i  = 0
    in_trade = False
    tp_price = sl_price = hp_i = None

    for i in range(atr_lookback, len(ohlcv)):
        window   = close[i - lookback: i]
        s_coefs, r_coefs = fit_trendlines_single(window)
        r_val    = r_coefs[1] + lookback * r_coefs[0]

        # ── Entrée ──────────────────────────────────────────────────────────
        if not in_trade and close[i] > r_val:

            tp_price = close[i] + atr_arr[i] * tp_mult
            sl_price = close[i] - atr_arr[i] * sl_mult
            hp_i     = i + hold_period
            in_trade = True

            trades.loc[trade_i, 'entry_i']   = i
            trades.loc[trade_i, 'entry_p']   = close[i]
            trades.loc[trade_i, 'atr']       = atr_arr[i]
            trades.loc[trade_i, 'sl']        = sl_price
            trades.loc[trade_i, 'tp']        = tp_price
            trades.loc[trade_i, 'hp_i']      = i + hold_period
            trades.loc[trade_i, 'slope']     = r_coefs[0]
            trades.loc[trade_i, 'intercept'] = r_coefs[1]

            # ── Features trendline (originales) ─────────────────────────────
            line_vals = r_coefs[1] + np.arange(lookback) * r_coefs[0]
            diff      = line_vals - window

            trades.loc[trade_i, 'resist_s']     = r_coefs[0] / atr_arr[i]
            trades.loc[trade_i, 'tl_err']       = (diff.sum() / lookback) / atr_arr[i]
            trades.loc[trade_i, 'max_dist']     = diff.max() / atr_arr[i]
            trades.loc[trade_i, 'vol']          = vol_arr[i]
            trades.loc[trade_i, 'adx']          = adx_arr[i]

            # ── Taille de la cassure ─────────────────────────────────────────
            trades.loc[trade_i, 'breakout_size'] = (close[i] - r_val) / atr_arr[i]

            # ── Nombre de touches ────────────────────────────────────────────
            tolerance = atr_arr[i] * 0.5
            n_touches = np.sum(np.abs(window - line_vals) < tolerance)
            trades.loc[trade_i, 'n_touches']    = n_touches

            # ── Heure encodée circulairement ────────────────────────────────
            trades.loc[trade_i, 'hour_sin']     = hour_sin[i]
            trades.loc[trade_i, 'hour_cos']     = hour_cos[i]

            # ── Jour encodé circulairement ───────────────────────────────────
            trades.loc[trade_i, 'dow_sin']      = dow_sin[i]
            trades.loc[trade_i, 'dow_cos']      = dow_cos[i]

            # ── Momentum ─────────────────────────────────────────────────────
            trades.loc[trade_i, 'ret_24h']      = ret_24[i]   if not np.isnan(ret_24[i])  else 0.0
            trades.loc[trade_i, 'ret_1w']       = ret_168[i]  if not np.isnan(ret_168[i]) else 0.0

            # ── Régime de volatilité ─────────────────────────────────────────
            trades.loc[trade_i, 'vol_regime']   = vol_regime[i]

            # ── Position dans le range ───────────────────────────────────────
            trades.loc[trade_i, 'price_pos']    = price_position[i]

        # ── Sortie ───────────────────────────────────────────────────────────
        if in_trade:
            if close[i] >= tp_price or close[i] <= sl_price or i >= hp_i:
                trades.loc[trade_i, 'exit_i'] = i
                trades.loc[trade_i, 'exit_p'] = close[i]
                in_trade = False
                trade_i += 1

    trades['return'] = trades['exit_p'] - trades['entry_p']

    feature_cols = [
        # Trendline (originales)
        'resist_s', 'tl_err', 'max_dist', 'vol', 'adx',
        # Cassure
        'breakout_size', 'n_touches',
        # Temps (circulaire)
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        # Momentum
        'ret_24h', 'ret_1w',
        # Contexte marché
        'vol_regime', 'price_pos'
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

    print(f"Total trades  : {len(trades)}")
    print(f"Win rate base : {round(len(trades[trades['return'] > 0]) / len(trades), 4)}")
    print(f"Features      : {list(data_x.columns)}")