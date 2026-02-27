"""
walkforward_multi.py
--------------------
Version multi-paires du walk-forward.
Entraîne le modèle sur BTC + ETH + SOL combinés
→ 3x plus de trades → modèle ML plus fiable
→ Évalue sur chaque paire séparément
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from trendline_break_dataset import trendline_breakout_dataset
import xgboost as xgb
import os


def load_pair(filepath: str) -> pd.DataFrame:
    """Charge un fichier CSV OHLCV"""
    data = pd.read_csv(filepath)
    # Compatible avec les deux formats (Binance et original)
    date_col = 'date' if 'date' in data.columns else data.columns[0]
    data[date_col] = data[date_col].astype('datetime64[s]')
    data = data.set_index(date_col)
    data = data[['open', 'high', 'low', 'close', 'volume']]
    return data.dropna()


def walkforward_multi(
        pairs_data: dict,          # {'BTC': df, 'ETH': df, 'SOL': df}
        lookback: int = 72,
        hold_period: int = 24,     # Config A : 24h
        train_size: int = 365*24*2,
        step_size: int  = 365*24
):
    """
    Entraîne sur toutes les paires combinées.
    Évalue sur chaque paire séparément.
    """

    # ── 1. Générer le dataset pour chaque paire ──────────────────────────────
    print("Génération des datasets...")
    all_trades = {}
    all_data_x = {}
    all_data_y = {}

    for name, df in pairs_data.items():
        print(f"  → {name} ({len(df)} bougies)")
        trades, data_x, data_y = trendline_breakout_dataset(
            df, lookback, hold_period=hold_period
        )
        all_trades[name] = trades
        all_data_x[name] = data_x
        all_data_y[name] = data_y
        print(f"     {len(trades)} trades détectés")

    # ── 2. Walk-forward sur chaque paire ─────────────────────────────────────
    results = {}

    for eval_name, eval_df in pairs_data.items():
        print(f"\nWalk-forward sur {eval_name}...")

        close      = np.log(eval_df['close'].to_numpy())
        trades     = all_trades[eval_name].copy()
        data_x     = all_data_x[eval_name]
        data_y     = all_data_y[eval_name]

        signal      = np.zeros(len(close))
        dumb_signal = np.zeros(len(close))

        next_train    = train_size
        trade_i       = 0
        in_trade_ml   = False
        in_trade_dumb = False
        tp_ml = sl_ml = hp_ml = None
        tp_du = sl_du = hp_du = None
        last_model    = None

        for i in range(len(close)):

            # Retraining : combine TOUTES les paires disponibles jusqu'à i
            if i == next_train:
                start_i = i - train_size
                combined_x = []
                combined_y = []

                for src_name in pairs_data:
                    src_trades = all_trades[src_name]
                    src_x      = all_data_x[src_name]
                    src_y      = all_data_y[src_name]

                    # On prend les trades dans la fenêtre temporelle
                    # Note : on utilise entry_i comme proxy temporel
                    # (approximation — idéalement aligner les timestamps)
                    idx = src_trades[
                        (src_trades['entry_i'] > start_i) &
                        (src_trades['exit_i']  < i)
                    ].index
                    if len(idx) > 0:
                        combined_x.append(src_x.loc[idx])
                        combined_y.append(src_y.loc[idx])

                if combined_x:
                    X = pd.concat(combined_x)
                    Y = pd.concat(combined_y)
                    print(f"  Training i={i}  N={len(X)} trades "
                          f"({', '.join(pairs_data.keys())})")
                    last_model = xgb.XGBClassifier(
                        n_estimators=500, max_depth=3,
                        learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, eval_metric='logloss',
                        random_state=42
                    )
                    last_model.fit(X.to_numpy(), Y.to_numpy())

                next_train += step_size

            # Sortie ML
            if in_trade_ml:
                if close[i] >= tp_ml or close[i] <= sl_ml or i >= hp_ml:
                    signal[i] = 0; in_trade_ml = False
                else:
                    signal[i] = 1

            # Sortie DUMB
            if in_trade_dumb:
                if close[i] >= tp_du or close[i] <= sl_du or i >= hp_du:
                    dumb_signal[i] = 0; in_trade_dumb = False
                else:
                    dumb_signal[i] = 1

            # Entrée
            if trade_i < len(trades) and i == int(trades['entry_i'].iloc[trade_i]):
                trade = trades.iloc[trade_i]

                if last_model is not None and not in_trade_dumb:
                    dumb_signal[i] = 1; in_trade_dumb = True
                    tp_du = trade['tp']; sl_du = trade['sl']
                    hp_du = int(trade['hp_i'])

                if last_model is not None and not in_trade_ml:
                    prob = last_model.predict_proba(
                        data_x.iloc[trade_i].to_numpy().reshape(1, -1)
                    )[0][1]
                    trades.loc[trade_i, 'model_prob'] = prob
                    if prob > 0.5:
                        signal[i] = 1; in_trade_ml = True
                        tp_ml = trade['tp']; sl_ml = trade['sl']
                        hp_ml = int(trade['hp_i'])

                trade_i += 1

        results[eval_name] = {
            'signal': signal,
            'dumb_signal': dumb_signal,
            'trades': trades,
            'data_x': data_x,
            'close': close,
            'df': eval_df,
            'model': last_model
        }

    return results


# ════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':

    # ── Charger les données disponibles ─────────────────────────────────────
    pairs_data = {}

    # BTC (fichier original)
    if os.path.exists('BTCUSDT3600.csv'):
        pairs_data['BTC'] = load_pair('BTCUSDT3600.csv')
        print(f"✓ BTC chargé : {len(pairs_data['BTC'])} bougies")

    # ETH et SOL (téléchargés)
    for sym in ['ETH', 'SOL']:
        path = f"data/{sym}USDT3600.csv"
        if os.path.exists(path):
            pairs_data[sym] = load_pair(path)
            print(f"✓ {sym} chargé : {len(pairs_data[sym])} bougies")
        else:
            print(f"⚠️  {sym} non trouvé — lance d'abord download_data.py")

    if len(pairs_data) == 0:
        print("❌ Aucune donnée disponible.")
        exit()

    print(f"\n→ Paires disponibles : {list(pairs_data.keys())}")

    # ── Walk-forward multi-paires ────────────────────────────────────────────
    results = walkforward_multi(
        pairs_data,
        lookback     = 72,
        hold_period  = 24,
        train_size   = 365 * 24 * 2,
        step_size    = 365 * 24
    )

    # ── Affichage des résultats ──────────────────────────────────────────────
    def prof_factor(rets):
        neg = rets[rets < 0].abs().sum()
        return round(rets[rets > 0].sum() / neg, 4) if neg > 0 else 0

    def win_rate(rets):
        return round(len(rets[rets > 0]) / len(rets), 4) if len(rets) > 0 else 0

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 6))
    colors_ml   = ['cyan', 'lime', 'yellow']
    colors_dumb = ['orange', 'salmon', 'wheat']

    for idx, (name, res) in enumerate(results.items()):
        df   = res['df'].copy()
        df   = df[df.index > '2020-01-01']
        df['r'] = np.log(df['close']).diff().shift(-1)

        sig_series  = pd.Series(res['signal'],      index=res['df'].index)
        dumb_series = pd.Series(res['dumb_signal'], index=res['df'].index)
        sig_series  = sig_series[sig_series.index > '2020-01-01']
        dumb_series = dumb_series[dumb_series.index > '2020-01-01']

        filter_rets    = df['r'] * sig_series
        no_filter_rets = df['r'] * dumb_series

        trades     = res['trades'].dropna(subset=['model_prob'])
        all_r      = trades['return']
        ml_r       = trades[trades['model_prob'] > 0.5]['return']

        print(f"\n{'='*50}")
        print(f"  {name}USDT — Résultats")
        print(f"{'='*50}")
        print(f"  SANS ML  PF={prof_factor(no_filter_rets)}  "
              f"WR={win_rate(all_r)}  N={len(all_r)}")
        print(f"  AVEC ML  PF={prof_factor(filter_rets)}  "
              f"WR={win_rate(ml_r)}  N={len(ml_r)}")

        filter_rets.cumsum().plot(
            ax=ax, label=f'{name} ML',
            color=colors_ml[idx % len(colors_ml)]
        )
        no_filter_rets.cumsum().plot(
            ax=ax, label=f'{name} Sans ML',
            color=colors_dumb[idx % len(colors_dumb)],
            alpha=0.5, linestyle='--'
        )

    ax.set_title("Walk-Forward Multi-Paires — BTC + ETH + SOL")
    ax.set_ylabel("Cumulative Log Return")
    ax.legend()
    plt.tight_layout()
    plt.show()