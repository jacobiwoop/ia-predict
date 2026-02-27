import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import xgboost as xgb
from base_strategy import Strategy


def walkforward_model(
        close: np.array, trades: pd.DataFrame,
        data_x: pd.DataFrame, data_y: pd.Series,
        train_size: int, step_size: int
):
    signal      = np.zeros(len(close))
    dumb_signal = np.zeros(len(close))

    next_train = train_size
    trade_i    = 0

    in_trade_ml   = False
    in_trade_dumb = False

    tp_price_ml   = sl_price_ml   = hp_i_ml   = None
    tp_price_dumb = sl_price_dumb = hp_i_dumb = None

    last_model = None  # on garde le dernier modèle pour l'analyse

    for i in range(len(close)):

        # ── 1. Retraining ───────────────────────────────────────────────────
        if i == next_train:
            start_i = i - train_size
            train_indices = trades[
                (trades['entry_i'] > start_i) & (trades['exit_i'] < i)
            ].index
            x_train = data_x.loc[train_indices]
            y_train = data_y.loc[train_indices]
            print(f'Training  i={i}  N trades={len(train_indices)}')
            last_model = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='logloss',
                random_state=42
            )
            last_model.fit(x_train.to_numpy(), y_train.to_numpy())
            next_train += step_size

        # ── 2. Sortie trade ML ──────────────────────────────────────────────
        if in_trade_ml:
            if close[i] >= tp_price_ml or close[i] <= sl_price_ml or i >= hp_i_ml:
                signal[i]   = 0
                in_trade_ml = False
            else:
                signal[i] = 1

        # ── 3. Sortie trade DUMB ────────────────────────────────────────────
        if in_trade_dumb:
            if close[i] >= tp_price_dumb or close[i] <= sl_price_dumb or i >= hp_i_dumb:
                dumb_signal[i] = 0
                in_trade_dumb  = False
            else:
                dumb_signal[i] = 1

        # ── 4. Entrée potentielle ───────────────────────────────────────────
        if trade_i < len(trades) and i == int(trades['entry_i'].iloc[trade_i]):

            trade = trades.iloc[trade_i]

            if last_model is not None and not in_trade_dumb:
                dumb_signal[i] = 1
                in_trade_dumb  = True
                tp_price_dumb  = trade['tp']
                sl_price_dumb  = trade['sl']
                hp_i_dumb      = int(trade['hp_i'])

            if last_model is not None and not in_trade_ml:
                prob = last_model.predict_proba(
                    data_x.iloc[trade_i].to_numpy().reshape(1, -1)
                )[0][1]
                trades.loc[trade_i, 'model_prob'] = prob

                if prob > 0.5:
                    signal[i]   = 1
                    in_trade_ml = True
                    tp_price_ml = trade['tp']
                    sl_price_ml = trade['sl']
                    hp_i_ml     = int(trade['hp_i'])

            trade_i += 1

    return signal, dumb_signal, last_model  # ← on retourne aussi le modèle


# ════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    data = pd.read_csv('BTCUSDT3600.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')
    data = data.dropna()

    from strategies.trendline_strategy import TrendlineBreakoutStrategy
    strategy = TrendlineBreakoutStrategy(lookback=72, hold_period=24)
    trades, data_x, data_y = strategy.generate_dataset(data)

    signal, dumb_signal, model = walkforward_model(
        np.log(data['close']).to_numpy(),
        trades, data_x, data_y,
        train_size=365 * 24 * 2,
        step_size=365 * 24
    )

    data['sig']      = signal
    data['dumb_sig'] = dumb_signal

    data = data[data.index > '2020-01-01']
    data['r'] = np.log(data['close']).diff().shift(-1)

    filter_rets    = data['r'] * data['sig']
    no_filter_rets = data['r'] * data['dumb_sig']

    trades        = trades.dropna(subset=['model_prob'])
    all_trades_r  = trades['return']
    ml_trades_r   = trades[trades['model_prob'] > 0.5]['return']

    def prof_factor(rets):
        neg = rets[rets < 0].abs().sum()
        return round(rets[rets > 0].sum() / neg, 4) if neg > 0 else 0

    def win_rate(rets):
        return round(len(rets[rets > 0]) / len(rets), 4) if len(rets) > 0 else 0

    print("=" * 48)
    print("  SANS FILTRE ML  (toutes les cassures)")
    print("=" * 48)
    print(f"  Profit Factor  : {prof_factor(no_filter_rets)}")
    print(f"  Avg Trade      : {round(all_trades_r.mean(), 6)}")
    print(f"  Win Rate       : {win_rate(all_trades_r)}")
    print(f"  N Trades       : {len(all_trades_r)}")
    print(f"  Time in Market : {round(len(data[data['dumb_sig'] > 0]) / len(data), 4)}")

    print("=" * 48)
    print("  AVEC FILTRE ML  (meta-label XGBoost)")
    print("=" * 48)
    print(f"  Profit Factor  : {prof_factor(filter_rets)}")
    print(f"  Avg Trade      : {round(ml_trades_r.mean(), 6)}")
    print(f"  Win Rate       : {win_rate(ml_trades_r)}")
    print(f"  N Trades       : {len(ml_trades_r)}")
    print(f"  Time in Market : {round(len(data[data['sig'] > 0]) / len(data), 4)}")
    print("=" * 48)

    # ── Graphique 1 : Courbe de rendement ────────────────────────────────────
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    ax1 = axes[0]
    filter_rets.cumsum().plot(ax=ax1, label='Meta-Label XGBoost', color='cyan')
    no_filter_rets.cumsum().plot(ax=ax1, label='Sans filtre ML', color='orange')
    data['r'].cumsum().plot(ax=ax1, label='Buy & Hold', color='white', alpha=0.4)
    ax1.set_title("Walk-Forward Backtest — BTC/USDT 1H")
    ax1.set_ylabel("Cumulative Log Return")
    ax1.legend()

    # ── Graphique 2 : Importance des features ────────────────────────────────
    ax2 = axes[1]
    importances = pd.Series(
        model.feature_importances_,
        index=data_x.columns
    ).sort_values(ascending=True)

    # Colorier en rouge les features faibles (< 5%) et en vert les fortes (>= 5%)
    colors = ['red' if v < 0.05 else 'cyan' for v in importances.values]
    importances.plot.barh(ax=ax2, color=colors)
    ax2.axvline(x=0.05, color='white', linestyle='--', alpha=0.5, label='Seuil 5%')
    ax2.set_title("Importance des features (rouge = < 5% = candidat à supprimer)")
    ax2.set_xlabel("Importance")
    ax2.legend()

    # Afficher les valeurs sur les barres
    for idx, (val, name) in enumerate(zip(importances.values, importances.index)):
        ax2.text(val + 0.001, idx, f'{val:.1%}', va='center', fontsize=9, color='white')

    plt.tight_layout()
    plt.show()

    # ── Résumé texte des features ─────────────────────────────────────────────
    print("\n" + "=" * 48)
    print("  IMPORTANCE DES FEATURES")
    print("=" * 48)
    for name, val in importances.sort_values(ascending=False).items():
        status = "✓ GARDER " if val >= 0.05 else "✗ SUPPRIMER"
        print(f"  {status}  {name:<15} : {val:.1%}")
    print("=" * 48)