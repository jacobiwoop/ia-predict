"""
optimize_params.py
------------------
Teste différentes combinaisons de TP / SL / Hold Period / Lookback
pour trouver la configuration qui maximise le win rate et le PF de base
(SANS ML — on cherche d'abord un bon edge brut)
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from trendline_automation import fit_trendlines_single
import itertools


def run_fast_grid_search(data, lookbacks, hold_periods, tp_mults, sl_mults, atr_lookback=168):
    close = np.log(data['close'].to_numpy())
    atr = ta.atr(np.log(data['high']), np.log(data['low']),
                 np.log(data['close']), atr_lookback).to_numpy()

    results = []
    total = len(lookbacks) * len(hold_periods) * len(tp_mults) * len(sl_mults)
    done  = 0
    print(f"Test de {total} combinaisons en cours avec pré-calcul (TRÈS RAPIDE)...")

    for lb in lookbacks:
        print(f"  > Pré-calcul Mathématique des Trendlines pour lookback={lb}...")
        r_vals = np.full(len(close), np.nan)
        for i in range(atr_lookback, len(close)):
            window = close[i - lb: i]
            _, r_coefs = fit_trendlines_single(window)
            r_vals[i] = r_coefs[1] + lb * r_coefs[0]

        for hp, tp, sl in itertools.product(hold_periods, tp_mults, sl_mults):
            returns = []
            in_trade = False
            trade_tp = trade_sl = hp_i = None

            for i in range(atr_lookback, len(close)):
                r_val = r_vals[i]
                
                # Entrée
                if not in_trade and close[i] > r_val:
                    entry  = close[i]
                    trade_tp = entry + atr[i] * tp
                    trade_sl = entry - atr[i] * sl
                    hp_i   = i + hp
                    in_trade = True

                # Sortie
                if in_trade:
                    if close[i] >= trade_tp:
                        returns.append(close[i] - entry)
                        in_trade = False
                    elif close[i] <= trade_sl:
                        returns.append(close[i] - entry)
                        in_trade = False
                    elif i >= hp_i:
                        returns.append(close[i] - entry)
                        in_trade = False

            if len(returns) >= 20:
                r = np.array(returns)
                wins  = r[r > 0].sum()
                loses = np.abs(r[r < 0]).sum()
                pf    = wins / loses if loses > 0 else 0
                wr    = len(r[r > 0]) / len(r)
                avg   = r.mean()
                results.append({
                    'lookback': lb, 'hold': hp,
                    'tp': tp, 'sl': sl,
                    'pf': pf, 'wr': wr, 'avg': avg, 'n': len(r)
                })
            
            done += 1

    return results


if __name__ == '__main__':
    data = pd.read_csv('BTCUSDT3600.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')
    data = data.dropna()

    # ── Grille de paramètres à tester (Grille Complète Restaurée) ───────────
    lookbacks    = [48, 72, 96, 120]       # 2j, 3j, 4j, 5j
    hold_periods = [12, 24, 48, 72]        # 12h, 24h, 48h, 72h
    tp_mults     = [1.5, 2.0, 3.0]         # TP en ATR
    sl_mults     = [1.0, 1.5, 2.0, 3.0]    # SL en ATR

    results = run_fast_grid_search(data, lookbacks, hold_periods, tp_mults, sl_mults)

    df = pd.DataFrame(results)

    # ── Top 15 par Profit Factor ─────────────────────────────────────────────
    top = df.sort_values('pf', ascending=False).head(15)

    print("\n" + "=" * 70)
    print("  TOP 15 COMBINAISONS PAR PROFIT FACTOR")
    print("=" * 70)
    print(f"  {'Lookback':>8} {'Hold':>6} {'TP':>5} {'SL':>5} "
          f"{'PF':>7} {'WinRate':>8} {'AvgTrade':>10} {'N':>6}")
    print("-" * 70)
    for _, row in top.iterrows():
        print(f"  {int(row.lookback):>8} {int(row.hold):>6} "
              f"{row.tp:>5.1f} {row.sl:>5.1f} "
              f"{row.pf:>7.4f} {row.wr:>8.4f} "
              f"{row.avg:>10.6f} {int(row.n):>6}")

    # ── Top 15 par Win Rate ──────────────────────────────────────────────────
    top_wr = df.sort_values('wr', ascending=False).head(15)

    print("\n" + "=" * 70)
    print("  TOP 15 COMBINAISONS PAR WIN RATE")
    print("=" * 70)
    print(f"  {'Lookback':>8} {'Hold':>6} {'TP':>5} {'SL':>5} "
          f"{'PF':>7} {'WinRate':>8} {'AvgTrade':>10} {'N':>6}")
    print("-" * 70)
    for _, row in top_wr.iterrows():
        print(f"  {int(row.lookback):>8} {int(row.hold):>6} "
              f"{row.tp:>5.1f} {row.sl:>5.1f} "
              f"{row.pf:>7.4f} {row.wr:>8.4f} "
              f"{row.avg:>10.6f} {int(row.n):>6}")

    # ── Heatmap : Lookback vs Hold Period (PF moyen) ─────────────────────────
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Heatmap PF
    pivot_pf = df.groupby(['lookback', 'hold'])['pf'].mean().unstack()
    im1 = axes[0].imshow(pivot_pf.values, cmap='RdYlGn', aspect='auto')
    axes[0].set_xticks(range(len(pivot_pf.columns)))
    axes[0].set_xticklabels([f'{h}h' for h in pivot_pf.columns])
    axes[0].set_yticks(range(len(pivot_pf.index)))
    axes[0].set_yticklabels([f'{l}' for l in pivot_pf.index])
    axes[0].set_xlabel('Hold Period')
    axes[0].set_ylabel('Lookback')
    axes[0].set_title('Profit Factor moyen')
    plt.colorbar(im1, ax=axes[0])
    for i in range(len(pivot_pf.index)):
        for j in range(len(pivot_pf.columns)):
            axes[0].text(j, i, f'{pivot_pf.values[i,j]:.3f}',
                        ha='center', va='center', fontsize=8)

    # Heatmap Win Rate
    pivot_wr = df.groupby(['lookback', 'hold'])['wr'].mean().unstack()
    im2 = axes[1].imshow(pivot_wr.values, cmap='RdYlGn', aspect='auto')
    axes[1].set_xticks(range(len(pivot_wr.columns)))
    axes[1].set_xticklabels([f'{h}h' for h in pivot_wr.columns])
    axes[1].set_yticks(range(len(pivot_wr.index)))
    axes[1].set_yticklabels([f'{l}' for l in pivot_wr.index])
    axes[1].set_xlabel('Hold Period')
    axes[1].set_ylabel('Lookback')
    axes[1].set_title('Win Rate moyen')
    plt.colorbar(im2, ax=axes[1])
    for i in range(len(pivot_wr.index)):
        for j in range(len(pivot_wr.columns)):
            axes[1].text(j, i, f'{pivot_wr.values[i,j]:.3f}',
                        ha='center', va='center', fontsize=8)

    plt.suptitle('Optimisation Paramètres — BTC/USDT 1H', fontsize=13)
    plt.tight_layout()
    plt.savefig('param_optimization.png', dpi=120, bbox_inches='tight')
    plt.show()

    print("\n✓ Graphique sauvegardé : param_optimization.png")
    print("\n⚠️  ATTENTION : Ces résultats sont IN-SAMPLE.")
    print("   La meilleure combinaison doit être validée en walk-forward.")
    print("   Ne pas choisir uniquement selon le PF le plus élevé —")
    print("   privilégier les combinaisons STABLES sur plusieurs paramètres.")