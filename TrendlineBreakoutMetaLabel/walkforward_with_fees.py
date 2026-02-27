"""
walkforward_with_fees.py
------------------------
Version finale avec frais de transaction réalistes.

Frais Deriv (source : documentation officielle) :
  - Spread BTC/USD  : ~0.05% par entrée + 0.05% par sortie = 0.10% aller-retour
  - Spread ETH/USD  : ~0.05% par entrée + 0.05% par sortie = 0.10% aller-retour
  - Spread EUR/USD  : ~0.5 pip ≈ 0.005% aller-retour (très faible)
  - Swap overnight  : ~0.02% par nuit (hold_period=24h = 1 nuit)
  - Commission      : 0% (commission-free)

En log-returns :
  fee_per_trade = log(1 - spread_pct)
  swap_fee      = log(1 - overnight_pct)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from walkforward_multi import load_pair, walkforward_multi
from trendline_break_dataset import trendline_breakout_dataset


# ── Frais par paire (aller-retour complet) ───────────────────────────────────
FEES = {
    'BTC': {
        'spread_pct'   : 0.0010,   # 0.10% aller-retour (spread BTC/USD)
        'overnight_pct': 0.0002,   # 0.02% par nuit
        'nights'       : 1         # hold_period=24h = 1 nuit
    },
    'ETH': {
        'spread_pct'   : 0.0010,   # 0.10% aller-retour (spread ETH/USD)
        'overnight_pct': 0.0002,
        'nights'       : 1
    },
    'SOL': {
        'spread_pct'   : 0.0015,   # 0.15% (moins liquide)
        'overnight_pct': 0.0003,
        'nights'       : 1
    },
}

# Frais en log-return (plus précis que %)
def fee_log(spread_pct, overnight_pct, nights):
    spread_cost   = np.log(1 - spread_pct)
    overnight_cost = np.log(1 - overnight_pct) * nights
    return spread_cost + overnight_cost   # valeur négative


def apply_fees(trades: pd.DataFrame, pair: str) -> pd.DataFrame:
    """Soustrait les frais de chaque trade."""
    if pair not in FEES:
        print(f"  ⚠️  Frais non définis pour {pair} — frais ignorés")
        return trades

    f = FEES[pair]
    fee = fee_log(f['spread_pct'], f['overnight_pct'], f['nights'])
    trades = trades.copy()
    trades['return_gross'] = trades['return']
    trades['return']       = trades['return'] + fee   # fee est négatif
    trades['fee']          = fee
    return trades


def summary(trades_raw, trades_fees, name):
    """Affiche les stats avant et après frais."""

    def stats(t, label):
        t = t.dropna(subset=['model_prob'])
        all_r = t['return']
        ml_r  = t[t['model_prob'] > 0.5]['return']

        wins_all  = all_r[all_r > 0].sum()
        loses_all = all_r[all_r < 0].abs().sum()
        pf_all    = round(wins_all / loses_all, 4) if loses_all > 0 else 0

        wins_ml  = ml_r[ml_r > 0].sum()
        loses_ml = ml_r[ml_r < 0].abs().sum()
        pf_ml    = round(wins_ml / loses_ml, 4) if loses_ml > 0 else 0
        wr_ml    = round(len(ml_r[ml_r > 0]) / len(ml_r), 4) if len(ml_r) > 0 else 0

        return pf_all, pf_ml, wr_ml, len(all_r), len(ml_r)

    pf_all_g, pf_ml_g, wr_ml_g, n_all, n_ml = stats(trades_raw,  "Brut")
    pf_all_f, pf_ml_f, wr_ml_f, _,    _     = stats(trades_fees, "Avec frais")

    fee_impact_all = round(pf_all_f - pf_all_g, 4)
    fee_impact_ml  = round(pf_ml_f  - pf_ml_g,  4)

    f = FEES.get(name, {})
    spread = f.get('spread_pct', 0) * 100
    swap   = f.get('overnight_pct', 0) * 100

    print(f"\n{'='*60}")
    print(f"  {name} — Impact des frais")
    print(f"  Spread : {spread:.2f}%  |  Swap/nuit : {swap:.3f}%")
    print(f"{'='*60}")
    print(f"  {'':20} {'SANS ML':>10} {'AVEC ML':>10}")
    print(f"  {'-'*45}")
    print(f"  {'PF brut (sans frais)':20} {pf_all_g:>10.4f} {pf_ml_g:>10.4f}")
    print(f"  {'PF net  (avec frais)':20} {pf_all_f:>10.4f} {pf_ml_f:>10.4f}")
    print(f"  {'Impact frais':20} {fee_impact_all:>10.4f} {fee_impact_ml:>10.4f}")
    print(f"  {'Win Rate (avec frais)':20} {'—':>10} {wr_ml_f:>10.4f}")
    print(f"  {'N trades':20} {n_all:>10} {n_ml:>10}")

    viable = "✅ VIABLE" if pf_ml_f > 1.05 else \
             "⚠️  MARGINAL" if pf_ml_f > 1.0 else \
             "❌ NON VIABLE"
    print(f"\n  → Verdict : {viable}  (PF net ML = {pf_ml_f})")

    return {
        'pf_brut_ml' : pf_ml_g,
        'pf_net_ml'  : pf_ml_f,
        'wr_ml'      : wr_ml_f,
        'n_ml'       : n_ml,
        'viable'     : pf_ml_f > 1.0
    }


if __name__ == '__main__':

    # ── Charger les données ──────────────────────────────────────────────────
    pairs_data = {}
    if os.path.exists('BTCUSDT3600.csv'):
        pairs_data['BTC'] = load_pair('BTCUSDT3600.csv')
    for sym in ['ETH', 'SOL']:
        path = f"data/{sym}USDT3600.csv"
        if os.path.exists(path):
            pairs_data[sym] = load_pair(path)

    print(f"Paires : {list(pairs_data.keys())}")

    # ── Walk-forward ─────────────────────────────────────────────────────────
    print("\nCalcul du walk-forward...")
    results = walkforward_multi(
        pairs_data,
        lookback    = 72,
        hold_period = 24,
        train_size  = 365 * 24 * 2,
        step_size   = 365 * 24
    )

    # ── Appliquer les frais et comparer ──────────────────────────────────────
    verdicts = {}
    plot_data = {}

    for name, res in results.items():
        trades_raw  = res['trades'].copy()
        trades_fees = apply_fees(trades_raw, name)

        verdicts[name] = summary(trades_raw, trades_fees, name)

        # Préparer données pour le graphique
        df = res['df'].copy()
        df = df[df.index > '2020-01-01']
        df['r'] = np.log(df['close']).diff().shift(-1)

        sig  = pd.Series(res['signal'],      index=res['df'].index)
        dumb = pd.Series(res['dumb_signal'], index=res['df'].index)
        sig  = sig[sig.index   > '2020-01-01']
        dumb = dumb[dumb.index > '2020-01-01']

        # Frais par bougie = fee total / hold_period (réparti uniformément)
        f = FEES.get(name, {})
        fee_total   = fee_log(
            f.get('spread_pct', 0),
            f.get('overnight_pct', 0),
            f.get('nights', 1)
        )
        hold = 24
        fee_per_bar = fee_total / hold

        filter_gross = df['r'] * sig
        filter_net   = filter_gross + (sig * fee_per_bar)
        dumb_gross   = df['r'] * dumb
        dumb_net     = dumb_gross + (dumb * fee_per_bar)

        plot_data[name] = {
            'filter_gross': filter_gross,
            'filter_net'  : filter_net,
            'dumb_gross'  : dumb_gross,
            'dumb_net'    : dumb_net
        }

    # ── Graphiques ───────────────────────────────────────────────────────────
    plt.style.use('dark_background')
    n = len(plot_data)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1: axes = [axes]

    colors = {'BTC': 'cyan', 'ETH': 'lime', 'SOL': 'yellow'}

    for idx, (name, pd_) in enumerate(plot_data.items()):
        ax    = axes[idx]
        color = colors.get(name, 'white')
        v     = verdicts[name]

        pd_['filter_gross'].cumsum().plot(
            ax=ax, color=color, linewidth=2,
            label=f'ML brut  (PF={v["pf_brut_ml"]})'
        )
        pd_['filter_net'].cumsum().plot(
            ax=ax, color=color, linewidth=2, linestyle='--',
            label=f'ML net   (PF={v["pf_net_ml"]})'
        )
        pd_['dumb_gross'].cumsum().plot(
            ax=ax, color='gray', linewidth=1, alpha=0.6,
            label='Sans ML brut'
        )
        pd_['dumb_net'].cumsum().plot(
            ax=ax, color='gray', linewidth=1, alpha=0.4, linestyle='--',
            label='Sans ML net'
        )

        verdict = "✅ VIABLE" if v['viable'] else "❌ NON VIABLE"
        ax.set_title(f"{name} — {verdict}", fontsize=11)
        ax.set_ylabel("Cumulative Log Return")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

        # Zone de frais (différence brut - net)
        ax.fill_between(
            pd_['filter_gross'].cumsum().index,
            pd_['filter_gross'].cumsum(),
            pd_['filter_net'].cumsum(),
            alpha=0.15, color='red', label='Impact frais'
        )

    plt.suptitle("Impact des Frais — Backtest Réaliste", fontsize=13)
    plt.tight_layout()
    plt.savefig('backtest_with_fees.png', dpi=120, bbox_inches='tight')
    plt.show()

    # ── Résumé final ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RÉSUMÉ FINAL — APRÈS FRAIS DERIV")
    print("=" * 60)
    print(f"  {'Paire':<6} {'PF brut':>9} {'PF net':>9} "
          f"{'WR net':>8} {'N trades':>10} {'Verdict':>14}")
    print(f"  {'-'*58}")
    for name, v in verdicts.items():
        print(f"  {name:<6} {v['pf_brut_ml']:>9.4f} {v['pf_net_ml']:>9.4f} "
              f"{v['wr_ml']:>8.4f} {v['n_ml']:>10} "
              f"{'✅ VIABLE' if v['viable'] else '❌ NON VIABLE':>14}")
    print("=" * 60)
    print("\n✓ Graphique sauvegardé : backtest_with_fees.png")