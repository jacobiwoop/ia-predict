"""
optimize_threshold.py
---------------------
Trouve le seuil de probabilité optimal pour chaque paire.
Teste les seuils de 0.30 à 0.70 par pas de 0.02.
Inclut V75 avec avertissement synthétique.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from walkforward_multi import load_pair, walkforward_multi
from trendline_break_dataset import trendline_breakout_dataset
import xgboost as xgb


def evaluate_threshold(trades, threshold):
    trades = trades.dropna(subset=['model_prob'])
    ml_trades = trades[trades['model_prob'] > threshold]
    if len(ml_trades) < 10:
        return None
    r     = ml_trades['return']
    wins  = r[r > 0].sum()
    loses = r[r < 0].abs().sum()
    return {
        'pf' : round(wins / loses, 4) if loses > 0 else 0,
        'wr' : round(len(r[r > 0]) / len(r), 4),
        'avg': round(r.mean(), 6),
        'n'  : len(r)
    }


def analyze_pair(name, trades, thresholds, is_synthetic=False):
    """Calcule PF/WR/N pour chaque seuil et retourne le meilleur."""
    pf_list, wr_list, n_list = [], [], []

    for thr in thresholds:
        ev = evaluate_threshold(trades, thr)
        if ev:
            pf_list.append(ev['pf'])
            wr_list.append(ev['wr'])
            n_list.append(ev['n'])
        else:
            pf_list.append(0)
            wr_list.append(0)
            n_list.append(0)

    trades_clean = trades.dropna(subset=['model_prob'])
    min_trades   = max(20, int(len(trades_clean) * 0.20))

    best_pf = 0
    best_thr = 0.5
    for i, thr in enumerate(thresholds):
        if n_list[i] >= min_trades and pf_list[i] > best_pf:
            best_pf  = pf_list[i]
            best_thr = round(thr, 2)

    # ── Affichage console ────────────────────────────────────────────────────
    label = f"{name} ⚠️  SYNTHÉTIQUE" if is_synthetic else name
    print(f"\n{'='*58}")
    print(f"  {label} — Analyse des seuils")
    print(f"{'='*58}")
    if is_synthetic:
        print(f"  ⚠️  Modèle entraîné sur crypto réelle — interpréter avec")
        print(f"      précaution. Le V75 est généré algorithmiquement.")
        print(f"  {'-'*50}")
    print(f"  {'Seuil':>7} {'PF':>7} {'WinRate':>8} {'N trades':>9}")
    print(f"  {'-'*40}")
    for i, thr in enumerate(thresholds):
        marker = " ← OPTIMAL" if round(thr, 2) == best_thr else ""
        if n_list[i] > 0:
            print(f"  {thr:>7.2f} {pf_list[i]:>7.4f} "
                  f"{wr_list[i]:>8.4f} {n_list[i]:>9}{marker}")
    print(f"\n  → Seuil optimal : {best_thr}  "
          f"(PF={best_pf}, min {min_trades} trades requis)")

    return pf_list, wr_list, n_list, best_thr, best_pf, min_trades


if __name__ == '__main__':

    # ── Charger les données réelles ──────────────────────────────────────────
    pairs_data = {}
    if os.path.exists('BTCUSDT3600.csv'):
        pairs_data['BTC'] = load_pair('BTCUSDT3600.csv')
        print(f"✓ BTC chargé")
    for sym in ['ETH', 'SOL']:
        path = f"data/{sym}USDT3600.csv"
        if os.path.exists(path):
            pairs_data[sym] = load_pair(path)
            print(f"✓ {sym} chargé")

    # ── Charger V75 séparément ───────────────────────────────────────────────
    v75_data = None
    v75_path = "data/V75USDT3600.csv"
    if os.path.exists(v75_path):
        v75_data = load_pair(v75_path)
        print(f"✓ V75 chargé (synthétique)")
    else:
        print(f"⚠️  V75 non trouvé — lance download_v75.py pour l'inclure")

    print(f"\nPaires réelles    : {list(pairs_data.keys())}")
    print(f"Indice synthétique: {'V75' if v75_data is not None else 'non disponible'}")

    # ── Walk-forward sur paires réelles ──────────────────────────────────────
    print("\nCalcul des probabilités (walk-forward sur crypto réelle)...")
    results = walkforward_multi(
        pairs_data,
        lookback    = 72,
        hold_period = 24,
        train_size  = 365 * 24 * 2,
        step_size   = 365 * 24
    )

    # ── Si V75 disponible : appliquer le DERNIER modèle entraîné ─────────────
    if v75_data is not None:
        print("\nApplication du modèle crypto sur V75...")
        last_model = list(results.values())[-1]['model']

        # Générer les trades V75
        v75_trades, v75_x, v75_y = trendline_breakout_dataset(
            v75_data, lookback=72, hold_period=24
        )
        v75_trades = v75_trades.dropna()

        if len(v75_trades) > 0:
            # Appliquer le modèle entraîné sur crypto réelle
            probs = last_model.predict_proba(
                v75_x.loc[v75_trades.index].to_numpy()
            )[:, 1]
            v75_trades['model_prob'] = probs
            print(f"  {len(v75_trades)} trades V75 scorés")
        else:
            v75_data = None
            print("  ❌ Aucun trade détecté sur V75")

    # ── Analyse des seuils ───────────────────────────────────────────────────
    thresholds  = np.arange(0.30, 0.71, 0.02)
    all_results = {}

    # Paires réelles
    for name, res in results.items():
        pf_list, wr_list, n_list, best_thr, best_pf, min_t = analyze_pair(
            name, res['trades'], thresholds, is_synthetic=False
        )
        all_results[name] = {
            'pf': pf_list, 'wr': wr_list, 'n': n_list,
            'best_thr': best_thr, 'best_pf': best_pf,
            'min_t': min_t, 'synthetic': False
        }

    # V75 synthétique
    if v75_data is not None:
        pf_list, wr_list, n_list, best_thr, best_pf, min_t = analyze_pair(
            'V75', v75_trades, thresholds, is_synthetic=True
        )
        all_results['V75'] = {
            'pf': pf_list, 'wr': wr_list, 'n': n_list,
            'best_thr': best_thr, 'best_pf': best_pf,
            'min_t': min_t, 'synthetic': True
        }

    # ── Graphiques ───────────────────────────────────────────────────────────
    n_pairs = len(all_results)
    plt.style.use('dark_background')
    fig, axes = plt.subplots(n_pairs, 3, figsize=(16, 4 * n_pairs))
    if n_pairs == 1:
        axes = [axes]

    colors = {
        'BTC': 'cyan',
        'ETH': 'lime',
        'SOL': 'yellow',
        'V75': 'orange'   # couleur distincte pour le synthétique
    }

    thr_list = list(thresholds)

    for row_idx, (name, res) in enumerate(all_results.items()):
        color    = colors.get(name, 'white')
        ax_pf    = axes[row_idx][0]
        ax_wr    = axes[row_idx][1]
        ax_n     = axes[row_idx][2]
        best_thr = res['best_thr']
        label    = f"{name} {'⚠️ SYNTHÉTIQUE' if res['synthetic'] else ''}"

        # PF
        ax_pf.plot(thr_list, res['pf'], color=color, linewidth=2)
        ax_pf.axvline(best_thr, color='white', linestyle='--',
                      label=f'Optimal={best_thr}')
        ax_pf.axhline(1.0, color='red', linestyle=':', alpha=0.5)
        ax_pf.set_title(f'{label} — Profit Factor vs Seuil')
        ax_pf.set_xlabel('Seuil de probabilité')
        ax_pf.set_ylabel('Profit Factor')
        ax_pf.legend(fontsize=8)
        ax_pf.grid(alpha=0.2)
        if res['synthetic']:
            ax_pf.set_facecolor('#1a1000')  # fond orange foncé pour V75

        # Win Rate
        ax_wr.plot(thr_list, res['wr'], color=color, linewidth=2)
        ax_wr.axvline(best_thr, color='white', linestyle='--',
                      label=f'Optimal={best_thr}')
        ax_wr.axhline(0.5, color='red', linestyle=':', alpha=0.5)
        ax_wr.set_title(f'{label} — Win Rate vs Seuil')
        ax_wr.set_xlabel('Seuil de probabilité')
        ax_wr.set_ylabel('Win Rate')
        ax_wr.legend(fontsize=8)
        ax_wr.grid(alpha=0.2)
        if res['synthetic']:
            ax_wr.set_facecolor('#1a1000')

        # N trades
        ax_n.plot(thr_list, res['n'], color=color, linewidth=2)
        ax_n.axvline(best_thr, color='white', linestyle='--',
                     label=f'Optimal={best_thr}')
        ax_n.axhline(res['min_t'], color='red', linestyle=':',
                     alpha=0.5, label=f'Min={res["min_t"]}')
        ax_n.set_title(f'{label} — N Trades vs Seuil')
        ax_n.set_xlabel('Seuil de probabilité')
        ax_n.set_ylabel('Nombre de trades')
        ax_n.legend(fontsize=8)
        ax_n.grid(alpha=0.2)
        if res['synthetic']:
            ax_n.set_facecolor('#1a1000')

    plt.suptitle('Optimisation du Seuil de Probabilité — Crypto + V75',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig('threshold_optimization.png', dpi=120, bbox_inches='tight')
    plt.show()

    # ── Résumé final ─────────────────────────────────────────────────────────
    print("\n" + "=" * 58)
    print("  RÉSUMÉ — SEUILS OPTIMAUX PAR PAIRE")
    print("=" * 58)
    for name, res in all_results.items():
        tag = " ⚠️  (synthétique — prudence)" if res['synthetic'] else ""
        print(f"  {name:<6} → seuil optimal = {res['best_thr']}"
              f"  PF={res['best_pf']}{tag}")
    print("=" * 58)
    print("\n⚠️  Ces seuils sont IN-SAMPLE — à valider en out-of-sample.")

    # Sauvegarder
    best = {n: r['best_thr'] for n, r in all_results.items()}
    pd.Series(best).to_csv('best_thresholds.csv', header=['threshold'])
    print("✓ Seuils sauvegardés : best_thresholds.csv")
    print("✓ Graphique sauvegardé : threshold_optimization.png")