"""
paper_trading_backtest.py
-------------------------
Simulateur de gestion de portefeuille ("Paper Trading" historique)
basé sur les prédictions du modèle XGBoost et les seuils optimisés.
Gère un solde en dollars réels, dimensionne les positions, 
et applique les frais Deriv.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates

from walkforward_multi import load_pair, walkforward_multi
from walkforward_with_fees import apply_fees, FEES

# Seuils ML optimisés empiriquement (Profit Factor maximum)
THRESHOLDS = {
    'BTC': 0.52,
    'ETH': 0.38,
    'SOL': 0.64
}

INITIAL_BALANCE = 10000.0   # Capital de départ en USD
RISK_PER_TRADE  = 0.03      # On risque 3% du capital par trade sur le Stop Loss respectif
MAX_LEVERAGE    = 3.0       # Levier maximum autorisé pour contrôler le risque global


def run_portfolio_simulation():
    print(f"\nSimulation de Portefeuille (Capital: ${INITIAL_BALANCE:,.0f} | Risque/Trade: {RISK_PER_TRADE*100}%)\n" + "="*70)
    
    # 1. Charger les données cibles
    pairs_data = {}
    for sym in ['BTC', 'ETH', 'SOL']:
        path = f"data/{sym}USDT3600.csv" if sym != 'BTC' else 'BTCUSDT3600.csv'
        if os.path.exists(path):
            pairs_data[sym] = load_pair(path)
            
    if not pairs_data:
        print("❌ Aucune donnée de marché disponible.")
        return

    # 2. Exécuter le modèle d'Intelligence Artificielle sur l'historique
    print("🧠 Génération des prédictions Walk-Forward XGBoost...")
    results = walkforward_multi(
        pairs_data,
        lookback    = 72,
        hold_period = 24,
        train_size  = 365 * 24 * 2,
        step_size   = 365 * 24,
        thresholds  = THRESHOLDS
    )

    # 3. Récolter tous les trades viables de l'IA
    all_trades = []
    print("\nFiltre des positions via les seuils IA...")
    for name, res in results.items():
        df     = res['df']
        trades = res['trades'].dropna(subset=['model_prob']).copy()
        
        # 3.a : Appliquer les frais réels logiques ( spread + swap par Deriv )
        trades = apply_fees(trades, name)
        
        # 3.b : Ne garder QUE les trades autorisés par l'IA (prob >= seuil d'exigence)
        thresh = THRESHOLDS.get(name, 0.5)
        filtered_trades = trades[trades['model_prob'] >= thresh].copy()
        print(f"  → {name:3} autorisé: {len(filtered_trades)} trades (Seuil: {thresh})")
        
        # Exclure les trades ouverts à la toute fin du dataset qui n'ont pas encore de date de sortie
        filtered_trades = filtered_trades.dropna(subset=['entry_i', 'exit_i']).copy()
        
        if len(filtered_trades) > 0:
            # 3.c : Traduire les dates pour le simulateur de compte
            filtered_trades['entry_date'] = df.index[filtered_trades['entry_i'].astype(int)]
            filtered_trades['exit_date']  = df.index[filtered_trades['exit_i'].astype(int)]
            filtered_trades['pair']       = name
            all_trades.append(filtered_trades)

    if not all_trades:
        print("\n❌ Aucun trade validé par l'IA.")
        return

    # Fusionner le carnet d'ordres
    # On trie chronologiquement par DATE DE SORTIE pour simuler l'arrivée du PnL Cash Flow
    carnet_ordres = pd.concat(all_trades).sort_values('exit_date').reset_index(drop=True)

    # 4. Exécuter le moteur financier (Money Management)
    print("\n📈 Lancement du moteur de Paper Trading...")
    balance_history = []
    balance = INITIAL_BALANCE
    
    for i, row in carnet_ordres.iterrows():
        # Prix réels d'entrée et de Stop Loss de l'Asset (dés-loggés)
        entry_price = np.exp(row['entry_p'])
        sl_price    = np.exp(row['sl'])
        
        # Quel % de l'Action le Stop Loss représente-t-il ?
        sl_distance_pct = abs((entry_price - sl_price) / entry_price)
        sl_distance_pct = max(0.005, sl_distance_pct) # Sécurité mathématique (min 0.5% d'écart)
        
        # Calcul Professionnel de Position : On place (X Dollars) dont la perte au SL = [3%] du Compte Courant
        capital_at_risk = balance * RISK_PER_TRADE
        position_size   = capital_at_risk / sl_distance_pct
        
        # Limite de garde-fou : Si SL très très serré, ça demanderait d'acheter 15x plus que ce qu'on possède.
        # On plafonne le trade à MAX_LEVERAGE * Compte
        if position_size > balance * MAX_LEVERAGE:
            position_size = balance * MAX_LEVERAGE
            
        # PnL réel en Dollars. "row['return']" contient déjà la purge finale des frais d'échange.
        net_return_pct = np.exp(row['return']) - 1
        pnl_usd        = position_size * net_return_pct
        
        # Mettre à jour le solde du compte
        balance += pnl_usd
        balance = max(1.0, balance) # Liquidation totale
        
        balance_history.append({
            'date': row['exit_date'],
            'pair': row['pair'],
            'pnl_usd': pnl_usd,
            'balance': balance,
            'pos_size': position_size,
            'roi_pct': net_return_pct * 100
        })

    # 5. Calcul des métriques & Graphiques
    history_df = pd.DataFrame(balance_history)
    history_df.set_index('date', inplace=True)
    
    final_balance = balance
    net_profit    = final_balance - INITIAL_BALANCE
    roi           = (net_profit / INITIAL_BALANCE) * 100
    
    # Calcul Drawdown Max
    peaks = history_df['balance'].cummax()
    drawdowns = (history_df['balance'] - peaks) / peaks * 100
    max_dd = drawdowns.min()
    
    wins = len(history_df[history_df['pnl_usd'] > 0])
    win_rate = (wins / len(history_df)) * 100

    print("="*70)
    print("  BILAN DU PORTEFEUILLE VIRTUEL (IA METAMODEL)")
    print("="*70)
    print(f"  Période           : {history_df.index[0].strftime('%Y-%m-%d')} → {history_df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Solde Initial     : ${INITIAL_BALANCE:,.2f}")
    print(f"  Solde Final       : ${final_balance:,.2f}")
    print(f"  Profit Net (Cash) : ${net_profit:,.2f}  ({roi:+.2f}%)")
    print(f"  Max Drawdown      : {max_dd:.2f}%")
    print(f"  Win Rate global   : {win_rate:.1f}%")
    print(f"  Total Trades      : {len(history_df)}")
    print("="*70)

    # Tracé visuel
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(history_df.index, history_df['balance'], color='#00ffcc', linewidth=2.5, label='Solde du Compte ($)')
    ax.axhline(INITIAL_BALANCE, color='gray', linestyle='--', linewidth=1.5, label='Capital Initial')
    
    # Remplissage couleur pour pertes (rouge) / gains (vert turquoise)
    ax.fill_between(history_df.index, history_df['balance'], INITIAL_BALANCE, 
                    where=(history_df['balance'] >= INITIAL_BALANCE), facecolor='#00ffcc', alpha=0.1)
    ax.fill_between(history_df.index, history_df['balance'], INITIAL_BALANCE, 
                    where=(history_df['balance'] < INITIAL_BALANCE), facecolor='#ff3333', alpha=0.1)
    
    ax.set_title(f"Simulation Crypto IA\nDépart: ${INITIAL_BALANCE:,.0f} | Risque Fixe: {RISK_PER_TRADE*100}% par Trade | Frais Deriv inclus", 
                 fontsize=14, pad=15)
    ax.set_ylabel("Portefeuille (USD)", fontsize=12)
    ax.grid(alpha=0.2)
    ax.legend()
    
    # Formatage de l'axe X pour dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    out_file = 'portfolio_simulation.png'
    plt.tight_layout()
    plt.savefig(out_file, dpi=120, bbox_inches='tight')
    print(f"✓ Capture du graphique sauvegardée 👉 {out_file}")


if __name__ == '__main__':
    run_portfolio_simulation()
