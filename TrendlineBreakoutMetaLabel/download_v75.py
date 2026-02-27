"""
download_v75.py
---------------
Télécharge les données OHLCV 1H du Volatility 75 Index
depuis l'API WebSocket publique de Deriv.
Sauvegarde au même format que BTCUSDT3600.csv
"""

import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from datetime import datetime
import time


async def fetch_candles(symbol: str, granularity: int, end_ts: int) -> list:
    """
    Récupère 5000 bougies OHLCV via WebSocket Deriv finissant à `end_ts`.
    """
    url = "wss://ws.binaryws.com/websockets/v3?app_id=1089"
    candles = []
    retries = 0
    max_retries = 5

    while retries < max_retries:
        try:
            # Timeout généreux
            async with websockets.connect(url, open_timeout=30, close_timeout=30) as ws:
                request = {
                    "ticks_history": symbol,
                    "adjust_start_time": 1,
                    "count": 5000,
                    "end": str(end_ts),  # L'API Deriv exige un string
                    "granularity": granularity,
                    "style": "candles"
                }
                await ws.send(json.dumps(request))
                response = json.loads(await ws.recv())

                if 'error' in response:
                    print(f"  ❌ Erreur Deriv : {response['error']['message']}")
                    return []

                if 'candles' in response:
                    candles = response['candles']
                    print(f"  ✓ {len(candles)} bougies reçues")
                    return candles

        except Exception as e:
            retries += 1
            print(f"  ⚠️  Perte de connexion ({e}) — tentative {retries}/{max_retries} dans 5s...")
            await asyncio.sleep(5)
            
    print("  ❌ Échec définitif après plusieurs tentatives.")
    return []


async def download_v75(symbol: str = "R_75",
                        start_date: str = "2020-01-01",
                        end_date: str = "2024-01-01",
                        granularity: int = 3600) -> pd.DataFrame:
    """
    Télécharge tout l'historique par batch de 5000 bougies.
    """
    start_ts = int(pd.Timestamp(start_date).timestamp())
    end_ts   = int(pd.Timestamp(end_date).timestamp())

    all_candles = []
    current_end = end_ts
    batch = 0
    seen_timestamps = set()

    print(f"\nTéléchargement {symbol} ({start_date} → {end_date})...")

    while current_end > start_ts:
        print(f"  Batch {batch + 1} — jusqu'au {datetime.fromtimestamp(current_end).strftime('%Y-%m-%d')}")

        candles = await fetch_candles(symbol, granularity, current_end)

        if not candles:
            break
            
        # Filtre anti-boucles infinies (Deriv renvoie parfois les mêmes bougies)
        new_candles = [c for c in candles if c['epoch'] not in seen_timestamps]
        
        if not new_candles:
             print("  ⚠️ Deriv boucle sur les mêmes dates. Arrêt prématuré.")
             break
             
        for c in new_candles:
            seen_timestamps.add(c['epoch'])

        all_candles = new_candles + all_candles

        # Remonter dans le temps : on prend l'epoch le plus ancien DU BATCH
        oldest_ts = new_candles[0]['epoch']
        if oldest_ts <= start_ts:
            break

        # On force le bond dans le temps pour éviter que request.end == oldest_ts ne renvoie les mêmes valeurs
        current_end = oldest_ts - granularity
        batch += 1
        await asyncio.sleep(0.5)

    if not all_candles:
        print("  ❌ Aucune donnée reçue")
        return pd.DataFrame()

    # Construire le DataFrame
    df = pd.DataFrame(all_candles)
    df['date']   = pd.to_datetime(df['epoch'], unit='s')
    df = df.set_index('date')
    df = df.rename(columns={
        'open':  'open',
        'high':  'high',
        'low':   'low',
        'close': 'close'
    })
    df = df[['open', 'high', 'low', 'close']]
    df = df.astype(float)

    # V75 n'a pas de volume réel — on met 1.0 (requis par notre code)
    df['volume'] = 1.0

    # Supprimer les doublons
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()

    # Filtrer la plage demandée
    df = df[df.index >= start_date]
    df = df[df.index <= end_date]

    if df.empty:
        print("  ❌ DataFrame vide après le filtrage.")
        return df

    print(f"  ✓ Total : {len(df)} bougies")
    print(f"  📅 Période : {df.index[0]} → {df.index[-1]}")

    return df


if __name__ == '__main__':
    # Symboles Deriv disponibles :
    # R_10  = Volatility 10
    # R_25  = Volatility 25
    # R_50  = Volatility 50
    # R_75  = Volatility 75  ← notre cible
    # R_100 = Volatility 100

    async def main():
        df = await download_v75(
            symbol     = "R_75",
            start_date = "2025-08-04",
            end_date   = "2026-02-27",
            granularity = 3600  # 1h
        )

        if df.empty:
            print("\n❌ Téléchargement échoué")
            return

        # Sauvegarder au format compatible avec notre pipeline
        output = "data/V75USDT3600.csv"
        df.to_csv(output)
        print(f"\n💾 Sauvegardé : {output}")

        # Statistiques de base
        print("\n── Statistiques V75 ──────────────────")
        print(f"  Bougies       : {len(df)}")
        print(f"  Close moyen   : {df['close'].mean():.4f}")
        print(f"  Volatilité    : {df['close'].pct_change().std() * 100:.2f}%/bougie")
        print(f"  Max drawdown  : ", end='')
        roll_max = df['close'].cummax()
        dd = ((df['close'] - roll_max) / roll_max * 100)
        print(f"{dd.min():.1f}%")

    asyncio.run(main())