"""
download_data.py
----------------
Télécharge les données OHLCV 1H depuis Binance (API publique, sans compte)
Paires : ETHUSDT, SOLUSDT (et optionnellement d'autres)

Usage :
    python3 download_data.py
"""

import requests
import pandas as pd
import time
import os


def download_binance_ohlcv(symbol: str, interval: str = '1h',
                            start_date: str = '2019-01-01',
                            end_date: str = '2024-01-01') -> pd.DataFrame:
    """
    Télécharge les données OHLCV depuis Binance API publique.
    Gère automatiquement la pagination (max 1000 bougies par requête).
    """
    url       = 'https://api.binance.com/api/v3/klines'
    start_ms  = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ms    = int(pd.Timestamp(end_date).timestamp() * 1000)

    all_data  = []
    current   = start_ms
    batch     = 0

    print(f"  Téléchargement {symbol} {interval} ({start_date} → {end_date})")

    while current < end_ms:
        params = {
            'symbol'   : symbol,
            'interval' : interval,
            'startTime': current,
            'endTime'  : end_ms,
            'limit'    : 1000
        }

        try:
            # Augmenter le timeout de 10 à 30 secondes
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  ⚠️  Erreur requête : {e} — nouvelle tentative dans 10s")
            time.sleep(10)
            continue

        if not data:
            break

        all_data.extend(data)
        current = data[-1][0] + 1  # prochaine bougie après la dernière reçue
        batch  += 1

        if batch % 10 == 0:
            print(f"    {len(all_data)} bougies téléchargées...")

        # Augmenter la pause pour ne pas se faire bannir par Binance
        time.sleep(0.3)

    if not all_data:
        print(f"  ❌ Aucune donnée reçue pour {symbol}")
        return pd.DataFrame()

    # Colonnes Binance : open_time, open, high, low, close, volume, ...
    df = pd.DataFrame(all_data, columns=[
        'date', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'n_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])

    # Nettoyage
    df['date']  = pd.to_datetime(df['date'], unit='ms')
    df          = df.set_index('date')
    df          = df[['open', 'high', 'low', 'close', 'volume']]
    df          = df.astype(float)
    df          = df.dropna()

    print(f"  ✓ {len(df)} bougies téléchargées pour {symbol}")
    return df


if __name__ == '__main__':

    # ── Paires à télécharger ─────────────────────────────────────────────────
    # ETH a déjà été téléchargé avec succès on garde que SOL
    pairs = [
        ('SOLUSDT',  '2020-09-01', '2024-01-01'),  # SOL lancé en 2020
    ]

    os.makedirs('data', exist_ok=True)

    for symbol, start, end in pairs:
        print(f"\n{'='*50}")
        df = download_binance_ohlcv(symbol, '1h', start, end)

        if df.empty:
            continue

        # Sauvegarder au même format que BTCUSDT3600.csv
        filename = f"data/{symbol}3600.csv"
        df.to_csv(filename)
        print(f"  💾 Sauvegardé : {filename}")
        print(f"  📅 Période    : {df.index[0]} → {df.index[-1]}")
        print(f"  📊 Bougies    : {len(df)}")

        # Pause entre les paires
        time.sleep(1)

    print(f"\n{'='*50}")
    print("✓ Téléchargement terminé !")
    print("\nFichiers disponibles dans le dossier data/ :")
    for f in os.listdir('data'):
        path = f"data/{f}"
        size = os.path.getsize(path) / 1024
        print(f"  {f:<25} {size:.0f} KB")

    print("\nProchaine étape :")
    print("  → Lancer : python3 walkforward_multi.py")