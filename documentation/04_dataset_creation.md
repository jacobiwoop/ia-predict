# 04 - Création du Dataset pour le Machine Learning

## 🎯 Objectif de ce Chapitre

Comprendre comment est créé le **dataset** qui servira à entraîner le modèle de Machine Learning dans le fichier `trendline_break_dataset.py`.

---

## 📊 Pourquoi un Dataset ?

### Le Problème

La stratégie de breakout simple prend **TOUS** les signaux :
- Beaucoup sont de **faux breakouts**
- Win rate d'environ 50%
- Profit factor faible (~1.02)

### La Solution

Entraîner un modèle de **Machine Learning** à reconnaître les **bons** des **mauvais** breakouts en utilisant :
- Des **features** (indicateurs, caractéristiques)
- Des **labels** (résultats : win ou loss)

---

## 🏗️ Architecture de la Fonction

### Signature

```python
def trendline_breakout_dataset(
    ohlcv: pd.DataFrame,      # Données OHLCV (Open, High, Low, Close, Volume)
    lookback: int,            # Période pour les trendlines (ex: 72)
    hold_period: int = 12,    # Durée maximum en position
    tp_mult: float = 3.0,     # Multiplicateur Take Profit (3 x ATR)
    sl_mult: float = 3.0,     # Multiplicateur Stop Loss (3 x ATR)
    atr_lookback: int = 168   # Période pour l'ATR (168h = 1 semaine)
):
    """
    Retourne :
    - trades  : DataFrame avec tous les trades et leurs features
    - data_x  : Features (indicateurs) pour le ML
    - data_y  : Labels (0 = loss, 1 = win) pour le ML
    """
```

---

## 📈 Étape 1 : Préparation des Données

### Log des Prix

```python
close = np.log(ohlcv['close'].to_numpy())
```

**Pourquoi le log ?**
- Rend les prix plus stables statistiquement
- Approxime les pourcentages de variation
- Évite les problèmes d'échelle (Bitcoin à 20k vs 60k)

```
    Prix normal         Log prix
      ▲                   ▲
      │                  │
  60K │    ╭────         │         ╭────
      │   ╱              │        ╱
  40K │  ╱               │       ╱
      │ ╱                │      ╱
  20K │╱                 │     ╱
      │                  │    ╱
      └──────────────────┴───────────────►
         (échelle          (échelle
          linéaire)         logarithmique)
```

---

### Calcul de l'ATR (Average True Range)

```python
atr = ta.atr(
    np.log(ohlcv['high']),
    np.log(ohlcv['low']),
    np.log(ohlcv['close']),
    atr_lookback
)
atr_arr = atr.to_numpy()
```

**Qu'est-ce que l'ATR ?**
- Mesure de la **volatilité** moyenne
- Plus l'ATR est élevé, plus le prix bouge
- Utilisé pour dimensionner Stop Loss et Take Profit

**Pourquoi `atr_lookback = 168` ?**
- 168 heures = **1 semaine**
- Donne une mesure de volatilité "récente" mais stable

```
    PRIX                    ATR
      ▲                      ▲
      │  ╭───╮              │
      │ ╱     ╲    ╭────    │    ╭──────
      │╱       ╲  ╱          │   ╱
      │         ╲╱           │  ╱
      │                      │ ╱
      │                      │╱
      └──────────────────────┴──────────►
         Prix volatile    →   ATR élevé
```

---

### Volume Normalisé

```python
vol_arr = (
    ohlcv['volume'] / ohlcv['volume'].rolling(atr_lookback).median()
).to_numpy()
```

**Pourquoi normaliser le volume ?**
- Le volume brut dépend de la période (2017 vs 2022)
- La normalisation permet de comparer dans le temps

**Interprétation :**
- `vol = 1` → Volume dans la moyenne
- `vol > 1` → Volume au-dessus de la moyenne (fort)
- `vol < 1` → Volume en-dessous de la moyenne (faible)

```
    Volume normalisé
      ▲
      │
  2.0 │         ╭───╮
      │        ╱     ╲
  1.5 │       ╱       ╲
      │      ╱         ╲
  1.0 │─────╱───────────╲──────────  ← Médiane (= 1)
      │    ╱             ╲
  0.5 │   ╱               ╲
      │  ╱                 ╲
      └──────────────────────────────► TEMPS
```

---

### ADX (Average Directional Index)

```python
adx = ta.adx(ohlcv['high'], ohlcv['low'], ohlcv['close'], lookback)
adx_arr = adx['ADX_' + str(lookback)].to_numpy()
```

**Qu'est-ce que l'ADX ?**
- Mesure la **force de la tendance** (pas la direction !)
- Valeurs typiques :
  - `ADX < 25` → Tendance faible (range)
  - `ADX > 25` → Tendance forte
  - `ADX > 50` → Tendance très forte

```
    ADX
      ▲
      │
  60  │                    ╭──────  ← Tendance TRÈS forte
      │                   ╱
  50  │                  ╱
      │                 ╱
  40  │                ╱
      │               ╱
  30  │              ╱
      │             ╱
  25  │────────────╱──────────────  ← Seuil de tendance forte
      │           ╱
  20  │          ╱
      │         ╱
  10  │        ╱
      │       ╱
   0  │──────╱────────────────────  ← Tendance faible (range)
      └─────────────────────────────► TEMPS
```

---

## 📝 Étape 2 : Détection et Enregistrement des Trades

### Variables de Suivi

```python
trades = pd.DataFrame()
trade_i = 0

in_trade = False
tp_price = None
sl_price = None
hp_i = None
```

| Variable | Type | Rôle |
|----------|------|------|
| `trades` | DataFrame | Stocke tous les trades avec leurs données |
| `trade_i` | int | Compteur de trades |
| `in_trade` | bool | True si on est actuellement en position |
| `tp_price` | float | Prix du Take Profit |
| `sl_price` | float | Prix du Stop Loss |
| `hp_i` | int | Index de sortie maximale (hold period) |

---

## 🔄 Étape 3 : Boucle Principale

### Structure de la Boucle

```python
for i in range(atr_lookback, len(ohlcv)):
    # Fenêtre de prix (SANS la bougie actuelle)
    window = close[i - lookback: i]

    # Calcul des trendlines
    s_coefs, r_coefs = fit_trendlines_single(window)

    # Projection de la résistance
    r_val = r_coefs[1] + lookback * r_coefs[0]

    # ... suite de la logique
```

**Pourquoi `range(atr_lookback, ...)` ?**
- On a besoin d'au moins `atr_lookback` bougies pour calculer l'ATR
- Les premières bougies ne peuvent pas être utilisées

---

## 🎯 Étape 4 : Détection d'Entrée (Entry)

### Condition d'Entrée

```python
if not in_trade and close[i] > r_val:
```

**Deux conditions :**
1. `not in_trade` → On n'est pas déjà en position
2. `close[i] > r_val` → Le prix franchit la résistance (breakout)

```
    PRIX
      ▲
      │              ╭────────────  ← Résistance
      │            ╱ │
      │          ╱   │  ← close[i] > r_val
      │        ╱     │   BREAKOUT !
      │      ╱       │
      │    ╱         │
      │  ╱           │
      └──┴───────────┴──────────────► TEMPS
         i-lookback  i
```

---

### Calcul des Niveaux de Sortie

```python
tp_price = close[i] + atr_arr[i] * tp_mult
sl_price = close[i] - atr_arr[i] * sl_mult
hp_i = i + hold_period
in_trade = True
```

**Take Profit (tp_price) :**
- `close[i] + ATR × 3`
- Objectif de gain à **3 fois la volatilité**

**Stop Loss (sl_price) :**
- `close[i] - ATR × 3`
- Perte maximum à **3 fois la volatilité**

**Hold Period (hp_i) :**
- `i + 12`
- Sortie automatique après **12 bougies** si TP/SL non touché

```
    PRIX
      ▲
      │                    ╭───────  ← TP = close + 3×ATR
      │                  ╱ │
      │                ╱   │
      │              ╱     │
      │            ╱       │
      │  ═══════════════════════════  ← Entrée (close[i])
      │            ╲       │
      │              ╲     │
      │                ╲   │
      │                  ╲ │
      │                    ╰───────  ← SL = close - 3×ATR
      │
      └──────────────────────────────► TEMPS
                     │←───→│ = 12 bougies (hold_period)
```

---

### Enregistrement des Données du Trade

```python
trades.loc[trade_i, 'entry_i'] = i       # Index d'entrée
trades.loc[trade_i, 'entry_p'] = close[i]  # Prix d'entrée
trades.loc[trade_i, 'atr'] = atr_arr[i]    # ATR au moment de l'entrée
trades.loc[trade_i, 'sl'] = sl_price       # Stop Loss
trades.loc[trade_i, 'tp'] = tp_price       # Take Profit
trades.loc[trade_i, 'hp_i'] = i + hold_period  # Hold period index

trades.loc[trade_i, 'slope'] = r_coefs[0]      # Pente de la résistance
trades.loc[trade_i, 'intercept'] = r_coefs[1]  # Intercept de la résistance
```

---

## 📊 Étape 5 : Calcul des Features (Indicateurs)

### Feature 1 : Resistance Slope (`resist_s`)

```python
trades.loc[trade_i, 'resist_s'] = r_coefs[0] / atr_arr[i]
```

**Calcul :**
- Pente de la résistance divisée par l'ATR
- **Normalisé par la volatilité**

**Intuition :**
- Pente positive → Tendance haussière → Breakout plus fiable
- Pente négative → Tendance baissière → Breakout moins fiable

```
    PRIX
      ▲
      │  ╭────────────  ← Résistance (pente positive)
      │ ╱
      │╱
      │
      │  ══════════════  ← Résistance (pente nulle)
      │
      │          ╱
      │        ╱
      │      ╱
      │    ╱
      │  ╱──────────────  ← Résistance (pente négative)
      └─────────────────────► TEMPS
```

---

### Feature 2 : Trendline Error (`tl_err`)

```python
# Valeurs de la ligne de résistance
line_vals = (r_coefs[1] + np.arange(lookback) * r_coefs[0])

# Erreur moyenne
err = np.sum(line_vals - window) / lookback
err /= atr_arr[i]  # Normalisation
trades.loc[trade_i, 'tl_err'] = err
```

**Calcul :**
- Somme des écarts entre la ligne et les prix
- Divisée par le nombre de points (moyenne)
- Normalisée par l'ATR

**Intuition :**
- `tl_err` faible → Prix proches de la résistance → Breakout plus fiable
- `tl_err` élevé → Prix loin de la résistance → Breakout moins fiable

```
    PRIX
      ▲
      │  ╭────────────  ← Résistance
      │ ╱│╲   │╲      │╲
      │╱ │ ╲  │ ╲     │ ╲
      │  │  ╲ │  ╲    │  ╲
      │  │   ╲│   ╲   │   ╲
      │  │    │    ╲  │    ╲
      │  │    │     ╲ │     ╲
      └──┴────┴──────┴┴──────┴───► TEMPS
         ← tl_err faible →   ← tl_err élevé →
```

---

### Feature 3 : Maximum Distance (`max_dist`)

```python
diff = line_vals - window
trades.loc[trade_i, 'max_dist'] = diff.max() / atr_arr[i]
```

**Calcul :**
- Maximum des écarts entre la ligne et les prix
- Normalisé par l'ATR

**Intuition :**
- `max_dist` faible → Tous les prix sont proches de la ligne → Bonne qualité
- `max_dist` élevé → Au moins un prix est très loin → Mauvaise qualité

```
    PRIX
      ▲
      │  ╭────────────  ← Résistance
      │ ╱│╲           │
      │╱ │ ╲          │
      │  │  ╲         │
      │  │   ╲        │
      │  │    ╲       │
      │  │     ╲      │ max_dist élevé
      │  │      ╲     │ (un prix très loin)
      │  │       ╲    │
      │  │        ╲   │
      └──┴──┴─────┴──┴─┴───────────► TEMPS
         ← max_dist faible →
```

---

### Feature 4 : Volume (`vol`)

```python
trades.loc[trade_i, 'vol'] = vol_arr[i]
```

**Intuition :**
- `vol > 1` → Volume fort → Breakout plus fiable
- `vol < 1` → Volume faible → Breakout moins fiable

**Pourquoi le volume est important ?**
- Un breakout avec **fort volume** indique une vraie conviction des acheteurs
- Un breakout avec **faible volume** peut être un faux signal

---

### Feature 5 : ADX (`adx`)

```python
trades.loc[trade_i, 'adx'] = adx_arr[i]
```

**Intuition :**
- `adx > 25` → Tendance forte → Breakout plus fiable
- `adx < 25` → Tendance faible (range) → Breakout moins fiable

---

## 🚪 Étape 6 : Gestion de la Sortie (Exit)

### Condition de Sortie

```python
if in_trade:
    if close[i] >= tp_price or close[i] <= sl_price or i >= hp_i:
        trades.loc[trade_i, 'exit_i'] = i       # Index de sortie
        trades.loc[trade_i, 'exit_p'] = close[i]  # Prix de sortie

        in_trade = False
        trade_i += 1
```

**Trois conditions de sortie (une seule suffit) :**

| Condition | Signification | Type |
|-----------|---------------|------|
| `close[i] >= tp_price` | Take Profit touché | ✅ Gain |
| `close[i] <= sl_price` | Stop Loss touché | ❌ Perte |
| `i >= hp_i` | Hold period écoulé | ⏱️ Temps |

---

## 🏷️ Étape 7 : Création des Labels

### Calcul du Return

```python
trades['return'] = trades['exit_p'] - trades['entry_p']
```

**Pourquoi cette formule ?**
- Les prix sont en **logarithmique**
- `exit_p - entry_p` ≈ **pourcentage de gain/perte**

```
    Si entry_p = 10.0 (log) et exit_p = 10.1 (log)
    → return = 10.1 - 10.0 = 0.1
    → ≈ 10% de gain
```

---

### Création du Label Binaire

```python
data_y = pd.Series(0, index=trades.index)
data_y.loc[trades['return'] > 0] = 1
```

**Label :**
- `1` → Trade **gagnant** (return > 0)
- `0` → Trade **perdant** (return ≤ 0)

```
    return > 0  →  Label = 1 (WIN)  ✅
    return ≤ 0  →  Label = 0 (LOSS) ❌
```

---

## 📦 Étape 8 : Extraction des Features

```python
data_x = trades[['resist_s', 'tl_err', 'vol', 'max_dist', 'adx']]
```

**Les 5 features finales :**

| Feature | Description | Type |
|---------|-------------|------|
| `resist_s` | Pente de la résistance / ATR | Numerique |
| `tl_err` | Erreur moyenne trendline / ATR | Numerique |
| `vol` | Volume normalisé | Numerique |
| `max_dist` | Distance maximale / ATR | Numerique |
| `adx` | ADX (force de tendance) | Numerique |

---

## 📊 Exemple de Dataset Final

### DataFrame `trades` (extrait)

| idx | entry_i | entry_p | atr | sl | tp | slope | resist_s | tl_err | vol | max_dist | adx | exit_i | exit_p | return |
|-----|---------|---------|-----|-----|-----|-------|----------|--------|-----|----------|-----|--------|--------|--------|
| 0 | 1250 | 9.45 | 0.02 | 9.39 | 9.51 | 0.001 | 0.05 | 0.01 | 1.2 | 0.03 | 28 | 1258 | 9.52 | 0.07 |
| 1 | 1340 | 9.52 | 0.025 | 9.445 | 9.595 | 0.002 | 0.08 | 0.015 | 0.8 | 0.04 | 22 | 1348 | 9.50 | -0.02 |
| 2 | 1456 | 9.60 | 0.018 | 9.546 | 9.654 | -0.001 | -0.055 | 0.02 | 1.5 | 0.05 | 35 | 1468 | 9.62 | 0.02 |

### DataFrame `data_x` (features)

| idx | resist_s | tl_err | vol | max_dist | adx |
|-----|----------|--------|-----|----------|-----|
| 0 | 0.05 | 0.01 | 1.2 | 0.03 | 28 |
| 1 | 0.08 | 0.015 | 0.8 | 0.04 | 22 |
| 2 | -0.055 | 0.02 | 1.5 | 0.05 | 35 |

### Series `data_y` (labels)

| idx | label |
|-----|-------|
| 0 | 1 |
| 1 | 0 |
| 2 | 1 |

---

## 📈 Statistiques du Dataset

### Code d'Analyse

```python
print("Profit Factor", returns[returns > 0].sum() / returns[returns < 0].abs().sum())
print("Win Rate", len(trades[trades['return'] > 0]) / len(trades))
print("Average Trade", trades['return'].mean())
```

### Résultats Typiques (Sans ML)

| Métrique | Valeur | Interprétation |
|----------|--------|----------------|
| Profit Factor | ~1.02 | Juste au-dessus de 1.0 |
| Win Rate | ~50% | Comme un pile ou face |
| Average Trade | ~0.05% | Très faible |

**Conclusion :** La stratégie de base n'est pas assez bonne seule. C'est là que le **Machine Learning** intervient pour filtrer les trades !

---

## 🎯 Résumé de la Fonction

```
┌─────────────────────────────────────────────────────────────┐
│  1. Préparer les données                                   │
│     → log(close), ATR, volume normalisé, ADX               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Boucle sur chaque bougie                               │
│     → Calcul des trendlines (lookback)                      │
│     → Détection des breakouts                               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Quand breakout détecté :                               │
│     → Enregistrer entry, tp, sl, hold_period                │
│     → Calculer les 5 features                               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Quand sortie (TP/SL/HP) :                              │
│     → Enregistrer exit                                      │
│     → Calculer le return                                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  5. Créer les labels                                       │
│     → return > 0 → Label = 1 (WIN)                          │
│     → return ≤ 0 → Label = 0 (LOSS)                         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  6. Retourner le dataset                                    │
│     → trades : Toutes les données des trades                │
│     → data_x : Features (5 colonnes)                        │
│     → data_y : Labels (0 ou 1)                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Points Clés à Retenir

1. **Stop Loss / Take Profit à 3 ATR** → Dimensionnés selon la volatilité
2. **Hold period de 12 bougies** → Sortie automatique si pas de résultat
3. **5 features principales** → resist_s, tl_err, vol, max_dist, adx
4. **Label binaire** → 1 si win, 0 si loss
5. **Normalisation par l'ATR** → Features indépendantes de la volatilité

---

*Document suivant : [05 - Explication des Features](./05_features_explication.md)*
