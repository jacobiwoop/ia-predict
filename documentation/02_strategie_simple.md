# 02 - Stratégie Simple (Sans Machine Learning)

## 🎯 Objectif de ce Chapitre

Comprendre comment fonctionne la stratégie de **Trendline Breakout** dans sa version la plus simple, **sans filtrage par Machine Learning**.

---

## 📝 Le Concept de Base

### Idée Principale

Quand le prix **franchit** (breakout) une ligne de tendance :
- **Au-dessus de la résistance** → Signal d'ACHAT (LONG)
- **En-dessous du support** → Signal de VENTE (SHORT)

```
    PRIX
      ▲
      │                              ╭─────────── ════════ Résistance
      │                            ╱ │
      │                          ╱   │
      │                        ╱     │
      │                      ╱       │  ← BREAKOUT !
      │                    ╱         │     Prix > Résistance
      │                  ╱           │     → Signal LONG (+1)
      │                ╱             │
      │              ╱               │
      │            ╱                 │
      │═══════════╱═══════════════════════════ Support
      │          ╱
      │        ╱
      │
      └─────────────────────────────────────────► TEMPS
```

---

## 💻 Le Code : `trendline_breakout.py`

### Signature de la Fonction

```python
def trendline_breakout(close: np.array, lookback: int):
    """
    close    : Tableau des prix de clôture
    lookback : Nombre de bougies pour calculer les trendlines (ex: 72)

    Retourne :
    - s_tl   : Ligne de support (support trendline)
    - r_tl   : Ligne de résistance (resistance trendline)
    - sig    : Signal de trading (+1, -1, ou 0)
    """
```

### Initialisation des Tableaux

```python
s_tl = np.zeros(len(close))   # Support trendline
s_tl[:] = np.nan              # Rempli de NaN (pour affichage)

r_tl = np.zeros(len(close))   # Resistance trendline
r_tl[:] = np.nan

sig = np.zeros(len(close))    # Signal de trading
```

**Pourquoi NaN ?**
- Les `NaN` (Not a Number) permettent de ne pas afficher les premières valeurs
- Les trendlines ne peuvent être calculées qu'après `lookback` bougies

---

## 🔄 La Boucle Principale

### Structure de la Boucle

```python
for i in range(lookback, len(close)):
    # NOTE window does NOT include the current candle
    window = close[i - lookback: i]

    s_coefs, r_coefs = fit_trendlines_single(window)

    # Find current value of line, projected forward to current bar
    s_val = s_coefs[1] + lookback * s_coefs[0]
    r_val = r_coefs[1] + lookback * r_coefs[0]

    s_tl[i] = s_val
    r_tl[i] = r_val

    if close[i] > r_val:
        sig[i] = 1.0
    elif close[i] < s_val:
        sig[i] = -1.0
    else:
        sig[i] = sig[i - 1]
```

### Explication Étape par Étape

#### Étape 1 : Récupérer la Fenêtre de Prix

```python
window = close[i - lookback: i]
```

**IMPORTANT** : La fenêtre **N'INCLUT PAS** la bougie actuelle !

```
    Index :     i-72              i-1      i
                │──────────────────│       │
                │     WINDOW       │       │  ← Bougie actuelle
                │   (72 bougies)   │       │     (exclue)
                │──────────────────│       │
                                    ↑
                              On calcule les
                              trendlines ici
```

**Pourquoi exclure la bougie actuelle ?**
- Pour permettre le **breakout** !
- Si on incluait la bougie actuelle, le prix ne pourrait jamais être au-dessus
- C'est un **décalage volontaire** (lag) pour détecter les franchissements

---

#### Étape 2 : Calculer les Trendlines

```python
s_coefs, r_coefs = fit_trendlines_single(window)
```

Cette fonction (définie dans `trendline_automation.py`) retourne :
- `s_coefs` : (pente, intercepte) pour la trendline de **support**
- `r_coefs` : (pente, intercepte) pour la trendline de **résistance**

```
    PRIX
      ▲
      │                    ╭─────────────── r_coefs (résistance)
      │                  ╱
      │                ╱
      │              ╱
      │    Prix  ╱
      │        ╱
      │      ╱
      │    ╱
      │  ╱─────────────────── s_coefs (support)
      │
      └─────────────────────────────────────►
```

---

#### Étape 3 : Projeter les Valeurs vers la Bougie Actuelle

```python
s_val = s_coefs[1] + lookback * s_coefs[0]
r_val = r_coefs[1] + lookback * r_coefs[0]
```

**Formule de la droite** : `y = slope * x + intercept`

Où :
- `slope` (pente) = `coefs[0]`
- `intercept` (origine) = `coefs[1]`
- `x` = position = `lookback` (car on projette d'une bougie en avant)

```
    PRIX
      ▲
      │
      │                    ╭─────────────── r_coefs
      │                  ╱│
      │                ╱  │
      │              ╱    │
      │            ╱      │  ← Projection
      │          ╱        │    r_val = intercept + lookback * slope
      │        ╱          │
      │      ╱            │
      │    ╱              │
      │  ╱────────────────│──────────────── s_coefs
      │                   │
      │              ←──→ │
      │              1    │
      │            bougie │
      │                   ▼
      │              close[i] (bougie actuelle)
      └─────────────────────────────────────► TEMPS
```

---

#### Étape 4 : Générer le Signal

```python
if close[i] > r_val:
    sig[i] = 1.0          # BREAKOUT HAUSSE → LONG
elif close[i] < s_val:
    sig[i] = -1.0         # BREAKOUT BAISSE → SHORT
else:
    sig[i] = sig[i - 1]   # PAS DE BREAKOUT → On garde le signal précédent
```

### Logique des Signaux

| Condition | Signal | Signification |
|-----------|--------|---------------|
| `close[i] > r_val` | **+1.0** | Le prix est au-dessus de la résistance → **LONG** |
| `close[i] < s_val` | **-1.0** | Le prix est en-dessous du support → **SHORT** |
| Sinon | `sig[i-1]` | Pas de breakout → On **conserve** la position précédente |

**Important** : Le signal est **persistant**
- Une fois qu'on a un signal +1 ou -1, on le garde tant qu'il n'y a pas de breakout inverse
- Cela signifie qu'on reste en position jusqu'à nouvel ordre

---

## 📊 Visualisation

### Graphique des Trendlines

```python
plt.style.use('dark_background')
data['close'].plot(label='Close')
data['resist'].plot(label='Resistance', color='green')
data['support'].plot(label='Support', color='red')
plt.show()
```

```
    PRIX (BTC/USDT)
      ▲
      │
      │     ╭───╮       ╭─────╮               ╭────────
  50K │    ╱     ╲     ╱       ╲             ╱    ╭────
      │   ╱       ╲   ╱         ╲           ╱    ╱
      │  ╱         ╲ ╱           ╲         ╱    ╱
      │ ╱           ╲             ╲       ╱    ╱
  40K │╱             ╲             ╲     ╱    ╱
      │               ╲             ╲   ╱    ╱
      │                ╲             ╲ ╱    ╱
      │                 ╲             ╲    ╱
  30K │                  ╲             ╲  ╱
      │                   ╲             ╲╱
      │                    ╲             ╲
      │───────────────────────────────────────────────────
      │  ════════════════════════════════════════════════ Support (rouge)
      │
      └───────────────────────────────────────────────────► TEMPS
```

---

## 📈 Calcul de la Performance

### Rendement Logarithmique

```python
data['r'] = np.log(data['close']).diff().shift(-1)
```

**Pourquoi log return ?**
- Approximation du pourcentage de gain/perte
- Additif dans le temps (plus facile à manipuler)
- `log(P1/P0) ≈ (P1 - P0) / P0` pour les petites variations

**Pourquoi `shift(-1)` ?**
- On prend le rendement de la **bougie suivante**
- Pour évaluer la performance de notre signal

---

### Rendement de la Stratégie

```python
strat_r = data['signal'] * data['r']
```

| Signal | Rendement | Explication |
|--------|-----------|-------------|
| +1 | `+1 × r` | En position LONG → On gagne si le prix monte |
| -1 | `-1 × r` | En position SHORT → On gagne si le prix descend |
| 0 | `0 × r = 0` | Pas en position → Pas de gain ni perte |

---

### Profit Factor

```python
pf = strat_r[strat_r > 0].sum() / strat_r[strat_r < 0].abs().sum()
print("Profit Factor", lookback, pf)
```

**Définition du Profit Factor :**

$$\text{Profit Factor} = \frac{\text{Somme des gains}}{\text{Somme des pertes (en valeur absolue)}}$$

| Profit Factor | Interprétation |
|---------------|----------------|
| > 1.5 | Excellente performance |
| 1.2 - 1.5 | Bonne performance |
| 1.0 - 1.2 | Performance limite |
| < 1.0 | Stratégie perdante |

**Résultat typique pour cette stratégie :**
- Profit Factor ≈ **1.02 - 1.035**
- C'est "OK" mais pas excellent
- Sans frais de trading, c'est légèrement profitable
- **Avec** frais de trading, ce serait probablement perdant

---

### Courbe de Performance Cumulée

```python
strat_r.cumsum().plot()
plt.ylabel("Cumulative Log Return")
plt.show()
```

```
    CUMULATIVE LOG RETURN
      ▲
      │
      │                    ╭──────╮
      │                  ╱        ╲
      │                ╱            ╲      ╭────────
      │              ╱                ╲    ╱
      │            ╱                    ╲╱
      │          ╱
      │        ╱
      │      ╱
      │    ╱
      │  ╱
      │╱
      └────────────────────────────────────────────────► TEMPS
```

---

## 🔍 Test sur Différents Lookbacks

### Code de Test

```python
lookbacks = list(range(24, 169, 2))
pfs = []

lookback_returns = pd.DataFrame()
for lookback in lookbacks:
    support, resist, signal = trendline_breakout(data['close'].to_numpy(), lookback)
    data['signal'] = signal

    data['r'] = np.log(data['close']).diff().shift(-1)
    strat_r = data['signal'] * data['r']

    pf = strat_r[strat_r > 0].sum() / strat_r[strat_r < 0].abs().sum()
    print("Profit Factor", lookback, pf)
    pfs.append(pf)

    lookback_returns[lookback] = strat_r
```

### Résultat Typique

```
    PROFIT FACTOR
      ▲
      │
  1.5 │
      │
  1.3 │        ╭───╮
      │       ╱     ╲
  1.1 │──────╱       ╲────────────────────────
      │     ╱         ╲
  1.0 │────╱───────────╲──────────────────────
      │   ╱             ╲
  0.8 │╱                 ╲
      └────────────────────────────────────────►
       24   50   72   100  120  150  LOOKBACK
```

**Observations :**
- Pic de performance entre 32 et 42 → Probablement de la **chance** (overfitting)
- Performance "OK" (~1.0-1.1) sur la plupart des valeurs
- Lookback de 72 est un bon compromis

---

## ⚠️ Limites de la Stratégie Simple

### Problèmes Identifiés

1. **Trop de trades** : La stratégie est en position ~100% du temps
2. **Win rate faible** : Environ 50% (pile ou face)
3. **Profit factor limite** : ~1.02, trop proche de 1.0
4. **Faux breakouts** : Beaucoup de signaux qui se retournent contre nous

### Solution : Le Meta-Labeling

Pour améliorer cette stratégie, on va ajouter une couche de **Machine Learning** :
- Analyser chaque breakout avec 5 indicateurs (features)
- Prédire la probabilité de succès
- Ne prendre que les trades avec probabilité > 50%

*C'est ce qu'on verra dans les prochains chapitres !*

---

## 📝 Résumé du Code Complet

```python
def trendline_breakout(close: np.array, lookback: int):
    # 1. Initialisation
    s_tl = np.zeros(len(close))
    s_tl[:] = np.nan
    r_tl = np.zeros(len(close))
    r_tl[:] = np.nan
    sig = np.zeros(len(close))

    # 2. Boucle sur chaque bougie
    for i in range(lookback, len(close)):
        # Fenêtre de prix (SANS la bougie actuelle !)
        window = close[i - lookback: i]

        # Calcul des trendlines
        s_coefs, r_coefs = fit_trendlines_single(window)

        # Projection vers la bougie actuelle
        s_val = s_coefs[1] + lookback * s_coefs[0]
        r_val = r_coefs[1] + lookback * r_coefs[0]

        # Sauvegarde pour affichage
        s_tl[i] = s_val
        r_tl[i] = r_val

        # Génération du signal
        if close[i] > r_val:
            sig[i] = 1.0       # LONG
        elif close[i] < s_val:
            sig[i] = -1.0      # SHORT
        else:
            sig[i] = sig[i - 1]  # Conserver

    return s_tl, r_tl, sig
```

---

## 🎯 Points Clés à Retenir

1. **Fenêtre décalée** : Les trendlines sont calculées sur les `lookback` bougies **précédentes**, pas la bougie actuelle
2. **Projection** : Les valeurs des trendlines sont projetées d'une bougie en avant
3. **Signal persistant** : Une fois un signal donné, on le garde jusqu'au breakout inverse
4. **Profit factor faible** : ~1.02, nécessite un filtrage pour être viable
5. **100% du temps en marché** : La stratégie simple est toujours en position

---

*Document suivant : [03 - Calcul des Trendlines](./03_trendline_calculation.md)*
