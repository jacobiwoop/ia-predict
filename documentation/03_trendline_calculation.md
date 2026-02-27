# 03 - Calcul des Lignes de Tendance (Trendline Automation)

## 🎯 Objectif de ce Chapitre

Comprendre **comment sont calculées mathématiquement** les lignes de tendance (support et résistance) dans le fichier `trendline_automation.py`.

---

## 📐 Le Défi du Calcul des Trendlines

### Problème à Résoudre

On veut tracer une ligne de tendance qui :
1. **Touche tous les prix** (ou presque)
2. **Passe soit au-dessus** (résistance) **soit en-dessous** (support) de **TOUS** les prix
3. **Minimise la distance** avec les prix

```
    PRIX
      ▲
      │   ╭─╮     ╭───╮
      │  ╱   ╲   ╱     ╲    ← Prix réels
      │ ╱     ╲ ╱       ╲
      │╱       ╲╱         ╲
      │                     ╲
      │  ════════════════════════  ← Trendline de support
      │    (doit être EN-DESSOUS de TOUS les prix)
      │
      └─────────────────────────────────► TEMPS
```

### Contrainte Importante

- **Pour le support** : La ligne doit être **strictement en-dessous** de tous les prix
- **Pour la résistance** : La ligne doit être **strictement au-dessus** de tous les prix

```
    PRIX
      ▲
      │  ╭───────────────────────  ← Résistance (au-dessus de tout)
      │ ╱ ╲   ╭─╮     ╭───╮
      │╱   ╲ ╱   ╲   ╱     ╲
      │     ╲     ╲ ╱       ╲
      │      ╲     ╲         ╲
      │       ╰─────╰─────────╰
      │
      │  ═══════════════════════  ← Support (en-dessous de tout)
      │
      └─────────────────────────────────► TEMPS
```

---

## 🔧 Fonction Principale : `fit_trendlines_single`

### Signature et Objectif

```python
def fit_trendlines_single(data: np.array):
    """
    data : Tableau des prix (fenêtre de lookback bougies)

    Retourne :
    - support_coefs : (slope, intercept) pour le support
    - resist_coefs  : (slope, intercept) pour la résistance
    """
```

---

## 📊 Étape 1 : Ligne de Best Fit (Régression Linéaire)

### Code

```python
x = np.arange(len(data))
coefs = np.polyfit(x, data, 1)
```

### Explication

**`np.polyfit(x, data, 1)`** :
- Ajuste un **polynôme de degré 1** (une droite) aux données
- Utilise la méthode des **moindres carrés** (minimise la somme des erreurs²)
- Retourne `(slope, intercept)` = `(pente, ordonnée à l'origine)`

```
    PRIX
      ▲
      │   ╭─╮     ╭───╮
      │  ╱   ╲   ╱     ╲
      │ ╱     ╲ ╱    ╭──╫────  ← Ligne de best fit
      │╱       ╲╱   ╱  ║
      │         ╲  ╱   ║
      │          ╲╱    ║
      │                ║
      └────────────────┴────────► TEMPS
                         x = len(data)
```

### Pourquoi une Ligne de Best Fit ?

La ligne de best fit sert de **point de départ** pour trouver :
- Le **pivot supérieur** (point le plus au-dessus de la ligne)
- Le **pivot inférieur** (point le plus en-dessous de la ligne)

---

## 🎯 Étape 2 : Trouver les Points Pivots

### Code

```python
line_points = coefs[0] * x + coefs[1]

upper_pivot = (data - line_points).argmax()
lower_pivot = (data - line_points).argmin()
```

### Explication

**`line_points`** :
- Calcule les valeurs de la ligne de best fit pour chaque point

**`data - line_points`** :
- Calcule l'écart entre chaque prix et la ligne de best fit

```
    PRIX
      ▲
      │         │
      │   ╭─╮   │   ← upper_pivot (plus grand écart positif)
      │  ╱   ╲ ╱│╲
      │ ╱     ╲ ║ ╲
      │╱       ╲║  ╲
      │─────────╫───╲──────  ← Ligne de best fit
      │         ║    ╲
      │         ║     ╲
      │         ║      ╲
      │         ║       ╰  ← lower_pivot (plus grand écart négatif)
      │
      └─────────┴───────────► TEMPS
```

**`argmax()`** :
- Retourne l'**indice** du point le plus **au-dessus** de la ligne
- Ce sera le point de pivot pour la **résistance**

**`argmin()`** :
- Retourne l'**indice** du point le plus **en-dessous** de la ligne
- Ce sera le point de pivot pour le **support**

---

## 🔍 Étape 3 : Optimisation de la Pente

### Code

```python
support_coefs = optimize_slope(True, lower_pivot, coefs[0], data)
resist_coefs = optimize_slope(False, upper_pivot, coefs[0], data)
```

### Paramètres de `optimize_slope`

| Paramètre | Valeur | Signification |
|-----------|--------|---------------|
| `support` | `True`/`False` | Type de trendline (support ou résistance) |
| `pivot` | `lower_pivot`/`upper_pivot` | Indice du point pivot |
| `init_slope` | `coefs[0]` | Pente initiale (de la ligne de best fit) |
| `data` | `data` | Tableau des prix |

### Objectif de l'Optimisation

Trouver la pente qui :
1. **Passe par le point pivot**
2. **Reste en-dessous (support) ou au-dessus (résistance) de TOUS les prix**
3. **Minimise la somme des erreurs au carré**

---

## 🧮 Fonction `optimize_slope` : L'Algorithme Complet

### Signature

```python
def optimize_slope(support: bool, pivot: int, init_slope: float, y: np.array):
    """
    support   : True pour support, False pour résistance
    pivot     : Indice du point pivot
    init_slope: Pente initiale (ligne de best fit)
    y         : Tableau des prix

    Retourne : (slope, intercept) de la trendline optimale
    """
```

---

## 📏 Étape 1 : Calculer l'Unité de Pente

### Code

```python
slope_unit = (y.max() - y.min()) / len(y)
```

### Explication

**`slope_unit`** :
- C'est la "résolution" de base pour ajuster la pente
- Proportionnel à la plage de prix divisée par le nombre de points

```
    PRIX
      ▲
      │
      │  ╭───╮
      │ ╱     ╲
      │╱       ╲
      │         ╲
      │          ╲
      │           ╰
      │
      │◄─────────► = y.max() - y.min() (plage de prix)
      │
      └─────────────────────────────► TEMPS
                     │←──────→│ = len(y)
```

---

## 🎚️ Étape 2 : Initialisation des Variables d'Optimisation

### Code

```python
opt_step = 1.0        # Pas d'optimisation initial
min_step = 0.0001     # Pas minimum (précision)
curr_step = opt_step  # Pas courant

best_slope = init_slope
best_err = check_trend_line(support, pivot, init_slope, y)
```

### Explication

**Approche de type "grid search adaptatif"** :
- On commence avec un grand pas (`opt_step = 1.0`)
- On réduit le pas de moitié à chaque échec (`curr_step *= 0.5`)
- On s'arrête quand le pas est trop petit (`min_step = 0.0001`)

```
    Itération 1 : curr_step = 1.0        (grand pas)
    Itération 2 : curr_step = 0.5        (pas moyen)
    Itération 3 : curr_step = 0.25       (pas plus fin)
    ...
    Itération N : curr_step < 0.0001     (STOP)
```

---

## 📐 Étape 3 : Fonction de Vérification `check_trend_line`

### Signature

```python
def check_trend_line(support: bool, pivot: int, slope: float, y: np.array):
    """
    Vérifie si une ligne avec cette slope est valide.

    Retourne :
    - L'erreur au carré si la ligne est valide
    - -1.0 si la ligne est INVALIDE
    """
```

### Code et Explication

```python
# 1. Calculer l'intercepte pour que la ligne passe par le pivot
intercept = -slope * pivot + y[pivot]
line_vals = slope * np.arange(len(y)) + intercept
```

**Formule** : Pour que la ligne passe par le point `(pivot, y[pivot])` :
```
y[pivot] = slope * pivot + intercept
→ intercept = -slope * pivot + y[pivot]
```

```
    PRIX
      ▲
      │
      │              ╭ pivot (pivot, y[pivot])
      │             ╱│
      │            ╱ │
      │           ╱  │
      │          ╱   │
      │         ╱    │
      │        ╱     │
      │       ╱      │
      │      ╱       │
      └─────┴───────┴────────► TEMPS
            intercept
```

---

```python
# 2. Calculer les différences entre ligne et prix
diffs = line_vals - y
```

**`diffs`** : Distance entre la ligne et chaque prix

```
    PRIX
      ▲
      │   ╭─╮  ← prix[0]
      │  ╱│   ╲
      │ ╱ │    ╲  ← prix[1]
      │╱  │     ╲
      │───┼──────╲────  ← line_vals
      │   │diffs[0]
      │   │
      └───┴─────────────► TEMPS
```

---

```python
# 3. Vérifier la contrainte de validité
if support and diffs.max() > 1e-5:
    return -1.0  # INVALIDE : un prix est en-dessous de la ligne
elif not support and diffs.min() < -1e-5:
    return -1.0  # INVALIDE : un prix est au-dessus de la ligne
```

**Pour le support (`support = True`)** :
- `line_vals - y` doit être **négatif ou nul** partout
- La ligne doit être **en-dessous** de tous les prix
- Si `diffs.max() > 0` → un prix est en-dessous → **INVALIDE**

**Pour la résistance (`support = False`)** :
- `line_vals - y` doit être **positif ou nul** partout
- La ligne doit être **au-dessus** de tous les prix
- Si `diffs.min() < 0` → un prix est au-dessus → **INVALIDE**

```
    PRIX
      ▲
      │  ╭─╮     ╭───╮
      │ ╱   ╲   ╱     ╲
      │╱     ╲ ╱       ╲
      │       ╲         ╲
      │        ╲         ╲
      │         ╲         ╲
      │══════════════════════  ← Support valide
      │
      └─────────────────────────► TEMPS

    PRIX
      ▲
      │══════════════════════  ← Résistance valide
      │         ╱         ╱
      │        ╱         ╱
      │       ╱         ╱
      │      ╱         ╱
      │     ╱         ╱
      │    ╱         ╱
      └─────────────────────────► TEMPS
```

---

```python
# 4. Calculer l'erreur (somme des carrés des différences)
err = (diffs ** 2.0).sum()
return err
```

**Objectif** : Minimiser cette erreur tout en respectant la contrainte.

---

## 🔄 Étape 4 : Boucle d'Optimisation

### Structure de la Boucle

```python
get_derivative = True
derivative = None
while curr_step > min_step:
    # ... logique d'optimisation ...
```

### Phase 1 : Calcul de la Dérivée

```python
if get_derivative:
    # Augmenter légèrement la pente
    slope_change = best_slope + slope_unit * min_step
    test_err = check_trend_line(support, pivot, slope_change, y)
    derivative = test_err - best_err

    # Si ça échoue, essayer de diminuer
    if test_err < 0.0:
        slope_change = best_slope - slope_unit * min_step
        test_err = check_trend_line(support, pivot, slope_change, y)
        derivative = best_err - test_err

    if test_err < 0.0:
        raise Exception("Derivative failed. Check your data.")

    get_derivative = False
```

**But** : Déterminer dans quelle direction aller pour réduire l'erreur.

```
    Erreur
      ▲
      │
      │        ╲
      │         ╲
      │          ╲  ← Pente actuelle
      │           ╲│
      │            ╲
      │             ╲
      │              ╲
      └───────────────┴──────────► Pente
                ←  →
              derivative
              (direction à suivre)
```

**Si `derivative > 0`** : Augmenter la pente augmente l'erreur → **Diminuer la pente**
**Si `derivative < 0`** : Augmenter la pente diminue l'erreur → **Augmenter la pente**

---

### Phase 2 : Tester une Nouvelle Pente

```python
if derivative > 0.0:
    test_slope = best_slope - slope_unit * curr_step
else:
    test_slope = best_slope + slope_unit * curr_step

test_err = check_trend_line(support, pivot, test_slope, y)
```

**On teste dans la direction opposée à la dérivée** (pour descendre vers le minimum).

---

### Phase 3 : Mettre à jour ou Réduire le Pas

```python
if test_err < 0 or test_err >= best_err:
    # La pente testée est invalide ou n'améliore pas
    curr_step *= 0.5  # Réduire le pas de moitié
else:
    # La pente testée améliore l'erreur
    best_err = test_err
    best_slope = test_slope
    get_derivative = True  # Recalculer la dérivée
```

**Logique** :
- Si ça marche → On met à jour et on continue dans cette direction
- Si ça ne marche pas → On réduit le pas et on réessaie

---

## 🏁 Étape 5 : Retourner le Résultat

```python
return (best_slope, -best_slope * pivot + y[pivot])
```

**Retourne** :
- `best_slope` : La pente optimale
- `intercept` : L'ordonnée à l'origine (calculée pour passer par le pivot)

---

## 📊 Autres Fonctions Utiles

### `fit_upper_trendline` (Résistance uniquement)

```python
def fit_upper_trendline(data: np.array):
    x = np.arange(len(data))
    coefs = np.polyfit(x, data, 1)
    line_points = coefs[0] * x + coefs[1]
    upper_pivot = (data - line_points).argmax()
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], data)
    return resist_coefs
```

**Utilisation** : Quand on veut seulement la trendline de résistance.

---

### `fit_lower_trendline` (Support uniquement)

```python
def fit_lower_trendline(data: np.array):
    x = np.arange(len(data))
    coefs = np.polyfit(x, data, 1)
    line_points = coefs[0] * x + coefs[1]
    lower_pivot = (data - line_points).argmin()
    support_coefs = optimize_slope(True, lower_pivot, coefs[0], data)
    return support_coefs
```

**Utilisation** : Quand on veut seulement la trendline de support.

---

### `fit_trendlines_high_low` (Avec High et Low)

```python
def fit_trendlines_high_low(high: np.array, low: np.array, close: np.array):
    x = np.arange(len(close))
    coefs = np.polyfit(x, close, 1)
    line_points = coefs[0] * x + coefs[1]
    upper_pivot = (high - line_points).argmax()
    lower_pivot = (low - line_points).argmin()

    support_coefs = optimize_slope(True, lower_pivot, coefs[0], low)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], high)

    return (support_coefs, resist_coefs)
```

**Différence** :
- Utilise les prix **High** pour la résistance
- Utilise les prix **Low** pour le support
- Plus précis car utilise toute la bougie, pas juste le close

---

## 🎨 Visualisation Complète

```
    PRIX (BTC/USDT)
      ▲
      │
      │   ╭───╮           ╭───────────────  ← Résistance (resist_coefs)
      │  ╱     ╲         ╱│
      │ ╱       ╲       ╱ │
      │╱         ╲     ╱  │
      │           ╲   ╱   │
      │            ╲ ╱    │
      │             ╲     │
      │              ╲    │
      │               ╲   │
      │                ╲  │
      │                 ╲ │
      │                  ╲│
      │══════════════════════════════════  ← Support (support_coefs)
      │
      └────────────────────────────────────► TEMPS

      ←────────────────→
         lookback = 72
```

---

## 📝 Résumé de l'Algorithme

```
┌─────────────────────────────────────────────────────────────┐
│  1. Calculer la ligne de best fit (polyfit degré 1)         │
│     → coefs = (slope, intercept)                            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Trouver les points pivots                               │
│     → upper_pivot = point le plus au-dessus (argmax)        │
│     → lower_pivot = point le plus en-dessous (argmin)       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Optimiser la pente pour le support                      │
│     → Contrainte: ligne en-dessous de TOUS les prix         │
│     → Objectif: minimiser somme des erreurs²                │
│     → Algorithme: descente de gradient avec pas adaptatif   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Optimiser la pente pour la résistance                   │
│     → Contrainte: ligne au-dessus de TOUS les prix          │
│     → Objectif: minimiser somme des erreurs²                │
│     → Algorithme: descente de gradient avec pas adaptatif   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  5. Retourner les coefficients                              │
│     → support_coefs = (slope, intercept)                    │
│     → resist_coefs = (slope, intercept)                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Points Clés à Retenir

1. **Contrainte forte** : La trendline doit être d'un seul côté de TOUS les prix
2. **Point pivot** : Point extrême qui sert d'ancrage pour la trendline
3. **Optimisation itérative** : Ajuste la pente pas à pas jusqu'à converger
4. **Minimisation des erreurs** : Cherche la ligne la plus "proche" des prix
5. **Deux trendlines** : Support (en-dessous) et Résistance (au-dessus)

---

*Document suivant : [04 - Création du Dataset](./04_dataset_creation.md)*
