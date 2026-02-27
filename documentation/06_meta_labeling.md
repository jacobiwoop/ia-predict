# 06 - Meta-Labeling et Machine Learning

## 🎯 Objectif de ce Chapitre

Comprendre le concept de **Meta-Labeling** et comment le modèle de **Random Forest** est utilisé pour filtrer les trades dans le fichier `walkforward.py`.

---

## 🤔 Qu'est-ce que le Meta-Labeling ?

### Définition Simple

Le **Meta-Labeling** est une technique où on utilise le Machine Learning non pas pour prédire directement les prix, mais pour **prédire si notre stratégie de trading va fonctionner**.

```
┌─────────────────────────────────────────────────────────────┐
│                    NIVEAU 1 : Stratégie                     │
│                    (Trendline Breakout)                     │
│                                                             │
│  Entrée : Données de prix                                   │
│  Sortie : Signal de trading (LONG / SHORT)                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    NIVEAU 2 : Meta-Label                    │
│                    (Random Forest Classifier)               │
│                                                             │
│  Entrée : Features du trade                                 │
│  Sortie : Probabilité que le trade soit gagnant             │
│                                                             │
│  Décision :                                                 │
│  • Si prob > 0.5 → Exécuter le trade                        │
│  • Si prob ≤ 0.5 → Ignorer le trade                         │
└─────────────────────────────────────────────────────────────┘
```

---

### Analogie : Le Conseiller en Investissement

Imaginez deux conseillers :

**Conseiller 1 (La Stratégie) :**
> "Achète cette action ! C'est un breakout !"

**Conseiller 2 (Le Modèle ML) :**
> "Attends, laisse moi analyser ce breakout..."
> "Hmm, le volume est faible, l'ADX est bas..."
> "Je donne 30% de chances que ce trade soit gagnant."
> "**N'achète pas**, c'est probablement un faux signal."

```
    Vous          Conseiller 1        Conseiller 2
      │                 │                   │
      │  "Quoi faire ?" │                   │
      │◄────────────────│                   │
      │                 │                   │
      │  "Achète !"     │                   │
      │────────────────►│                   │
      │                 │                   │
      │                 │  "Attends,        │
      │                 │   analysons..."    │
      │                 │──────────────────►│
      │                 │                   │
      │                 │  "30% de chances" │
      │                 │◄──────────────────│
      │                 │                   │
      │  "Non, attends" │                   │
      │◄────────────────│                   │
```

---

## 🌲 Le Modèle : Random Forest Classifier

### Pourquoi Random Forest ?

**De la vidéo :**
> "But when we train many decision trees in a random forest, the noise has a tendency to cancel out."

**Avantages de Random Forest :**

| Avantage | Pourquoi c'est important |
|----------|-------------------------|
| **Gère le bruit** | Les données financières sont très bruitées |
| **Non-linéaire** | Capture des relations complexes entre features |
| **Peu d'overfitting** | Surtout avec `max_depth=3` |
| **Interprétable** | On peut voir l'importance des features |
| **Robuste** | Fonctionne bien sans tuning excessif |

---

### Qu'est-ce qu'une Decision Tree ?

**Arbre de Décision Simplifié :**

```
                        ┌─────────────────┐
                        │  resist_s > 0 ? │
                        └────────┬────────┘
                                 │
                    ┌────────────┴────────────┐
                    │ YES                     │ NO
                    ▼                         ▼
            ┌───────────────┐         ┌───────────────┐
            │  adx > 25 ?   │         │  vol > 1.5 ?  │
            └───────┬───────┘         └───────┬───────┘
                    │                         │
            ┌───────┴───────┐         ┌───────┴───────┐
            │ YES           │ NO        │ YES           │ NO
            ▼               ▼           ▼               ▼
        ┌────────┐     ┌────────┐   ┌────────┐     ┌────────┐
        │ WIN    │     │ LOSS   │   │ WIN    │     │ LOSS   │
        │ 80%    │     │ 60%    │   │ 55%    │     │ 70%    │
        └────────┘     └────────┘   └────────┘     └────────┘
```

---

### Random Forest = Multiple Decision Trees

```
    Tree 1          Tree 2          Tree 3          Tree 4          Tree 5
       │               │               │               │               │
       ▼               ▼               ▼               ▼               ▼
    ┌─────┐         ┌─────┐         ┌─────┐         ┌─────┐         ┌─────┐
    │     │         │     │         │     │         │     │         │     │
    │  🌳 │         │  🌳 │         │  🌳 │         │  🌳 │         │  🌳 │
    │     │         │     │         │     │         │     │         │     │
    └─────┘         └─────┘         └─────┘         └─────┘         └─────┘
       │               │               │               │               │
       ▼               ▼               ▼               ▼               ▼
    WIN: 0.7        WIN: 0.6        WIN: 0.4        WIN: 0.8        WIN: 0.5

                          Moyenne = 0.60 (60%)
                              │
                              ▼
                    Si seuil = 50% → PRENDRE LE TRADE ✅
```

**Pourquoi ça marche :**
- Chaque arbre voit un sous-ensemble différent des données
- Le bruit se "cancel out" (s'annule) dans la moyenne
- La prédiction finale est plus stable qu'un seul arbre

---

## ⚙️ Configuration du Modèle

### Paramètres Utilisés

```python
model = RandomForestClassifier(
    n_estimators=1000,    # Nombre d'arbres
    max_depth=3,          # Profondeur maximum des arbres
    random_state=69420    # Graine aléatoire (reproductibilité)
)
```

---

### Pourquoi `n_estimators=1000` ?

**Plus d'arbres = Plus stable**

```
    Nombre d'arbres    Stabilité de la prédiction
    ─────────────────────────────────────────────
    10                 ❌ Très variable
    100                ⚠️ Correct
    500                ✅ Bon
    1000               ✅ Excellent
    5000               ✅ (mais diminishing returns)
```

**Pourquoi pas plus ?**
- Temps de calcul plus long
- Gains marginaux après 1000 arbres
- 1000 est un bon compromis performance/temps

---

### Pourquoi `max_depth=3` ?

**De la vidéo :**
> "I set max depth to 3 to control how deep the trees go. Ideally you should do a walk forward cross validation to set max depth, but I'm trying to keep this video from being too long, and in my experience a cross validation will almost always yield 2 or 3 as the best max depth."

**Explication :**

```
    max_depth = 1 (Trop simple)
    ┌─────────────────┐
    │  resist_s > 0 ? │
    └────────┬────────┘
             │
        ┌────┴────┐
        │         │
        ▼         ▼
     WIN:60%   WIN:40%

    → Sous-ajustement (underfitting)
    → Ne capture pas les relations complexes


    max_depth = 3 (Juste ce qu'il faut)
    ┌─────────────────┐
    │  resist_s > 0 ? │
    └────────┬────────┘
             │
        ┌────┴────┐
        │         │
        ▼         ▼
    ┌───────┐  ┌───────┐
    │adx>25?│  │vol>1.5│
    └───┬───┘  └───┬───┘
        │          │
    ┌───┴───┐  ┌───┴───┐
    ▼       ▼  ▼       ▼
   WIN    WIN WIN    WIN
   80%    50% 60%    30%

    → Bon équilibre
    → Capture les interactions sans overfitting


    max_depth = 10 (Trop complexe)
    ┌─────────────────┐
    │  resist_s > 0 ? │
    └────────┬────────┘
             │
        ┌────┴────────────────────┐
        │                        │
        ▼                        ▼
    ┌───────┐               ┌───────────┐
    │ ...   │               │ ...       │
    └───┬───┘               └─────┬─────┘
        │                        │
    ┌───┴───┐               ┌─────┴─────┐
    │ ...   │               │ ...       │
    └───┬───┘               └─────┬─────┘
        │                        │
        ▼                        ▼
    (10 niveaux de profondeur...)

    → Overfitting
    → Apprend le bruit par cœur
    → Ne généralise pas
```

---

### Pourquoi `random_state=69420` ?

**Reproductibilité :**

```python
# Avec random_state
model1 = RandomForestClassifier(random_state=69420)
model2 = RandomForestClassifier(random_state=69420)

# model1 et model2 donneront EXACTEMENT les mêmes résultats
# (même après plusieurs exécutions)


# Sans random_state
model3 = RandomForestClassifier()
model4 = RandomForestClassifier()

# model3 et model4 donneront des résultats DIFFÉRENTS
# (à cause du bootstrap aléatoire)
```

**Pourquoi c'est important ?**
- Permet de **reproduire** les résultats
- Essentiel pour le **backtesting** et la **validation**
- Utile pour le **débogage**

---

## 📚 Entraînement du Modèle

### Données d'Entraînement

```python
x_train = data_x.loc[train_indices]
y_train = data_y.loc[train_indices]

model.fit(x_train.to_numpy(), y_train.to_numpy())
```

**Features (X) :**
```
┌─────────────────────────────────────────┐
│  resist_s  tl_err  max_dist  vol  adx  │
├─────────────────────────────────────────┤
│   0.05     0.01    0.03     1.2   28   │  ← Trade 1
│   0.08     0.015   0.04     0.8   22   │  ← Trade 2
│  -0.055    0.02    0.05     1.5   35   │  ← Trade 3
│   ...      ...     ...      ...  ...   │
└─────────────────────────────────────────┘
```

**Labels (y) :**
```
┌─────────┐
│   1     │  ← Trade 1 = WIN
│   0     │  ← Trade 2 = LOSS
│   1     │  ← Trade 3 = WIN
│  ...    │
└─────────┘
```

---

### Prédiction

```python
prob = model.predict_proba(data_x.iloc[trade_i].to_numpy().reshape(1, -1))[0][1]
```

**Décomposé :**

```python
# 1. Prendre les features du trade actuel
features = data_x.iloc[trade_i]
# → [resist_s, tl_err, max_dist, vol, adx]

# 2. Convertir en tableau numpy
features_array = features.to_numpy()
# → array([0.05, 0.01, 0.03, 1.2, 28])

# 3. Reshaper pour avoir 2D (1 sample, 5 features)
features_2d = features_array.reshape(1, -1)
# → array([[0.05, 0.01, 0.03, 1.2, 28]])

# 4. Prédire les probabilités
probas = model.predict_proba(features_2d)
# → array([[0.35, 0.65]])
#    Classe 0 (LOSS)  Classe 1 (WIN)

# 5. Prendre la probabilité de WIN (classe 1)
prob = probas[0][1]
# → 0.65 (65% de chances de win)
```

---

### Interprétation de la Probabilité

```
    Probabilité    Interprétation          Décision
    ─────────────────────────────────────────────────
    0.90           90% de win             ✅ PRENDRE (très confiant)
    0.70           70% de win             ✅ PRENDRE (confiant)
    0.60           60% de win             ✅ PRENDRE
    0.55           55% de win             ✅ PRENDRE (juste au-dessus)
    0.50           50% de win             ⚠️ SEUIL (pile ou face)
    0.45           45% de win             ❌ IGNORER
    0.30           30% de win             ❌ IGNORER (peu confiant)
    0.10           10% de win             ❌ IGNORER (très peu confiant)
```

---

## 🎯 Seuil de Décision

### Seuil par Défaut : 0.5 (50%)

```python
if prob > 0.5:  # greater than 50%, take trade
    signal[i] = 1
```

**Pourquoi 0.5 ?**
- C'est le point d'équilibre
- > 50% = Plus de chances de gagner que de perdre
- < 50% = Plus de chances de perdre que de gagner

---

### Ajuster le Seuil

On peut utiliser un seuil différent selon l'objectif :

```python
# Seuil plus élevé (plus sélectif)
if prob > 0.6:  # Seulement les trades très confiants
    signal[i] = 1

# Résultat :
# - Moins de trades
# - Win rate plus élevé
# - Mais on rate des opportunités


# Seuil plus bas (moins sélectif)
if prob > 0.4:  # Plus de trades acceptés
    signal[i] = 1

# Résultat :
# - Plus de trades
# - Win rate plus faible
# - Mais plus d'opportunités capturées
```

---

### Trade-off : Nombre de Trades vs Performance

```
    Seuil      Trades    Win Rate    Profit Factor
    ───────────────────────────────────────────────
    0.40       35%       52%         1.15
    0.45       30%       54%         1.18
    0.50       25%       56%         1.20  ← Default
    0.55       20%       58%         1.22
    0.60       15%       60%         1.25
    0.65       10%       63%         1.28
    0.70        5%       67%         1.30

    → Plus le seuil est élevé, moins on prend de trades
    → Mais les trades pris sont de meilleure qualité
```

---

## 📊 Enregistrement des Résultats

### Sauvegarde de la Probabilité

```python
trades.loc[trade_i, 'model_prob'] = prob
```

**Utilité :**
- Analyser les trades pris vs ignorés
- Comprendre les décisions du modèle
- Ajuster le seuil si nécessaire

---

### Exemple de Trades avec Probabilités

| Trade # | Features | Probabilité | Décision (seuil 0.5) | Résultat |
|---------|----------|-------------|---------------------|----------|
| 1 | resist_s=0.08, adx=35, vol=1.8 | 0.72 | ✅ PRENDRE | ✅ WIN |
| 2 | resist_s=-0.02, adx=18, vol=0.7 | 0.28 | ❌ IGNORER | ❌ LOSS |
| 3 | resist_s=0.03, adx=28, vol=1.2 | 0.55 | ✅ PRENDRE | ❌ LOSS |
| 4 | resist_s=0.12, adx=42, vol=2.1 | 0.85 | ✅ PRENDRE | ✅ WIN |
| 5 | resist_s=-0.05, adx=22, vol=0.9 | 0.35 | ❌ IGNORER | ✅ WIN |

**Observations :**
- Le modèle n'est pas parfait (Trade 3 = LOSS, Trade 5 = WIN)
- Mais sur beaucoup de trades, il améliore la performance globale
- Les trades ignorés avec faible probabilité sont souvent des pertes évitées

---

## 🎯 Résumé du Processus de Décision

```
┌─────────────────────────────────────────────────────────────┐
│  1. Breakout détecté par la stratégie                       │
│     → Prix > Résistance                                     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Extraction des features                                 │
│     → [resist_s, tl_err, max_dist, vol, adx]                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Prédiction du modèle                                    │
│     → model.predict_proba(features)                         │
│     → prob = 0.67 (67%)                                     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Comparaison avec le seuil                               │
│     → prob > 0.5 ?                                          │
│     → 0.67 > 0.5 → OUI                                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  5. Décision                                                │
│     → PRENDRE LE TRADE ✅                                   │
│     → Entrée, TP, SL, HP définis                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Points Clés à Retenir

1. **Meta-Labeling = ML sur les signaux d'une stratégie** - Pas de prédiction directe des prix
2. **Random Forest = Multiple Decision Trees** - Moyenne des prédictions pour réduire le bruit
3. **max_depth=3** - Assez profond pour capturer des relations, pas trop pour éviter l'overfitting
4. **Probabilité > 0.5 = Prendre le trade** - Seuil ajustable selon l'objectif
5. **Le modèle n'est pas parfait** - Mais améliore la performance globale sur beaucoup de trades

---

*Document suivant : [07 - Walkforward Validation](./07_walkforward.md)*
