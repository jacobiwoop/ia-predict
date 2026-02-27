# 01 - Introduction et Concepts de Base

## 🎯 Objectif du Projet

Ce projet a pour but de créer une **stratégie de trading automatisée** qui détecte les **breakouts** (franchissements) de lignes de tendance sur le prix du Bitcoin, puis utilise le **Machine Learning** pour filtrer les faux signaux.

---

## 📈 Qu'est-ce qu'une Ligne de Tendance ?

### Définition Simple

Une **ligne de tendance** (trendline) est une ligne droite tracée sur un graphique de prix qui relie :
- **Pour une trendline de support (inférieure)** : Les points les plus bas successifs
- **Pour une trendline de résistance (supérieure)** : Les points les plus hauts successifs

```
                    ╭─────────╮         ╭───── Résistance (trendline supérieure)
    PRIX            │         ╰─────╮   │
      ▲             │               ╰───╫─────────
      │           ╭─╯                 │ │
      │     ╭─────╯                   │ │
      │     │                         │ │
      │   ╭─╯                         │ │
      │   │                           │ │
      │   │     ╭─────────────────────╫─╯
      │   │     │                     │
      │   ╰─────╯                     │
      │                               │
      └───────────────────────────────┴──────────►
                                  TEMPS
                    ╰─────────╯         ╰───── Support (trendline inférieure)
```

### À Quoi Ça Sert ?

Les lignes de tendance agissent comme des **barrières invisibles** :
- Le prix a tendance à **rebondir** sur ces lignes
- Quand le prix **franchit** (breakout) la ligne, cela indique souvent un **changement de tendance**

---

## 🚀 Qu'est-ce qu'un Breakout ?

### Définition

Un **breakout** se produit lorsque le prix **traverse** une ligne de tendance ou un niveau de support/résistance.

```
                    ╭─────────╮
    PRIX            │         ╰─────╮   ╭───────────
      ▲             │               ╰───╯  ↑
      │           ╭─╯                    │ BREAKOUT !
      │     ╭─────╯                      │ (le prix franchit
      │     │                            │  la résistance)
      │   ╭─╯                            │
      │   │                              │
      │   │                              │
      └───┴──────────────────────────────┴──────────►
                                      TEMPS
```

### Pourquoi Trader les Breakouts ?

Quand un breakout se produit :
1. **La tendance accélère** : Le prix a tendance à continuer dans la direction du breakout
2. **Signal d'entrée** : C'est un bon moment pour entrer en position
3. **Moins de risque** : On sait rapidement si on a tort (le prix revient en arrière)

---

## ⚠️ Le Problème : Les Faux Breakouts

### C'est Quoi un Faux Breakout ?

Un **faux breakout** se produit quand le prix franchit une ligne de tendance mais **revient rapidement en arrière**.

```
    PRIX
      ▲             ╭─╮
      │           ╭─╯ ╰──╮
      │     ╭─────╯      ╰───╮  ╭─── FAUX !
      │     │                ╰──╯   (le prix revient)
      │   ╭─╯                     │
      │   │                       │
      │   │                       │
      └───┴───────────────────────┴──────────►
                                  TEMPS
```

### Conséquence

Si on prend **tous** les breakouts :
- Beaucoup de **pertes** dues aux faux signaux
- Profit factor faible (~1.02)
- Win rate d'environ 50%

---

## 🤖 La Solution : Meta-Labeling (Machine Learning)

### Qu'est-ce que le Meta-Labeling ?

Le **Meta-Labeling** est une technique de Machine Learning où on entraîne un modèle à **prédire si un trade va être gagnant ou perdant**.

```
┌─────────────────────────────────────────────────────────────┐
│                    STRATÉGIE DE BASE                        │
│  (Trendline Breakout)                                       │
│                                                             │
│  Entrée : Données de prix                                   │
│  Sortie : Signal de trading (LONG / SHORT / NEUTRE)         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    MODÈLE DE ML                             │
│  (Random Forest Classifier)                                 │
│                                                             │
│  Entrée : Features du breakout                              │
│  Sortie : Probabilité que le trade soit gagnant             │
│                                                             │
│  Décision :                                                 │
│  • Si probabilité > 50% → PRENDRE LE TRADE                  │
│  • Si probabilité ≤ 50% → IGNORER LE TRADE                  │
└─────────────────────────────────────────────────────────────┘
```

### Analogie Simple

Imaginez que vous avez un **ami qui donne des conseils de trading** :
- Il vous dit : "Achète maintenant !" (c'est la stratégie de base)
- Mais avant de suivre son conseil, vous vérifiez :
  - Est-ce que le volume est bon ?
  - Est-ce que la tendance est forte ?
  - Est-ce que les autres indicateurs sont bons ?

Le modèle de ML, c'est comme un **deuxième ami expert** qui analyse les conseils du premier ami et vous dit :
> "Oui, ce conseil est bon, tu peux le suivre"
> ou
> "Non, ce conseil est douteux, ignore-le"

---

## 📊 Architecture Globale du Système

### Vue d'Ensemble

```
┌────────────────────────────────────────────────────────────────────┐
│                         DONNÉES BRUTES                             │
│  Fichier CSV : BTCUSDT3600.csv                                     │
│  (Prix Bitcoin horaire : Open, High, Low, Close, Volume)           │
└────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────┐
│  ÉTAPE 1 : Calcul des Lignes de Tendance                           │
│  Fichier : trendline_automation.py                                  │
│                                                                    │
│  • Calcule la ligne de support (en dessous des prix)               │
│  • Calcule la ligne de résistance (au-dessus des prix)             │
│  • Utilise une fenêtre glissante de 72 bougies                      │
└────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────┐
│  ÉTAPE 2 : Détection des Breakouts                                 │
│  Fichier : trendline_breakout.py                                    │
│                                                                    │
│  • Si prix > résistance → Signal LONG (+1)                         │
│  • Si prix < support → Signal SHORT (-1)                           │
│  • Sinon → On garde le signal précédent                            │
└────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────┐
│  ÉTAPE 3 : Création du Dataset                                     │
│  Fichier : trendline_break_dataset.py                               │
│                                                                    │
│  Pour chaque breakout détecté :                                    │
│  • Enregistre les 5 features (indicateurs)                         │
│  • Calcule le résultat du trade (win/loss)                         │
│  • Stop Loss et Take Profit à 3 x ATR                              │
│  • Hold period maximum : 12 bougies                                │
└────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────┐
│  ÉTAPE 4 : Entraînement du Modèle                                  │
│  Fichier : walkforward.py                                           │
│                                                                    │
│  • Random Forest Classifier (1000 arbres, max_depth=3)             │
│  • Entraînement glissant sur 2 ans de données                      │
│  • Réentraînement tous les ans                                     │
│  • Prédit la probabilité de succès                                 │
└────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────┐
│  ÉTAPE 5 : Filtrage des Signaux                                    │
│                                                                    │
│  • Si probabilité > 0.5 → Exécuter le trade                        │
│  • Si probabilité ≤ 0.5 → Ignorer le trade                         │
│                                                                    │
│  Résultat : Moins de trades, mais de meilleure qualité             │
└────────────────────────────────────────────────────────────────────┘
```

---

## 🔍 Les 5 Features (Indicateurs) du Modèle

Le modèle utilise 5 indicateurs pour décider si un trade est bon :

| Feature | Description | Intuition |
|---------|-------------|-----------|
| **resist_s** | Pente de la résistance / ATR | Les breakouts en tendance haussière sont meilleurs |
| **tl_err** | Distance moyenne prix/résistance | Plus les prix sont proches, mieux c'est |
| **max_dist** | Distance maximale prix/résistance | Une grande distance = signal faible |
| **vol** | Volume normalisé | Un breakout avec fort volume est plus fiable |
| **adx** | ADX (force de la tendance) | Une tendance forte aide le breakout |

*Chaque feature est expliquée en détail dans le fichier [05_features_explication.md](./05_features_explication.md)*

---

## 📁 Structure des Fichiers

```
mt5script/
│
├── TrendlineBreakoutMetaLabel/
│   ├── trendline_automation.py    ← Calcul mathématique des trendlines
│   ├── trendline_breakout.py      ← Stratégie de breakout simple
│   ├── trendline_break_dataset.py ← Création du dataset ML
│   ├── walkforward.py             ← Entraînement et test du modèle
│   ├── in_sample_test.py          ← Visualisation et analyse
│   ├── BTCUSDT3600.csv            ← Données de prix (hourly)
│   └── README.md                  ← Lien vers la vidéo YouTube
│
└── documentation/                 ← Cette documentation
    ├── README.md
    ├── 01_introduction.md
    ├── 02_strategie_simple.md
    ├── 03_trendline_calculation.md
    ├── 04_dataset_creation.md
    ├── 05_features_explication.md
    ├── 06_meta_labeling.md
    ├── 07_walkforward.md
    └── scenarios/
        └── (fichiers interactifs)
```

---

## 🎓 Prérequis pour Comprendre

Pour bien comprendre cette stratégie, il faut connaître :

### Bases de Trading
- ✅ Qu'est-ce qu'une bougie (candlestick)
- ✅ Support et résistance
- ✅ Long (achat) vs Short (vente)
- ✅ Stop Loss et Take Profit

### Concepts Techniques
- ✅ Lignes de tendance (trendlines)
- ✅ ATR (Average True Range) - volatilité
- ✅ ADX (Average Directional Index) - force de tendance
- ✅ Log returns (rendements logarithmiques)

### Machine Learning
- ✅ Classification binaire (win/loss)
- ✅ Random Forest (forêt aléatoire)
- ✅ Features et Labels
- ✅ Walkforward validation

*Si un de ces concepts ne vous est pas familier, ne vous inquiétez pas ! Chaque concept sera expliqué en détail dans les fichiers suivants.*

---

## 📈 Prochaines Étapes

Maintenant que vous avez une vue d'ensemble, vous pouvez :

1. **[Lire la stratégie simple](./02_strategie_simple.md)** - Comment détecter les breakouts
2. **[Comprendre le calcul des trendlines](./03_trendline_calculation.md)** - Les mathématiques derrière
3. **[Voir comment le dataset est créé](./04_dataset_creation.md)** - Préparation des données pour le ML
4. **[Apprendre les 5 features](./05_features_explication.md)** - Les indicateurs clés
5. **[Comprendre le meta-labeling](./06_meta_labeling.md)** - Le modèle de ML
6. **[Voir le walkforward](./07_walkforward.md)** - Validation en temps réel

---

## 💡 Points Clés à Retenir

1. **Le breakout simple seul n'est pas suffisant** - Win rate de 50%, profit factor faible
2. **Le meta-labeling améliore la stratégie** - Filtre les faux signaux
3. **5 features principales** - Slope, erreur trendline, distance max, volume, ADX
4. **Random Forest avec max_depth=3** - Assez simple pour éviter l'overfitting
5. **Walkforward validation** - Teste la stratégie comme en trading réel

---

*Document suivant : [02 - Stratégie Simple (Sans ML)](./02_strategie_simple.md)*
