# Documentation - Stratégie de Trading Trendline Breakout

## 📚 Table des Matières

Cette documentation explique en détail le fonctionnement de la stratégie de trading **Trendline Breakout** avec filtrage par **Meta-Labeling** (Machine Learning).

### Fichiers de Documentation

| Fichier | Description |
|---------|-------------|
| [01_introduction.md](./01_introduction.md) | Présentation générale du projet et concepts de base |
| [02_strategie_simple.md](./02_strategie_simple.md) | La stratégie de breakout simple (sans ML) |
| [03_trendline_calculation.md](./03_trendline_calculation.md) | Comment les lignes de tendance sont calculées |
| [04_dataset_creation.md](./04_dataset_creation.md) | Création du dataset pour le Machine Learning |
| [05_features_explication.md](./05_features_explication.md) | Les 5 features/indicateurs utilisés par le modèle |
| [06_meta_labeling.md](./06_meta_labeling.md) | Le modèle de Machine Learning et son entraînement |
| [07_walkforward.md](./07_walkforward.md) | Test en walkforward (validation temporelle) |

### Fichiers de Scénarios et Visualisations

| Fichier | Description |
|---------|-------------|
| [scenarios/](./scenarios/) | Scénarios interactifs et schémas animés |

---

## 🎯 Résumé du Projet

Ce projet implémente une stratégie de trading basée sur les **breakouts de lignes de tendance** sur le prix du Bitcoin (BTC/USDT), avec un filtrage des faux signaux par **Machine Learning**.

### Architecture du Code

```
trendline_breakout.py      → Stratégie de base (détection des breakouts)
trendline_automation.py    → Calcul mathématique des lignes de tendance
trendline_break_dataset.py → Création du dataset avec features pour le ML
walkforward.py             → Entraînement et test du modèle en walkforward
in_sample_test.py          → Analyse et visualisation des trades
```

### Flux de Données

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DONNÉES BRUTES (CSV)                          │
│                    OHLCV : Open, High, Low, Close, Volume               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    1. CALCUL DES LIGNES DE TENDANCE                     │
│              (trendline_automation.py - fit_trendlines_single)          │
│   • Ligne de support (inférieure)                                       │
│   • Ligne de résistance (supérieure)                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   2. DÉTECTION DES BREAKOUTS                            │
│              (trendline_breakout.py)                                    │
│   • Prix > Résistance → Signal LONG (+1)                                │
│   • Prix < Support → Signal SHORT (-1)                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              3. CRÉATION DU DATASET (pour le ML)                        │
│           (trendline_break_dataset.py)                                  │
│   • Extraction des 5 features                                           │
│   • Calcul des labels (win/loss)                                        │
│   • Stop Loss / Take Profit à 3 ATR                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   4. ENTRAÎNEMENT DU MODÈLE                             │
│              (walkforward.py - RandomForestClassifier)                  │
│   • Random Forest avec max_depth=3                                      │
│   • Entraînement glissant sur 2 ans                                     │
│   • Prédiction probabilité de succès                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   5. FILTRAGE DES SIGNAUX                               │
│   • Probabilité > 50% → Prendre le trade                                │
│   • Probabilité ≤ 50% → Ignorer le trade                                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Performance de la Stratégie

### Sans Machine Learning (Breakout Simple)
| Métrique | Valeur |
|----------|--------|
| Profit Factor | ~1.02 - 1.035 |
| Win Rate | ~50% |
| Average Trade | ~0.05% |
| Temps en marché | 100% |

### Avec Machine Learning (Meta-Labeling)
| Métrique | Valeur |
|----------|--------|
| Profit Factor | Amélioré |
| Win Rate | >50% |
| Average Trade | ~0.1% (doublé) |
| Temps en marché | ~20% |

---

## 🔧 Paramètres de la Stratégie

| Paramètre | Valeur par défaut | Description |
|-----------|-------------------|-------------|
| `lookback` | 72 | Nombre de bougies pour calculer les trendlines (72h = 3 jours) |
| `tp_mult` | 3.0 | Multiplicateur pour Take Profit (3 x ATR) |
| `sl_mult` | 3.0 | Multiplicateur pour Stop Loss (3 x ATR) |
| `hold_period` | 12 | Nombre maximum de bougies en position |
| `atr_lookback` | 168 | Période pour le calcul de l'ATR (168h = 1 semaine) |
| `train_size` | 365 * 24 * 2 | Taille de l'entraînement (2 ans) |
| `step_size` | 365 * 24 | Pas de réentraînement (1 an) |

---

## 📖 Glossaire

| Terme | Définition |
|-------|------------|
| **Breakout** | Quand le prix franchit un niveau de support ou résistance |
| **Trendline** | Ligne de tendance reliant des points de prix |
| **Support** | Niveau de prix où la tendance baissière s'arrête |
| **Résistance** | Niveau de prix où la tendance haussière s'arrête |
| **ATR** | Average True Range - mesure de la volatilité |
| **ADX** | Average Directional Index - mesure de la force de la tendance |
| **Meta-Labeling** | Technique de ML pour filtrer les signaux d'une stratégie |
| **Walkforward** | Méthode de validation qui simule un trading en temps réel |
| **Feature** | Variable/indicateur utilisé par le modèle de ML |
| **Label** | Résultat à prédire (ici: trade gagnant ou perdant) |

---

## 🚀 Comment Utiliser

### Prérequis
```bash
pip install numpy pandas pandas_ta matplotlib mplfinance scikit-learn
```

### Exécuter la stratégie simple
```bash
python trendline_breakout.py
```

### Générer le dataset
```bash
python trendline_break_dataset.py
```

### Lancer le walkforward avec ML
```bash
python walkforward.py
```

### Analyser les trades
```bash
python in_sample_test.py
```

---

## 📝 Notes Importantes

1. **Les pics de performance sont suspects** : Un spike de performance sur certains paramètres est souvent dû à la chance (overfitting).

2. **Le lookback de 72** est arbitraire mais fonctionne raisonnablement bien sur la plupart des valeurs.

3. **La profondeur des arbres (max_depth=3)** a été choisie car la cross-validation donne presque toujours 2 ou 3 comme optimal.

4. **Le modèle Random Forest** est utilisé car il gère bien le bruit dans les données financières.

5. **Les features sont normalisées par l'ATR** pour être indépendantes de la volatilité et du niveau de prix.

---

*Cette documentation est basée sur la vidéo YouTube : [Trendline Breakout Strategy](https://www.youtube.com/watch?v=jCBnbQ1PUkE)*
