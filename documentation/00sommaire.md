# Documentation - Stratégie de Trading Trendline Breakout

## 📚 Sommaire de la Documentation

Cette documentation explique de manière détaillée le fonctionnement d'une stratégie de trading automatique basée sur les cassures de lignes de tendance, avec filtrage par machine learning.

---

## 📁 Fichiers de Documentation

| Fichier | Description |
|---------|-------------|
| [01_introduction.md](./01_introduction.md) | Présentation générale du projet et concepts de base |
| [02_strategie_simple.md](./02_strategie_simple.md) | La stratégie de base sans machine learning |
| [03_lignes_tendance.md](./03_lignes_tendance.md) | Comment fonctionnent les lignes de tendance |
| [04_dataset_trades.md](./04_dataset_trades.md) | Création du dataset de trades pour le ML |
| [05_features_indicateurs.md](./05_features_indicateurs.md) | Les 5 indicateurs/features pour le modèle |
| [06_meta_labeling.md](./06_meta_labeling.md) | Le modèle de machine learning (Random Forest) |
| [07_walkforward.md](./07_walkforward.md) | Système de validation walk-forward |
| [08_resultats.md](./08_resultats.md) | Résultats et performance de la stratégie |

---

## 🗂️ Fichiers du Projet Original

| Fichier | Rôle |
|---------|------|
| `trendline_breakout.py` | Stratégie de base (sans ML) |
| `trendline_automation.py` | Fonctions pour dessiner les lignes de tendance |
| `trendline_break_dataset.py` | Création du dataset de trades |
| `walkforward.py` | Modèle ML et validation walk-forward |
| `in_sample_test.py` | Tests et visualisations |
| `BTCUSDT3600.csv` | Données de prix (Bitcoin hourly) |

---

## 🎯 En Résumé

**Objectif du projet :** Créer une stratégie de trading qui :
1. Détecte les cassures de lignes de tendance
2. Utilise le machine learning pour filtrer les faux signaux
3. Améliore la performance par rapport à une stratégie naive

**Concept clé :** Le "meta-labeling" - au lieu de prédire directement si le prix va monter ou descendre, on prédit si une stratégie de trading donnée va fonctionner ou non sur un signal particulier.
