# Pronostic Coupe du Monde 2026

Système de prédiction statistique pour la **FIFA World Cup 2026**
(11 juin – 19 juillet, USA/Canada/Mexique, 48 équipes, 12 groupes).

---

## Livrables

| Fichier | Contenu |
|---|---|
| `outputs/matchs.csv` | Score prédit + P(1)/P(N)/P(2) pour chaque match |
| `outputs/champion.csv` | Probabilité de titre par équipe (Monte-Carlo) |
| `outputs/buteurs.csv` | Probabilité Soulier d'or par joueur ⚠ |

---

## Installation

```bash
pip install -r requirements.txt
```

Aucune autre dépendance. La clé API football-data.org est dans `config.py`
(ou surchargeable via la variable d'environnement `FOOTBALL_DATA_API_KEY`).

---

## Utilisation

```bash
# Run complet (1re fois) : télécharge tout, estime le modèle, simule
python predict.py

# Mise à jour quotidienne pendant le tournoi
python predict.py --update

# Backtest + prédiction
python predict.py --backtest

# Re-télécharge toutes les données et re-estime
python predict.py --force

# Test rapide avec moins de simulations
python predict.py --n-sims 1000
```

---

## Méthodologie

### Ratings

- **Elo international** : [eloratings.net](https://www.eloratings.net/) — TSV téléchargeable, mis à jour en continu.
- **Classement FIFA** : [fifa.com/fifa-world-ranking](https://www.fifa.com/fifa-world-ranking/) — utilisé comme feature de validation croisée.

### Modèle de scores (Dixon-Coles)

Référence : *Dixon & Coles (1997)*, Applied Statistics.

Paramètres estimés par **maximum de vraisemblance pondéré temporellement** :
- `α_i` (attaque), `β_i` (faiblesse défensive), `γ` (avantage domicile), `ρ` (correction scores bas)
- Pondération : `w = exp(−ln(2) × Δjours / 547)` — demi-vie de 18 mois
- Données d'entraînement : résultats internationaux 2020–2024 (≈ 5 000 matchs)
- Terrain neutre (CdM) : `γ = 1`

La correction Dixon-Coles ajuste les probabilités des scores 0-0, 1-0, 0-1, 1-1
pour tenir compte de leur sur/sous-représentation par rapport au modèle de Poisson pur.

Pour les équipes avec peu de données, un **prior Elo** est mélangé avec les paramètres DC
(poids 25 % Elo / 75 % DC).

### Issue d'un match (Elo)

```
P(victoire A) = 1 / (1 + 10^(−ΔElo/400))
P(nul) ≈ 0.265 × exp(−(|ΔElo| / 200)²)
```

Paramètres calibrés sur les résultats historiques (voir `backtest.py`).

### Simulation du tournoi (Monte-Carlo)

- **20 000 runs** avec seed fixée à `42`
- Phase de groupes : round-robin, classement FIFA standard (pts → diff buts → buts marqués → H2H → aléatoire)
- 8 meilleurs 3es : classés par points → diff buts → buts marqués
- Phase KO : prolongation (Poisson × 0.35) + tirs au but (50/50) si égalité
- À chaque run, les matchs déjà joués sont figés au score réel

#### Bracket R32 (hypothèse)

Le bracket utilisé est une approximation basée sur les conventions FIFA.
Il sera mis à jour automatiquement depuis football-data.org dès que les matchs
KO seront publiés avec leurs slots officiels.

Groupes A-H : 12 matchs 1er vs 2e cross-groupes
Groupes I-L : 4 matchs 1er vs meilleur 3e + 4 matchs 2e vs meilleur 3e

### Soulier d'or (⚠ prédiction très incertaine)

À chaque match simulé, les buts attendus de l'équipe sont répartis entre ses joueurs
selon :

```
poids_joueur = taux_buts_récent × prob_titularisation × (1.3 si tireur de pénos)
```

**Sources** :
- `goalscorers.csv` (martj42) — buts par joueur en sélection (primaire)
- FBref.com — caps récents, confirmation des titulaires (enrichissement)
- Transfermarkt — fallback si FBref non disponible

**Limitations** : ne modélise pas les blessures, suspensions, rotations tactiques.
La variance sur cette prédiction est structurellement élevée (voir `avg_goals_per_sim`).

---

## Sources de données

| Donnée | Source | Méthode | Mise à jour |
|---|---|---|---|
| Ratings Elo | eloratings.net | GET TSV public | Continue |
| Classement FIFA | fifa.com | Scrape HTML/JSON | Mensuelle |
| Résultats historiques | github.com/martj42/international-results | CSV GitHub raw | Mensuelle |
| Buts par joueur | idem (goalscorers.csv) | CSV GitHub raw | Mensuelle |
| Fixtures + résultats live | football-data.org v4 | REST API (clé gratuite) | Quotidienne |
| Stats joueurs caps | FBref.com (StatsBomb) | Scrape HTML | Hebdomadaire |

---

## Structure des fichiers

```
pronostic-cdm/
├── config.py                   # Seed, chemins, clés API, hyperparamètres
├── predict.py                  # Point d'entrée principal
├── backtest.py                 # Calibration du modèle sur 2024-2025
├── ingestion/
│   ├── elo_fetch.py            # Elo ratings
│   ├── fifa_ranking.py         # Classement FIFA
│   ├── results_fetch.py        # Résultats historiques (martj42)
│   ├── fixtures_fetch.py       # Fixtures CdM 2026 + résultats live
│   ├── players_fetch.py        # Buts par joueur en sélection
│   └── competition_history.py  # Palmarès par équipe (WC, Euro, Copa...)
├── model/
│   ├── elo.py                  # Formule Elo → probabilités
│   └── poisson_dixoncoles.py   # MLE Dixon-Coles + matrice de scores
├── sim/
│   └── tournament.py           # Monte-Carlo : groupes → KO → titre
├── data/                       # Cache auto-généré (gitignore)
└── outputs/
    ├── matchs.csv
    ├── champion.csv
    └── buteurs.csv
```

---

## Reproductibilité

Seed globale : `SEED = 42` dans `config.py`.
Tous les générateurs aléatoires (numpy) utilisent cette seed.
Le modèle DC est sérialisé dans `data/dc_params.pkl` après chaque estimation.

---

## Backtest (calibration)

```bash
python backtest.py
```

Protocole :
- Entraînement : jusqu'à septembre 2024
- Test : octobre 2024 – mai 2026 (Euro 2024, Copa América 2024, Nations League, qualifications)
- Métriques : Brier score, log-loss, accuracy 1/N/2, RMSE buts, courbe de calibration

Résultats sauvegardés dans `outputs/backtest_metrics.csv`.
