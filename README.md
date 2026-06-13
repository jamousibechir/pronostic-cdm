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

# Re-télécharge toutes les données et re-estime
python predict.py --force

# Test rapide avec moins de simulations
python predict.py --n-sims 1000

# Backtest hors échantillon du modèle de match
python backtest.py

# Validation de la couche simulation sur les Mondiaux passés (coûteux)
python validate_tournament.py --years 2022 --n-sims 1000

# Mise à jour quotidienne PENDANT le tournoi (xG + K-factor + ajustements)
python update_daily.py
```

> Sous PowerShell : préfixer avec `$env:PYTHONIOENCODING="utf-8"` et suffixer les
> commandes longues avec `2>$null` pour masquer la barre de progression tqdm.

---

## Méthodologie

### Ratings

- **Elo international** : [eloratings.net](https://www.eloratings.net/) — TSV téléchargeable, mis à jour en continu.
- **Classement FIFA** : [fifa.com/fifa-world-ranking](https://www.fifa.com/fifa-world-ranking/) — utilisé comme feature de validation croisée.

### Modèle de scores (Poisson à force unique)

Référence : *Ley, Van de Wiele & Van Eetvelde (2019)*, Statistical Modelling —
en sélection nationale, **une seule force par équipe** prédit mieux et de façon
plus calibrée que le couple attaque/défense de Dixon-Coles, car bien plus robuste
à la rareté des données. Correction des scores bas : *Dixon & Coles (1997)*.

Paramétrisation (échelle log), estimée par **MLE pondéré temporellement** :

```
log λ_dom = β0 + (r_i − r_j) + h · 1[pas neutre]
log μ_ext = β0 + (r_j − r_i)
```

- `r_i` : force de l'équipe i (un seul paramètre/équipe au lieu de deux)
- `β0` : taux de buts de base ; `h` : avantage domicile (additif en log)
- `ρ` : correction Dixon-Coles des scores 0-0/1-0/0-1/1-1
- Pondération : `w = exp(−ln(2) × Δjours / 547)` — demi-vie de 18 mois
- Données : ~49 500 matchs internationaux (martj42), fenêtre glissante de 8 ans

**Points clés de la mise en œuvre :**

| Aspect | Choix | Pourquoi |
|---|---|---|
| Gradient | **Analytique** (fourni à L-BFGS-B) | Convergence garantie en < 0,1 s (l'ancien DC ne convergeait pas) |
| Shrinkage | **Hiérarchique à 2 niveaux** (équipe → confédération → global) | Corrige le biais des équipes dominant des qualifs faibles (cf. ci-dessous) |
| Calibration | **Isotonique post-hoc, auto-désactivante** | N'agit que si elle améliore la log-loss en validation |
| Incertitude | **Block-bootstrap (40 réplicas)** | Propage l'incertitude d'estimation en préservant l'autocorrélation temporelle |
| Avantage hôte | **Bonus offensif** pour USA/Mexique/Canada en CdM | Les 3 hôtes jouent réellement à domicile |

#### Shrinkage hiérarchique par confédération

Le foot international forme un graphe « qui a joué qui » mal connecté (l'AFC joue
surtout l'AFC, etc.). Un modèle à force unique simple **surévalue** alors les
équipes qui écrasent des qualifications faibles. On ajoute donc un prior à deux
niveaux :

```
pénalité = λ_team · Σ_i (r_i − m_{c(i)})²  +  λ_conf · Σ_c m_c²
```

où `m_c` est la force moyenne de la confédération `c`, identifiée surtout par les
matchs **inter-confédérations**. Résultat : les `m_c` estimés sont
`CONMEBOL +0.9 > UEFA +0.6 > CONCACAF +0.2 > CAF ≈ 0 > AFC −0.1 > OFC −0.7`, et
le biais disparaît (ex. le Japon passe du 3ᵉ rang des forces au ~12ᵉ).

- `λ_team` est réglé par validation temporelle.
- `λ_conf` est **fixé par prior de domaine** : la validation, dominée par
  l'intra-confédération (où le niveau de conf s'annule dans `r_i − r_j`), n'est
  pas sensible à `λ_conf`. Il est validé indirectement par le backtest de tournoi
  (`validate_tournament.py`).
- Repli d'une équipe hors échantillon : projection depuis son Elo, puis moyenne
  de sa confédération.

### Issue d'un match (Elo)

Utilisé comme feature d'initialisation/repli :

```
P(victoire A) = 1 / (1 + 10^(−ΔElo/400))
```

### Validation de la couche simulation

`validate_tournament.py` vérifie que la simulation n'est pas seulement bonne
au niveau match mais **calibrée au niveau tournoi** : sur les WC 2014/2018/2022,
il entraîne le modèle sur les matchs antérieurs, rejoue le Monte-Carlo (format 32)
et compare les probabilités d'accès par tour (8ᵉs, quarts, demies, finale) à la
réalité. Une équipe annoncée à 70 % d'atteindre les quarts doit y parvenir ~70 %.

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

**Source** : `goalscorers.csv` (martj42) — ~47 600 buts en sélection, avec
buteur, minute, pénalty. Le `goal_rate` est calculé comme
`buts_récents / matchs_de_l'équipe_sur_la_fenêtre` (le dénominateur vient de
`results.csv`, car goalscorers.csv ne contient pas les présences). La
`start_prob` décroît selon les mois depuis le dernier but (écarte les retraités).

**Limitations** : ne modélise pas les blessures, suspensions, rotations tactiques.
La variance sur cette prédiction est structurellement élevée (voir `avg_goals_per_sim`).

### Mise à jour quotidienne en tournoi (`update_daily.py`)

Pendant le Mondial, un modèle a priori devient vite obsolète et la variance d'un
tournoi court est élevée. Le pipeline quotidien réinjecte l'information du jour :

1. **Résultats live** (football-data + martj42).
2. **Débruitage bayésien par les xG** : le score d'un match CdM est remplacé par
   `XG_BLEND·xG + (1−XG_BLEND)·score`. Une équipe malchanceuse (défaite 1-0 avec
   3.0 xG contre 0.5) voit sa pénalité de force fortement atténuée.
   Fichier d'entrée : `data/xg_daily.csv`.
3. **K-factor** : les matchs CdM en cours reçoivent un poids fortement majoré
   (`WC_K_FACTOR`), pour que le modèle apprenne vite la forme du moment.
4. **Ajustements d'effectif** : `data/daily_adjustments.json` applique des
   malus/bonus manuels (blessure, suspension) en unités de log-force
   (voir `daily_adjustments.example.json`).
5. **Re-simulation** sur le bracket partiel (matchs joués figés).

---

## Sources de données

| Donnée | Source | Méthode | Mise à jour |
|---|---|---|---|
| Résultats internationaux 1872–2026 (~49 500 matchs, toutes compétitions) | github.com/martj42/**international_results** | CSV GitHub raw | Quotidienne |
| Buts par joueur (~47 600) | martj42/international_results (`goalscorers.csv`) | CSV GitHub raw | Quotidienne |
| Ratings Elo | eloratings.net | GET TSV public (col 2 = code pays, col 3 = elo) | Continue |
| Stades WC (détection champions) | github.com/jfjelstul/worldcup | CSV GitHub raw (`stage_name`) | Définitif |
| Fixtures + résultats live WC 2026 | football-data.org v4 | REST API (clé gratuite) | Quotidienne |
| Classement FIFA | fifa.com / repli Elo | Scrape JSON, non bloquant | Mensuelle |

> ⚠ Le dépôt martj42 a été **renommé** `international-results` → `international_results`
> (underscore) ; l'ancienne URL avec tiret renvoie 404. Tous les noms d'équipes
> sont normalisés vers l'espace martj42 par `ingestion/names.py`.

---

## Structure des fichiers

```
pronostic-cdm/
├── config.py                   # Seed, chemins, clés API, hyperparamètres
├── predict.py                  # Point d'entrée principal
├── backtest.py                 # Backtest hors échantillon du modèle de match
├── validate_tournament.py      # Validation de la simulation (WC passés)
├── update_daily.py             # Pipeline quotidien en tournoi (xG + K-factor)
├── ingestion/
│   ├── names.py                # Normalisation centrale des noms d'équipes
│   ├── confederations.py       # Assignation équipe → confédération
│   ├── elo_fetch.py            # Elo ratings
│   ├── fifa_ranking.py         # Classement FIFA (repli Elo)
│   ├── results_fetch.py        # Résultats + buteurs (martj42)
│   ├── fixtures_fetch.py       # Fixtures CdM 2026 + résultats live
│   ├── players_fetch.py        # Buts par joueur (Soulier d'or)
│   └── competition_history.py  # Palmarès par équipe (jfjelstul)
├── model/
│   ├── elo.py                  # Formule Elo → probabilités
│   ├── strength_poisson.py     # Modèle de force unique + hiérarchie + bootstrap
│   ├── calibration.py          # Calibration isotonique (PAVA, sans sklearn)
│   └── poisson_dixoncoles.py   # (legacy, non utilisé — conservé pour référence)
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
Le modèle est sérialisé dans `data/strength_params.pkl` après chaque estimation.

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
