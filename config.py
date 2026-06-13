"""
Central configuration — seeds, paths, API keys, model hyperparameters.
Never hard-code the API key elsewhere; import FOOTBALL_DATA_API_KEY from here.
"""
import os
from pathlib import Path

# ── Reproductibilité ──────────────────────────────────────────────────────────
SEED = 42

# ── Répertoires ───────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent
DATA_DIR    = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"

DATA_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# Journal des pronostics GELÉS avant match (suivi/versionné, PAS dans .gitignore).
# Indispensable pour une notation honnête : on ne note que le pronostic figé
# AVANT le coup d'envoi, jamais un recalcul post-match (qui aurait vu le résultat).
PREDICTIONS_LOG = ROOT_DIR / "predictions_log.csv"

# ── Clés API ──────────────────────────────────────────────────────────────────
# Peut être surchargée via variable d'environnement FOOTBALL_DATA_API_KEY
FOOTBALL_DATA_API_KEY = os.getenv(
    "FOOTBALL_DATA_API_KEY",
    "483b2f1dd70c49b3b0b22335fb46937f",
)

# ── URLs sources de données ───────────────────────────────────────────────────
FOOTBALL_DATA_BASE  = "https://api.football-data.org/v4"
ELO_URL             = "https://www.eloratings.net/World.tsv"
# Dépôt martj42 — mis à jour mensuellement
RESULTS_URL         = "https://raw.githubusercontent.com/martj42/international-results/main/results.csv"
GOALSCORERS_URL     = "https://raw.githubusercontent.com/martj42/international-results/main/goalscorers.csv"
SHOOTOUTS_URL       = "https://raw.githubusercontent.com/martj42/international-results/main/shootouts.csv"

# ── Modèle de buts (force unique + Poisson) ──────────────────────────────────
HALF_LIFE_DAYS     = 547  # ~18 mois ; poids = exp(-ln2 * Δt / 547)
TRAIN_CUTOFF_YEARS = 8    # fenêtre d'entraînement (la pondération atténue le reste)
MAX_GOALS          = 10   # taille de la matrice de probabilité de scores
MIN_MATCHES_TEAM   = 3    # minimum de matchs pour estimer la force d'une équipe
RIDGE_TEAM         = 0.5  # shrinkage équipe->confédération (réglé par validation)
# ridge_conf : NON identifiable par la validation (dominée par l'intra-confédération
# où le niveau de conf s'annule). Fixé par prior de domaine — les confédérations
# diffèrent réellement (CONMEBOL/UEFA >> OFC). Validé via le backtest de tournoi.
RIDGE_CONF         = 1.0
N_BOOTSTRAP        = 40   # réplicas bootstrap (propagation d'incertitude au MC)
BOOTSTRAP_BLOCK    = "tournament"  # block-bootstrap : par (compétition x année)

# ── Avantage hôte CdM 2026 (USA / Mexique / Canada) ──────────────────────────
# Les 3 hôtes jouent réellement à domicile (public, voyages, altitude Mexico,
# chaleur) ; ~0.12 en log ≈ +13 % de buts attendus, ~½ de l'avantage domicile
# estimé par le modèle (~0.24). Appliqué uniquement aux matchs du Mondial.
HOST_TEAMS     = {"United States", "Mexico", "Canada"}
HOST_ADVANTAGE = 0.12

# ── Mise à jour quotidienne en tournoi (update_daily.py) ─────────────────────
XG_BLEND    = 0.5          # score effectif = XG_BLEND*xG + (1-XG_BLEND)*score réel
WC_K_FACTOR = 8.0          # poids majoré des matchs CdM en cours (forme actuelle)
WC_SINCE    = "2026-06-01" # date à partir de laquelle un match "World Cup" est en cours

# ── Monte-Carlo ───────────────────────────────────────────────────────────────
N_SIMULATIONS = 20_000

# ── Tournoi CdM 2026 ─────────────────────────────────────────────────────────
WC_GROUPS         = list("ABCDEFGHIJKL")  # 12 groupes de 4
N_THIRD_ADVANCE   = 8   # 8 meilleurs 3es qualifiés

# ── Compétitions majeures (filtre history) ────────────────────────────────────
MAJOR_TOURNAMENTS = [
    "FIFA World Cup",
    "UEFA Euro",
    "Copa América",
    "Africa Cup of Nations",
    "AFC Asian Cup",
    "CONCACAF Gold Cup",
    "FIFA Confederations Cup",
    "FIFA World Cup qualification",
]
