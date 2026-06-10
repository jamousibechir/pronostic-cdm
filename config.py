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

# ── Modèle Dixon-Coles ────────────────────────────────────────────────────────
HALF_LIFE_DAYS   = 547   # ~18 mois ; poids = exp(-ln2 * Δt / 547)
TRAIN_CUTOFF_YEARS = 6   # on utilise les 6 dernières années pour l'estimation
MAX_GOALS        = 10    # taille de la matrice de probabilité de scores
MIN_MATCHES_TEAM = 5     # minimum de matchs récents pour inclure une équipe

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
