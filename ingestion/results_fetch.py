"""
Télécharge et normalise les résultats historiques internationaux.

Source primaire : martj42/international_results (GitHub raw CSV)
  - results.csv     : ~49 500 matchs internationaux 1872–2026 (toutes compétitions)
  - goalscorers.csv : ~47 600 buts (buteur, minute, pénalty, csc)
  - shootouts.csv   : tirs au but
  Schéma results.csv déjà aligné : date, home_team, away_team, home_score,
  away_score, tournament, city, country, neutral.

  NOTE : le repo a été renommé `international-results` -> `international_results`
  (underscore) en 2024. L'ancienne URL avec tiret renvoie 404.

Fallback : jfjelstul/worldcup (WC 1930–2022) si martj42 est indisponible.

Tous les noms d'équipes passent par ingestion.names.canonical().
"""
import io
import requests
import pandas as pd

from config import DATA_DIR, TRAIN_CUTOFF_YEARS
from ingestion.names import canonical

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

RESULTS_OUT     = DATA_DIR / "results.csv"
GOALSCORERS_OUT = DATA_DIR / "goalscorers.csv"
SHOOTOUTS_OUT   = DATA_DIR / "shootouts.csv"

MARTJ42_BASE = "https://raw.githubusercontent.com/martj42/international_results/master"

# Fallback WC complet
JFJELSTUL_URL = "https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/matches.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Source primaire : martj42
# ─────────────────────────────────────────────────────────────────────────────

def _download_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def fetch_martj42_results() -> pd.DataFrame:
    """Tous les matchs internationaux 1872–présent depuis martj42."""
    df = _download_csv(f"{MARTJ42_BASE}/results.csv")
    df["date"]       = pd.to_datetime(df["date"], errors="coerce")
    df["home_team"]  = df["home_team"].map(canonical)
    df["away_team"]  = df["away_team"].map(canonical)
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["neutral"]    = df["neutral"].fillna(False).astype(bool)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Fallback : jfjelstul/worldcup
# ─────────────────────────────────────────────────────────────────────────────

def fetch_jfjelstul_wc() -> pd.DataFrame:
    """Matchs WC hommes 1930–2022 (fallback + détection des stades KO)."""
    df = _download_csv(JFJELSTUL_URL)
    mens = df[df["tournament_id"].str.match(r"WC-\d{4}$")].copy()
    mens = mens[~mens["tournament_name"].str.contains("Women", na=False, case=False)]
    out = pd.DataFrame({
        "date":       pd.to_datetime(mens["match_date"], errors="coerce"),
        "home_team":  mens["home_team_name"].map(canonical),
        "away_team":  mens["away_team_name"].map(canonical),
        "home_score": pd.to_numeric(mens["home_team_score"], errors="coerce"),
        "away_score": pd.to_numeric(mens["away_team_score"], errors="coerce"),
        "tournament": "FIFA World Cup",
        "stage_name": mens["stage_name"],
        "city":       mens["city_name"],
        "country":    mens["country_name"],
        "neutral":    True,
    })
    return out.dropna(subset=["date", "home_score", "away_score"])


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée principal
# ─────────────────────────────────────────────────────────────────────────────

def fetch_results(force: bool = False) -> pd.DataFrame:
    """
    Charge tous les résultats internationaux (martj42 en primaire).

    Returns
    -------
    pd.DataFrame : [date, home_team, away_team, home_score, away_score,
                    tournament, city, country, neutral]
    """
    if RESULTS_OUT.exists() and not force:
        df = pd.read_csv(RESULTS_OUT, low_memory=False)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    print("=" * 55)
    print("Collecte des resultats internationaux (martj42)")
    print("=" * 55)

    try:
        df = fetch_martj42_results()
        source = "martj42"
    except Exception as e:
        print(f"  martj42 indisponible ({e}), fallback jfjelstul WC...")
        df = fetch_jfjelstul_wc()
        source = "jfjelstul (fallback)"

    # Nettoyage : on ne garde que les matchs joués (scores non nuls)
    df = df.dropna(subset=["date", "home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = (df.sort_values("date")
            .drop_duplicates(subset=["date", "home_team", "away_team"])
            .reset_index(drop=True))

    df.to_csv(RESULTS_OUT, index=False)
    yr_min, yr_max = df["date"].dt.year.min(), df["date"].dt.year.max()
    print(f"\nResultats sauvegardes ({source}) : {len(df)} matchs "
          f"({yr_min}-{yr_max}) -> {RESULTS_OUT}")
    print("\nTop compétitions :")
    print(df.groupby("tournament").size().sort_values(ascending=False).head(12).to_string())
    return df


def fetch_goalscorers(force: bool = False) -> pd.DataFrame:
    """
    Buts par joueur depuis martj42 goalscorers.csv.
    Colonnes : date, home_team, away_team, team, scorer, minute, own_goal, penalty
    """
    if GOALSCORERS_OUT.exists() and not force:
        df = pd.read_csv(GOALSCORERS_OUT, low_memory=False)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    try:
        df = _download_csv(f"{MARTJ42_BASE}/goalscorers.csv")
        df["date"]      = pd.to_datetime(df["date"], errors="coerce")
        df["home_team"] = df["home_team"].map(canonical)
        df["away_team"] = df["away_team"].map(canonical)
        df["team"]      = df["team"].map(canonical)
        df["own_goal"]  = df["own_goal"].fillna(False).astype(bool)
        df["penalty"]   = df["penalty"].fillna(False).astype(bool)
        df = df.dropna(subset=["date", "scorer"])
        df.to_csv(GOALSCORERS_OUT, index=False)
        print(f"  Buteurs martj42 : {len(df)} buts -> {GOALSCORERS_OUT}")
        return df
    except Exception as e:
        print(f"  goalscorers.csv indisponible ({e})")
        df = pd.DataFrame(columns=["date", "home_team", "away_team", "team",
                                    "scorer", "minute", "own_goal", "penalty"])
        df.to_csv(GOALSCORERS_OUT, index=False)
        return df


def fetch_shootouts(force: bool = False) -> pd.DataFrame:
    """Résultats des séances de tirs au but (martj42)."""
    if SHOOTOUTS_OUT.exists() and not force:
        df = pd.read_csv(SHOOTOUTS_OUT, low_memory=False)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df
    try:
        df = _download_csv(f"{MARTJ42_BASE}/shootouts.csv")
        df["date"]      = pd.to_datetime(df["date"], errors="coerce")
        df["home_team"] = df["home_team"].map(canonical)
        df["away_team"] = df["away_team"].map(canonical)
        if "winner" in df.columns:
            df["winner"] = df["winner"].map(canonical)
        df.to_csv(SHOOTOUTS_OUT, index=False)
        return df
    except Exception:
        df = pd.DataFrame(columns=["date", "home_team", "away_team", "winner"])
        df.to_csv(SHOOTOUTS_OUT, index=False)
        return df


def recent_results(years: int = TRAIN_CUTOFF_YEARS,
                   force: bool = False) -> pd.DataFrame:
    """Résultats des `years` dernières années (pour l'estimation du modèle)."""
    df = fetch_results(force=force)
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
    return df[df["date"] >= cutoff].reset_index(drop=True)


if __name__ == "__main__":
    df = fetch_results(force=True)
    print(f"\nTotal : {len(df)} matchs | {df['date'].dt.year.min()}-{df['date'].dt.year.max()}")
    gs = fetch_goalscorers(force=True)
    print(f"Buteurs : {len(gs)} buts")
