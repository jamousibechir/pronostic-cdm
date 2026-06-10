"""
Télécharge et nettoie les résultats historiques internationaux.

Source : github.com/martj42/international-results
  - results.csv     : résultats de matchs (1872 → présent)
  - goalscorers.csv : buteurs par match
  - shootouts.csv   : résultats des tirs au but

Sorties : data/results.csv, data/goalscorers.csv, data/shootouts.csv
"""
import requests
import pandas as pd
from io import StringIO

from config import (
    DATA_DIR, RESULTS_URL, GOALSCORERS_URL,
    SHOOTOUTS_URL, TRAIN_CUTOFF_YEARS,
)

HEADERS = {"User-Agent": "pronostic-cdm/1.0 (research project)"}


def _download(url: str, dest: str, force: bool) -> pd.DataFrame:
    dest_path = DATA_DIR / dest
    if dest_path.exists() and not force:
        return pd.read_csv(dest_path, low_memory=False)
    print(f"Téléchargement : {url}")
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text), low_memory=False)
    df.to_csv(dest_path, index=False)
    print(f"  → {len(df)} lignes sauvegardées dans {dest_path}")
    return df


def fetch_results(force: bool = False) -> pd.DataFrame:
    """
    Retourne les résultats de matchs internationaux.

    Colonnes : date, home_team, away_team, home_score, away_score,
               tournament, city, country, neutral
    """
    df = _download(RESULTS_URL, "results.csv", force)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(bool)
    return df


def fetch_goalscorers(force: bool = False) -> pd.DataFrame:
    """
    Retourne les buteurs par match.

    Colonnes : date, home_team, away_team, team, scorer, own_goal, penalty
    """
    df = _download(GOALSCORERS_URL, "goalscorers.csv", force)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["own_goal"] = df["own_goal"].fillna(False).astype(bool)
    df["penalty"]  = df["penalty"].fillna(False).astype(bool)
    return df


def fetch_shootouts(force: bool = False) -> pd.DataFrame:
    """
    Retourne les résultats de tirs au but.

    Colonnes : date, home_team, away_team, winner, first_shooter
    """
    df = _download(SHOOTOUTS_URL, "shootouts.csv", force)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def recent_results(years: int = TRAIN_CUTOFF_YEARS,
                   force: bool = False) -> pd.DataFrame:
    """
    Résultats des `years` dernières années, toutes compétitions confondues.
    Utilisé pour l'estimation Dixon-Coles.
    """
    df = fetch_results(force=force)
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
    return df[df["date"] >= cutoff].reset_index(drop=True)


if __name__ == "__main__":
    r  = fetch_results(force=True)
    gs = fetch_goalscorers(force=True)
    so = fetch_shootouts(force=True)
    print(f"Résultats : {len(r)} matchs | Buteurs : {len(gs)} | Tirs au but : {len(so)}")
    print(recent_results().tail())
