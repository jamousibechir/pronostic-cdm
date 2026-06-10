"""
Récupère les fixtures de la Coupe du Monde 2026 via football-data.org v4.

Endpoint : GET /v4/competitions/WC/matches
Clé API  : FOOTBALL_DATA_API_KEY (config.py)
Free tier : 10 req/min, pas de données de buts en direct en tier gratuit.

Sorties : data/fixtures.csv, data/wc2026_teams.csv
Colonnes fixtures :
  match_id, stage, group, matchday, utc_date,
  home_team, away_team, home_score, away_score, status
"""
import time
import requests
import pandas as pd

from config import DATA_DIR, FOOTBALL_DATA_BASE, FOOTBALL_DATA_API_KEY

HEADERS = {
    "X-Auth-Token": FOOTBALL_DATA_API_KEY,
    "User-Agent":   "pronostic-cdm/1.0",
}
FIXTURES_OUT = DATA_DIR / "fixtures.csv"
TEAMS_OUT    = DATA_DIR / "wc2026_teams.csv"

STAGE_ORDER = {
    "GROUP_STAGE":    0,
    "ROUND_OF_32":    1,
    "ROUND_OF_16":    2,
    "QUARTER_FINALS": 3,
    "SEMI_FINALS":    4,
    "THIRD_PLACE":    5,
    "FINAL":          6,
}


def _get(endpoint: str, params: dict | None = None) -> dict:
    """Effectue un GET avec gestion du rate-limit (429)."""
    url = f"{FOOTBALL_DATA_BASE}/{endpoint}"
    for attempt in range(3):
        r = requests.get(url, headers=HEADERS, params=params, timeout=20)
        if r.status_code == 429:
            wait = int(r.headers.get("X-RequestCounter-Reset", 60))
            print(f"Rate-limit atteint, attente {wait}s...")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Échec après 3 tentatives : {url}")


def _parse_matches(data: dict) -> pd.DataFrame:
    rows = []
    for m in data.get("matches", []):
        home = m["homeTeam"].get("name") or m["homeTeam"].get("shortName")
        away = m["awayTeam"].get("name") or m["awayTeam"].get("shortName")
        score = m.get("score", {})
        full  = score.get("fullTime", {})
        rows.append({
            "match_id":   m["id"],
            "stage":      m.get("stage", ""),
            "group":      m.get("group", ""),
            "matchday":   m.get("matchday"),
            "utc_date":   m.get("utcDate"),
            "home_team":  home,
            "away_team":  away,
            "home_score": full.get("home"),   # None si pas encore joué
            "away_score": full.get("away"),
            "status":     m.get("status", "SCHEDULED"),
        })
    df = pd.DataFrame(rows)
    df["utc_date"] = pd.to_datetime(df["utc_date"], errors="coerce")
    df["stage_order"] = df["stage"].map(STAGE_ORDER).fillna(-1).astype(int)
    return df.sort_values(["stage_order", "matchday", "utc_date"]).reset_index(drop=True)


def _parse_teams(data: dict) -> pd.DataFrame:
    rows = []
    for t in data.get("teams", []):
        rows.append({
            "team_id": t["id"],
            "team":    t["name"],
            "tla":     t.get("tla", ""),
            "group":   t.get("group", ""),
        })
    return pd.DataFrame(rows)


def fetch_fixtures(force: bool = False) -> pd.DataFrame:
    """
    Charge les fixtures WC 2026. Utilise le cache si force=False.

    En mode mise à jour quotidienne, relance avec force=True pour
    figer les scores des matchs terminés.

    Returns
    -------
    pd.DataFrame : colonnes décrites dans le header du module
    """
    if FIXTURES_OUT.exists() and not force:
        df = pd.read_csv(FIXTURES_OUT)
        df["utc_date"] = pd.to_datetime(df["utc_date"], errors="coerce")
        return df

    data = _get("competitions/WC/matches")
    df = _parse_matches(data)
    df.to_csv(FIXTURES_OUT, index=False)
    print(f"Fixtures sauvegardées : {len(df)} matchs → {FIXTURES_OUT}")
    return df


def fetch_wc_teams(force: bool = False) -> pd.DataFrame:
    """
    Charge la liste des équipes WC 2026 avec leur groupe.
    """
    if TEAMS_OUT.exists() and not force:
        return pd.read_csv(TEAMS_OUT)

    data = _get("competitions/WC/teams")
    df = _parse_teams(data)
    df.to_csv(TEAMS_OUT, index=False)
    print(f"Équipes WC 2026 sauvegardées : {len(df)} équipes → {TEAMS_OUT}")
    return df


def update_live_results() -> pd.DataFrame:
    """
    Met à jour uniquement les matchs déjà joués (status=FINISHED).
    Conserve les prédictions pour les matchs à venir.
    Appelé quotidiennement pendant le tournoi.
    """
    print("Mise à jour des résultats en direct...")
    fresh = fetch_fixtures(force=True)
    print(f"  Matchs terminés : {(fresh['status'] == 'FINISHED').sum()}")
    print(f"  Matchs à venir  : {(fresh['status'] == 'SCHEDULED').sum()}")
    return fresh


if __name__ == "__main__":
    teams    = fetch_wc_teams(force=True)
    fixtures = fetch_fixtures(force=True)
    print(teams.head(20))
    print(fixtures.head(10))
