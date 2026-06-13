"""
Récupère le classement FIFA officiel.

Sources tentées (par ordre) :
  1. API officielle FIFA (endpoint Next.js interne — change à chaque déploiement)
  2. ESPN API non officielle (via /apis/site/v2/sports/soccer/rankings)
  3. Fallback : dérive un pseudo-classement depuis les Elo ratings

En cas d'échec total, retourne un DataFrame vide plutôt que de planter —
le classement FIFA n'est qu'une feature de validation ; les Elo font le vrai travail.

Sortie : data/fifa_rankings.csv
  Colonnes : team, fifa_rank, fifa_points
"""
import re
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup

from config import DATA_DIR

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "en-US,en;q=0.9",
}
OUTPUT = DATA_DIR / "fifa_rankings.csv"

# ── Tentative 1 : Endpoint JSON embarqué dans la page FIFA ───────────────────

def _try_fifa_nextjs() -> pd.DataFrame | None:
    """Cherche les données de classement dans le bundle Next.js de fifa.com."""
    try:
        r = requests.get("https://www.fifa.com/fifa-world-ranking/men",
                         headers=HEADERS, timeout=15)
        if not r.ok:
            return None
        # Les données sont souvent dans <script id="__NEXT_DATA__">
        soup = BeautifulSoup(r.text, "lxml")
        tag = soup.find("script", {"id": "__NEXT_DATA__"})
        if not tag or not tag.string:
            return None
        data = json.loads(tag.string)

        # Cherche récursivement une liste de rankings
        def _find_rankings(obj):
            if isinstance(obj, list) and len(obj) > 10:
                if isinstance(obj[0], dict) and any(
                    k in obj[0] for k in ("rank", "ranking", "totalPoints", "points")
                ):
                    return obj
            if isinstance(obj, dict):
                for v in obj.values():
                    result = _find_rankings(v)
                    if result:
                        return result
            if isinstance(obj, list):
                for item in obj:
                    result = _find_rankings(item)
                    if result:
                        return result
            return None

        rankings = _find_rankings(data)
        if not rankings:
            return None

        rows = []
        for item in rankings:
            name   = (item.get("countryName") or item.get("name")
                      or item.get("team", {}).get("name", "") if isinstance(item.get("team"), dict) else "")
            rank   = item.get("rank") or item.get("ranking") or 0
            points = item.get("totalPoints") or item.get("points") or 0
            if name:
                rows.append({"team": name, "fifa_rank": rank, "fifa_points": points})

        return pd.DataFrame(rows) if rows else None
    except Exception:
        return None


# ── Tentative 2 : API ESPN pour les rankings soccer ──────────────────────────

ESPN_RANKING_URLS = [
    "https://site.api.espn.com/apis/site/v2/sports/soccer/FIFA.WC/rankings?limit=200",
    "https://site.api.espn.com/apis/site/v2/sports/soccer/rankings?limit=200",
    "https://sports.core.api.espn.com/v2/sports/soccer/leagues/FIFA.WC/seasons/2026/rankings",
]

def _try_espn() -> pd.DataFrame | None:
    for url in ESPN_RANKING_URLS:
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            if not r.ok:
                continue
            data = r.json()
            # ESPN rankings structure varie selon l'endpoint
            rows = []
            items = (data.get("rankings", [{}])[0].get("ranks", [])
                     if data.get("rankings") else
                     data.get("teams", []))
            for i, item in enumerate(items):
                team = (item.get("team", {}).get("displayName", "")
                        or item.get("displayName", ""))
                pts  = item.get("points", 0) or item.get("value", 0)
                if team:
                    rows.append({"team": team, "fifa_rank": i + 1, "fifa_points": pts})
            if rows:
                return pd.DataFrame(rows)
        except Exception:
            continue
    return None


# ── Fallback : pseudo-classement depuis les Elo ───────────────────────────────

def _fallback_from_elo() -> pd.DataFrame:
    """Dérive un classement indicatif depuis les Elo ratings (déjà téléchargés)."""
    elo_path = DATA_DIR / "elo_ratings.csv"
    if not elo_path.exists():
        return pd.DataFrame(columns=["team", "fifa_rank", "fifa_points"])
    elo = pd.read_csv(elo_path)
    elo = elo.sort_values("elo", ascending=False).reset_index(drop=True)
    elo["fifa_rank"]   = elo.index + 1
    elo["fifa_points"] = elo["elo"].round(0).astype(int)
    return elo[["team", "fifa_rank", "fifa_points"]]


# ── Point d'entrée ────────────────────────────────────────────────────────────

def fetch_fifa_rankings(force: bool = False) -> pd.DataFrame:
    """
    Charge le classement FIFA. Utilise le cache si disponible et force=False.
    Ne lève jamais d'exception — retourne un DataFrame vide en dernier recours.

    Returns
    -------
    pd.DataFrame : colonnes [team, fifa_rank, fifa_points]
    """
    if OUTPUT.exists() and not force:
        return pd.read_csv(OUTPUT)

    print("  Tentative 1 : FIFA Next.js...")
    df = _try_fifa_nextjs()
    if df is not None and not df.empty:
        source = "FIFA Next.js"
    else:
        print("  Tentative 2 : ESPN rankings...")
        df = _try_espn()
        if df is not None and not df.empty:
            source = "ESPN"
        else:
            print("  Fallback    : pseudo-classement depuis Elo ratings")
            df = _fallback_from_elo()
            source = "Elo fallback"

    if df is None or df.empty:
        print("  Avertissement : classement FIFA indisponible (non bloquant)")
        df = pd.DataFrame(columns=["team", "fifa_rank", "fifa_points"])
    else:
        df = df.sort_values("fifa_rank").reset_index(drop=True)
        df.to_csv(OUTPUT, index=False)
        print(f"  Classement FIFA sauvegarde ({source}) : {len(df)} equipes -> {OUTPUT}")

    return df


if __name__ == "__main__":
    df = fetch_fifa_rankings(force=True)
    print(df.head(20).to_string(index=False))
