"""
Télécharge et normalise les résultats historiques internationaux.

Sources confirmées opérationnelles :
  1. jfjelstul/worldcup (GitHub raw CSV)
     → Tous les matchs WC hommes 1930–2022 (1248 matchs)
     → URL: https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/matches.csv

  2. football-data.org v4 API (clé gratuite)
     → EC 2024 (51 matchs)

  3. ESPN API non officielle (sans clé, rate-limit non documenté)
     → WC Qualifications (UEFA, CONMEBOL, AFC, CAF, CONCACAF)
     → UEFA Nations League, AFCON, Asian Cup, Gold Cup (2020–2025)

Schéma de sortie unifié :
  date, home_team, away_team, home_score, away_score,
  tournament, city, country, neutral
"""
import time
import io
import requests
import pandas as pd

from config import (
    DATA_DIR, FOOTBALL_DATA_BASE, FOOTBALL_DATA_API_KEY,
    TRAIN_CUTOFF_YEARS,
)

HEADERS_API  = {"X-Auth-Token": FOOTBALL_DATA_API_KEY, "User-Agent": "pronostic-cdm/1.0"}
HEADERS_ESPN = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json",
}

RESULTS_OUT     = DATA_DIR / "results.csv"
GOALSCORERS_OUT = DATA_DIR / "goalscorers.csv"
SHOOTOUTS_OUT   = DATA_DIR / "shootouts.csv"

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"

# ── ESPN : compétitions + plages de dates à scraper ───────────────────────────
ESPN_COMPETITIONS = [
    # (league_code, tournament_name, date_start, date_end)
    ("FIFA.WORLDQ.UEFA",      "FIFA World Cup qualification (UEFA)",     "20231012", "20251115"),
    ("FIFA.WORLDQ.CONMEBOL",  "FIFA World Cup qualification (CONMEBOL)", "20231108", "20251119"),
    ("FIFA.WORLDQ.AFC",       "FIFA World Cup qualification (AFC)",      "20231112", "20251119"),
    ("FIFA.WORLDQ.CAF",       "FIFA World Cup qualification (CAF)",      "20231113", "20251118"),
    ("FIFA.WORLDQ.CONCACAF",  "FIFA World Cup qualification (CONCACAF)", "20240306", "20251119"),
    ("FIFA.WORLDQ.OFC",       "FIFA World Cup qualification (OFC)",      "20231107", "20241124"),
    ("UEFA.NATIONS",          "UEFA Nations League",                     "20200905", "20250325"),
    ("CAF.NATIONS",           "Africa Cup of Nations",                   "20210109", "20250209"),
    ("AFC.ASIAN.CUP",         "AFC Asian Cup",                           "20190105", "20240210"),
    ("CONCACAF.GOLD",         "CONCACAF Gold Cup",                       "20190615", "20230717"),
    ("UEFA.EURO",             "UEFA European Championship",              "20210611", "20240714"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Source 1 : jfjelstul/worldcup
# ─────────────────────────────────────────────────────────────────────────────

JFJELSTUL_URL = "https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/matches.csv"


def fetch_jfjelstul_wc() -> pd.DataFrame:
    """Charge tous les matchs WC hommes 1930–2022 depuis jfjelstul/worldcup."""
    r = requests.get(JFJELSTUL_URL,
                     headers={"User-Agent": "pronostic-cdm/1.0"},
                     timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))

    # Filtre : hommes uniquement (exclut WC-2011, WC-2015, WC-2019 = Women's)
    mens = df[df["tournament_id"].str.match(r"WC-\d{4}$")].copy()
    # Note : WC-2011, WC-2019 etc. sont les Women's WC — exclus par le filtre
    # (les Women's WC ont des IDs différents dans ce dataset)
    mens = mens[~mens["tournament_name"].str.contains("Women", na=False, case=False)]

    out = pd.DataFrame({
        "date":       pd.to_datetime(mens["match_date"], errors="coerce"),
        "home_team":  mens["home_team_name"].str.strip(),
        "away_team":  mens["away_team_name"].str.strip(),
        "home_score": pd.to_numeric(mens["home_team_score"], errors="coerce"),
        "away_score": pd.to_numeric(mens["away_team_score"], errors="coerce"),
        "tournament": mens["tournament_name"],
        "city":       mens["city_name"],
        "country":    mens["country_name"],
        "neutral":    True,   # tous les WC = terrain neutre
    })
    return out.dropna(subset=["date", "home_score", "away_score"])


# ─────────────────────────────────────────────────────────────────────────────
# Source 2 : football-data.org
# ─────────────────────────────────────────────────────────────────────────────

def _fd_get(endpoint: str) -> dict:
    url = f"{FOOTBALL_DATA_BASE}/{endpoint}"
    for _ in range(3):
        r = requests.get(url, headers=HEADERS_API, timeout=20)
        if r.status_code == 429:
            wait = int(r.headers.get("X-RequestCounter-Reset", 65))
            print(f"    Rate-limit football-data.org, attente {wait}s...")
            time.sleep(wait)
            continue
        if r.status_code in (403, 404):
            return {}
        r.raise_for_status()
        return r.json()
    return {}


def fetch_fd_ec2024() -> pd.DataFrame:
    """Récupère les 51 matchs EC 2024 via football-data.org."""
    data = _fd_get("competitions/EC/matches?season=2024")
    rows = []
    for m in data.get("matches", []):
        if m.get("status") != "FINISHED":
            continue
        ht = m["homeTeam"].get("name", "")
        at = m["awayTeam"].get("name", "")
        ft = m.get("score", {}).get("fullTime", {})
        hs, as_ = ft.get("home"), ft.get("away")
        if hs is None or as_ is None:
            continue
        rows.append({
            "date":       pd.to_datetime(m.get("utcDate", "")[:10], errors="coerce"),
            "home_team":  ht,
            "away_team":  at,
            "home_score": int(hs),
            "away_score": int(as_),
            "tournament": "UEFA Euro",
            "city":       "",
            "country":    "",
            "neutral":    True,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Source 3 : ESPN API non officielle
# ─────────────────────────────────────────────────────────────────────────────

def _espn_fetch(league: str, date_start: str, date_end: str,
                tournament: str) -> list[dict]:
    """
    Récupère les résultats d'une compétition ESPN sur une plage de dates.
    Itère par mois pour éviter de dépasser les limites non documentées.
    """
    from datetime import datetime, timedelta

    results = []
    # Génère les 1ers de chaque mois dans la plage
    dt_start = datetime.strptime(date_start, "%Y%m%d")
    dt_end   = datetime.strptime(date_end,   "%Y%m%d")

    current = dt_start.replace(day=1)
    seen    = set()

    while current <= dt_end:
        next_month = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
        end_of_month = min(next_month - timedelta(days=1), dt_end)

        date_param = (f"{current.strftime('%Y%m%d')}"
                      f"-{end_of_month.strftime('%Y%m%d')}")
        url = f"{ESPN_BASE}/{league}/scoreboard?dates={date_param}&limit=100"

        try:
            time.sleep(0.8)
            r = requests.get(url, headers=HEADERS_ESPN, timeout=15)
            if not r.ok:
                current = next_month
                continue
            events = r.json().get("events", [])
            for e in events:
                comps = e.get("competitions", [{}])
                c     = comps[0] if comps else {}
                status_name = (c.get("status", {})
                               .get("type", {})
                               .get("name", ""))
                if "STATUS_FULL_TIME" not in status_name and "STATUS_FT" not in status_name and "FULL" not in status_name:
                    continue

                teams  = c.get("competitors", [])
                if len(teams) < 2:
                    continue

                # ESPN place parfois home/away dans un ordre variable
                home_t = next((t for t in teams if t.get("homeAway") == "home"), teams[0])
                away_t = next((t for t in teams if t.get("homeAway") == "away"), teams[1])

                ht = home_t.get("team", {}).get("displayName", "")
                at = away_t.get("team", {}).get("displayName", "")
                try:
                    hs = int(float(home_t.get("score", "0") or "0"))
                    as_ = int(float(away_t.get("score", "0") or "0"))
                except (ValueError, TypeError):
                    continue

                date_str = e.get("date", "")[:10]
                key = (date_str, ht, at)
                if key in seen:
                    continue
                seen.add(key)

                results.append({
                    "date":       pd.to_datetime(date_str, errors="coerce"),
                    "home_team":  ht,
                    "away_team":  at,
                    "home_score": hs,
                    "away_score": as_,
                    "tournament": tournament,
                    "city":       "",
                    "country":    "",
                    "neutral":    bool(c.get("neutralSite", False)),
                })
        except Exception as ex:
            pass   # Silencieux sur erreur réseau individuelle

        current = next_month

    return results


def fetch_espn_competitions() -> pd.DataFrame:
    """Scrape toutes les compétitions ESPN listées dans ESPN_COMPETITIONS."""
    all_rows = []
    for league, tournament, d_start, d_end in ESPN_COMPETITIONS:
        print(f"  ESPN : {tournament} ({d_start[:4]}–{d_end[:4]})...")
        rows = _espn_fetch(league, d_start, d_end, tournament)
        print(f"    → {len(rows)} matchs")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    return df.dropna(subset=["date"]) if not df.empty else df


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée principal
# ─────────────────────────────────────────────────────────────────────────────

def fetch_results(force: bool = False) -> pd.DataFrame:
    """
    Charge les résultats depuis toutes les sources disponibles.

    Returns
    -------
    pd.DataFrame : colonnes [date, home_team, away_team, home_score, away_score,
                              tournament, city, country, neutral]
    """
    if RESULTS_OUT.exists() and not force:
        df = pd.read_csv(RESULTS_OUT, low_memory=False)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    print("=" * 55)
    print("Collecte des résultats historiques internationaux")
    print("=" * 55)

    frames = []

    # Source 1 : WC historique complet (1930–2022)
    print("\n[1/3] jfjelstul/worldcup (WC 1930–2022)...")
    try:
        df_wc = fetch_jfjelstul_wc()
        print(f"  → {len(df_wc)} matchs WC ({df_wc['date'].dt.year.min()}–{df_wc['date'].dt.year.max()})")
        frames.append(df_wc)
    except Exception as e:
        print(f"  ERREUR : {e}")

    # Source 2 : EC 2024 via football-data.org
    print("\n[2/3] football-data.org (EC 2024)...")
    try:
        df_ec = fetch_fd_ec2024()
        print(f"  → {len(df_ec)} matchs EC 2024")
        frames.append(df_ec)
    except Exception as e:
        print(f"  ERREUR : {e}")
    time.sleep(7)

    # Source 3 : ESPN — qualifs, Nations League, AFCON, Asian Cup, Gold Cup
    print("\n[3/3] ESPN API (qualifications + compétitions 2019–2025)...")
    try:
        df_espn = fetch_espn_competitions()
        print(f"  → {len(df_espn)} matchs ESPN")
        frames.append(df_espn)
    except Exception as e:
        print(f"  ERREUR ESPN : {e}")

    if not frames:
        raise RuntimeError("Aucune donnée récupérée. Vérifiez la connexion réseau.")

    # ── Fusion et nettoyage ────────────────────────────────────────────────
    df = pd.concat(frames, ignore_index=True)
    df["date"]       = pd.to_datetime(df["date"], errors="coerce")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["neutral"]    = df["neutral"].fillna(False).astype(bool)

    df = df.dropna(subset=["date", "home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Déduplication sur (date, home_team, away_team)
    df = (df.sort_values("date")
            .drop_duplicates(subset=["date", "home_team", "away_team"])
            .reset_index(drop=True))

    df.to_csv(RESULTS_OUT, index=False)
    print(f"\nResultats sauvegardes : {len(df)} matchs -> {RESULTS_OUT}")
    print(df.groupby("tournament").size().sort_values(ascending=False).head(15).to_string())
    return df


def fetch_goalscorers(force: bool = False) -> pd.DataFrame:
    """Charge les buteurs. Retourne schéma vide si source indisponible."""
    if GOALSCORERS_OUT.exists() and not force:
        df = pd.read_csv(GOALSCORERS_OUT, low_memory=False)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df
    df = pd.DataFrame(columns=[
        "date", "home_team", "away_team", "team", "scorer", "own_goal", "penalty"
    ])
    df.to_csv(GOALSCORERS_OUT, index=False)
    return df


def fetch_shootouts(force: bool = False) -> pd.DataFrame:
    """Charge les résultats de tirs au but."""
    if SHOOTOUTS_OUT.exists() and not force:
        df = pd.read_csv(SHOOTOUTS_OUT, low_memory=False)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df
    df = pd.DataFrame(columns=["date", "home_team", "away_team", "winner"])
    df.to_csv(SHOOTOUTS_OUT, index=False)
    return df


def recent_results(years: int = TRAIN_CUTOFF_YEARS,
                   force: bool = False) -> pd.DataFrame:
    """Résultats des `years` dernières années."""
    df = fetch_results(force=force)
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
    return df[df["date"] >= cutoff].reset_index(drop=True)


if __name__ == "__main__":
    df = fetch_results(force=True)
    print(f"\nTotal : {len(df)} matchs | Années : {df['date'].dt.year.min()}–{df['date'].dt.year.max()}")
