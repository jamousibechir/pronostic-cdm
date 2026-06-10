"""
Construit la table des buteurs en sélection pour la simulation Soulier d'or.

Sources (par priorité) :
  1. goalscorers.csv (martj42) — données jusqu'à la date de téléchargement
  2. FBref.com — pages statistiques d'équipes nationales (scrape respectueux)
  3. Transfermarkt — fallback si FBref est surchargé / bloqué

Hypothèse : seuls les joueurs avec >= 1 but en sélection depuis 2022 sont inclus.
La probabilité de partir au Mondial est approximée via le nombre de sélections récentes.

Sortie : data/players.csv
Colonnes : player, team, goals_total, goals_recent, caps_recent,
           goal_rate, pen_goals, is_pen_taker, start_prob
"""
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

from config import DATA_DIR
from ingestion.results_fetch import fetch_goalscorers

HEADERS = {"User-Agent": "pronostic-cdm/1.0 (research project)"}
OUTPUT  = DATA_DIR / "players.csv"

RECENT_CUTOFF = "2022-01-01"
FBREF_BASE    = "https://fbref.com"
TM_BASE       = "https://www.transfermarkt.com"

# Délai entre requêtes FBref pour respecter le rate-limit
FBREF_DELAY = 2.5  # secondes


# ── Construction depuis goalscorers.csv ───────────────────────────────────────

def _build_from_goalscorers(wc_teams: list[str]) -> pd.DataFrame:
    """
    Calcule les statistiques de buts par joueur à partir de goalscorers.csv.
    Si le fichier est vide (source martj42 indisponible), retourne un DataFrame vide
    et l'enrichissement FBref prendra le relais.
    """
    gs = fetch_goalscorers()

    if gs.empty:
        return pd.DataFrame(columns=[
            "player", "team", "goals_total", "goals_recent",
            "caps_recent", "goal_rate", "pen_goals", "is_pen_taker", "start_prob"
        ])

    # Filtre : exclut les csc, garde seulement les équipes qualifiées
    gs = gs[~gs["own_goal"]]
    if wc_teams:
        gs = gs[gs["team"].isin(wc_teams)]

    gs_recent = gs[gs["date"] >= RECENT_CUTOFF]

    total = (
        gs.groupby(["team", "scorer"])
        .size()
        .reset_index(name="goals_total")
    )

    recent = (
        gs_recent.groupby(["team", "scorer"])
        .agg(goals_recent=("scorer", "count"),
             pen_goals=("penalty", "sum"))
        .reset_index()
    )

    caps = (
        gs_recent.groupby(["team", "scorer"])["date"]
        .nunique()
        .reset_index(name="caps_recent")
    )

    df = total.merge(recent, on=["team", "scorer"], how="outer")
    df = df.merge(caps,   on=["team", "scorer"], how="left")
    df = df.rename(columns={"scorer": "player"})
    df = df.fillna(0)

    df["goal_rate"]    = df["goals_recent"] / df["caps_recent"].clip(lower=1)
    df["is_pen_taker"] = (df["pen_goals"] >= 2).astype(int)

    df["start_prob"] = (df["caps_recent"] / df["caps_recent"].max()).clip(upper=1.0)
    df["start_prob"] = 0.5 + 0.35 * df["start_prob"]

    return df[["player", "team", "goals_total", "goals_recent",
               "caps_recent", "goal_rate", "pen_goals",
               "is_pen_taker", "start_prob"]]


# ── Enrichissement FBref (caps actuels, titulaires probables) ─────────────────

# Mapping nom FIFA → identifiant FBref (les plus importants)
FBREF_TEAM_IDS = {
    "France":        ("040792ee", "France"),
    "Brazil":        ("e8d9f1da", "Brazil"),
    "Argentina":     ("f9fddd6e", "Argentina"),
    "Spain":         ("7c07c67e", "Spain"),
    "England":       ("26f300ef", "England"),
    "Germany":       ("e020d6ae", "Germany"),
    "Portugal":      ("34630005", "Portugal"),
    "Netherlands":   ("5a7c2241", "Netherlands"),
    "Belgium":       ("9f9f6718", "Belgium"),
    "Morocco":       ("e2d58d5d", "Morocco"),
    "USA":           ("7fb87a9f", "United-States"),
    "Mexico":        ("5410d07d", "Mexico"),
    "Japan":         ("a3d88bd8", "Japan"),
    "Senegal":       ("fb466c73", "Senegal"),
    "Croatia":       ("dd9f15bc", "Croatia"),
    "Uruguay":       ("3fd3bbe4", "Uruguay"),
    "Colombia":      ("04c8e8cf", "Colombia"),
    "Ecuador":       ("c3d90f8b", "Ecuador"),
    "Saudi Arabia":  ("e3c54ee5", "Saudi-Arabia"),
    "Australia":     ("63c1cf00", "Australia"),
}


def _fbref_team_stats(team_name: str, team_id: str,
                       fbref_name: str) -> pd.DataFrame | None:
    """Scrape la page stats d'une équipe nationale sur FBref."""
    url = f"{FBREF_BASE}/en/squads/{team_id}/2025-2026/{fbref_name}-Stats"
    try:
        time.sleep(FBREF_DELAY)
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        # Table principale des stats standard
        table = soup.find("table", {"id": "stats_standard"})
        if table is None:
            return None
        df = pd.read_html(str(table))[0]
        # Multiindex header : on garde le niveau bas
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [" ".join(c).strip() for c in df.columns]
        # Standardise les colonnes
        rename = {}
        for col in df.columns:
            cl = col.lower()
            if "player" in cl:      rename[col] = "player"
            elif "goals" in cl and "pk" not in cl and rename.get(col) is None:
                rename[col] = "fbref_goals"
            elif "caps" in cl or "mp" == cl.lower() or "mp" in cl.lower():
                rename[col] = "fbref_caps"
        df = df.rename(columns=rename)
        if "player" not in df.columns:
            return None
        df["team"] = team_name
        keep = ["player", "team"] + [c for c in ["fbref_goals", "fbref_caps"]
                                       if c in df.columns]
        return df[keep].dropna(subset=["player"])
    except Exception as e:
        print(f"  FBref scrape échoué pour {team_name}: {e}")
        return None


def _enrich_with_fbref(df: pd.DataFrame,
                        wc_teams: list[str]) -> pd.DataFrame:
    """Enrichit les données goalscorers avec les caps FBref pour les équipes connues."""
    enriched_frames = []
    for team_name in wc_teams:
        if team_name not in FBREF_TEAM_IDS:
            continue
        tid, fbref_name = FBREF_TEAM_IDS[team_name]
        fb = _fbref_team_stats(team_name, tid, fbref_name)
        if fb is not None:
            enriched_frames.append(fb)

    if not enriched_frames:
        return df

    fbref_df = pd.concat(enriched_frames, ignore_index=True)
    # Merge pour mettre à jour les caps
    merged = df.merge(
        fbref_df[["player", "team", "fbref_caps"]],
        on=["player", "team"],
        how="left",
    )
    mask = merged["fbref_caps"].notna()
    merged.loc[mask, "caps_recent"] = merged.loc[mask, "fbref_caps"]
    merged.loc[mask, "start_prob"] = (
        0.5 + 0.35 * (merged.loc[mask, "caps_recent"] /
                      merged["caps_recent"].max()).clip(upper=1.0)
    )
    return merged.drop(columns=["fbref_caps"])


# ── Point d'entrée ────────────────────────────────────────────────────────────

def _build_from_fbref_only(wc_teams: list[str]) -> pd.DataFrame:
    """
    Construit la table joueurs entièrement depuis FBref quand goalscorers.csv
    est indisponible. Scrape toutes les équipes connues dans FBREF_TEAM_IDS.
    """
    frames = []
    teams_to_scrape = [t for t in wc_teams if t in FBREF_TEAM_IDS]
    for team_name in teams_to_scrape:
        tid, fbref_name = FBREF_TEAM_IDS[team_name]
        fb = _fbref_team_stats(team_name, tid, fbref_name)
        if fb is not None:
            frames.append(fb)

    if not frames:
        return pd.DataFrame(columns=[
            "player", "team", "goals_total", "goals_recent",
            "caps_recent", "goal_rate", "pen_goals", "is_pen_taker", "start_prob"
        ])

    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"fbref_goals": "goals_recent",
                             "fbref_caps":  "caps_recent"})
    df["goals_total"]  = df.get("goals_recent", 0)
    df["goal_rate"]    = (
        pd.to_numeric(df.get("goals_recent", 0), errors="coerce").fillna(0) /
        pd.to_numeric(df.get("caps_recent",  1), errors="coerce").clip(lower=1).fillna(1)
    )
    df["pen_goals"]    = 0
    df["is_pen_taker"] = 0
    caps_max = pd.to_numeric(df.get("caps_recent", 1), errors="coerce").max()
    df["start_prob"]   = (
        0.5 + 0.35 * (
            pd.to_numeric(df.get("caps_recent", 1), errors="coerce")
            .fillna(0) / max(caps_max, 1)
        ).clip(upper=1.0)
    )
    cols = ["player", "team", "goals_total", "goals_recent",
            "caps_recent", "goal_rate", "pen_goals", "is_pen_taker", "start_prob"]
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    return df[cols].dropna(subset=["player"])


def fetch_players(wc_teams: list[str] | None = None,
                  enrich_fbref: bool = True,
                  force: bool = False) -> pd.DataFrame:
    """
    Charge les statistiques joueurs pour la simulation Soulier d'or.

    Parameters
    ----------
    wc_teams    : liste des noms d'équipes qualifiées (filtre)
    enrich_fbref: active l'enrichissement FBref (plus lent)
    force       : recharge même si le cache existe
    """
    if OUTPUT.exists() and not force:
        return pd.read_csv(OUTPUT)

    wc_list = wc_teams or []
    print("Construction des stats joueurs depuis goalscorers.csv...")
    df = _build_from_goalscorers(wc_list)

    if df.empty and enrich_fbref and wc_list:
        # goalscorers.csv indisponible → construit entièrement depuis FBref
        print("goalscorers.csv vide — construction depuis FBref...")
        df = _build_from_fbref_only(wc_list)
    elif enrich_fbref and wc_list:
        print("Enrichissement FBref (peut prendre quelques minutes)...")
        df = _enrich_with_fbref(df, wc_list)

    # Filtre final : au moins 1 but récent ou taux > 0
    df["goals_recent"] = pd.to_numeric(df.get("goals_recent", 0), errors="coerce").fillna(0)
    df = df[df["goals_recent"] >= 1].copy()
    df = df.sort_values(["team", "goal_rate"], ascending=[True, False])
    df.to_csv(OUTPUT, index=False)
    print(f"Données joueurs sauvegardées : {len(df)} joueurs → {OUTPUT}")
    return df


if __name__ == "__main__":
    df = fetch_players(force=True, enrich_fbref=False)
    print(df.sort_values("goal_rate", ascending=False).head(30))
