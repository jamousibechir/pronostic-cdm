"""
Construit la table des buteurs en sélection pour la simulation Soulier d'or.

Source : martj42 goalscorers.csv (~47 600 buts, avec buteur, minute, pénalty, csc).

Modèle de pondération (par joueur, sur une fenêtre récente) :
  goal_rate  = buts_récents / matchs_de_l'équipe_dans_la_fenêtre
               -> buts attendus par match de l'équipe (le bon dénominateur :
                  goalscorers.csv ne contient pas les présences, mais results.csv
                  donne le nombre de matchs joués par l'équipe sur la fenêtre)
  start_prob = décroissance selon les mois depuis le dernier but
               -> écarte naturellement les joueurs retraités / hors rotation
  is_pen_taker = a marqué >= 2 pénos sur la fenêtre

Le poids final (dans sim/tournament.build_player_weights) :
  w = goal_rate * start_prob * (1.3 si tireur de pénos)
Quand un match simulé donne N buts à une équipe, les buteurs sont tirés ~ w.

Sortie : data/players.csv
Colonnes : player, team, goals_total, goals_recent, caps_recent,
           goal_rate, pen_goals, is_pen_taker, start_prob
"""
import numpy as np
import pandas as pd

from config import DATA_DIR
from ingestion.results_fetch import fetch_goalscorers, fetch_results

OUTPUT = DATA_DIR / "players.csv"

# Fenêtre récente : cycle Mondial courant. Aujourd'hui = mi-2026 -> ~4 ans.
WINDOW_YEARS       = 4
MIN_RECENT_GOALS   = 2      # filtre les buteurs accidentels
START_HALFLIFE_MO  = 18.0   # demi-vie de la "présence probable" en mois


def _team_match_counts(window_start: pd.Timestamp) -> dict[str, int]:
    """Nombre de matchs joués par équipe sur la fenêtre (dénominateur goal_rate)."""
    res = fetch_results()
    res = res[res["date"] >= window_start]
    counts: dict[str, int] = {}
    for col in ("home_team", "away_team"):
        vc = res[col].value_counts()
        for team, n in vc.items():
            counts[team] = counts.get(team, 0) + int(n)
    return counts


def _build(wc_teams: list[str], ref_date: pd.Timestamp) -> pd.DataFrame:
    gs = fetch_goalscorers()
    if gs.empty:
        return pd.DataFrame(columns=[
            "player", "team", "goals_total", "goals_recent",
            "caps_recent", "goal_rate", "pen_goals", "is_pen_taker", "start_prob"
        ])

    gs = gs[~gs["own_goal"]].copy()
    if wc_teams:
        gs = gs[gs["team"].isin(wc_teams)]

    window_start = ref_date - pd.DateOffset(years=WINDOW_YEARS)
    gs_recent = gs[gs["date"] >= window_start]

    # Total carrière (affichage)
    total = (gs.groupby(["team", "scorer"]).size()
               .reset_index(name="goals_total"))

    # Agrégats récents
    recent = (gs_recent.groupby(["team", "scorer"])
              .agg(goals_recent=("scorer", "count"),
                   pen_goals=("penalty", "sum"),
                   last_goal=("date", "max"))
              .reset_index())

    df = recent.merge(total, on=["team", "scorer"], how="left")
    df = df.rename(columns={"scorer": "player"})
    df["goals_total"] = df["goals_total"].fillna(df["goals_recent"]).astype(int)

    # Dénominateur : matchs de l'équipe sur la fenêtre
    team_matches = _team_match_counts(window_start)
    df["caps_recent"] = df["team"].map(team_matches).fillna(0).astype(int)

    # goal_rate = buts attendus par match d'équipe (borné pour éviter division folle)
    df["goal_rate"] = df["goals_recent"] / df["caps_recent"].clip(lower=8)

    # is_pen_taker
    df["is_pen_taker"] = (df["pen_goals"] >= 2).astype(int)

    # start_prob : décroissance selon les mois depuis le dernier but
    months_since = ((ref_date - df["last_goal"]).dt.days / 30.44).clip(lower=0)
    df["start_prob"] = (0.5 ** (months_since / START_HALFLIFE_MO)).clip(0.05, 1.0)

    df = df[df["goals_recent"] >= MIN_RECENT_GOALS].copy()

    return df[["player", "team", "goals_total", "goals_recent",
               "caps_recent", "goal_rate", "pen_goals",
               "is_pen_taker", "start_prob"]]


def fetch_players(wc_teams: list[str] | None = None,
                  enrich_fbref: bool = False,   # conservé pour compat ; FBref bloqué
                  force: bool = False) -> pd.DataFrame:
    """
    Charge les statistiques joueurs pour la simulation Soulier d'or.

    Parameters
    ----------
    wc_teams : liste des équipes qualifiées (filtre). Si None, toutes les équipes.
    force    : recharge même si le cache existe.
    """
    if OUTPUT.exists() and not force:
        return pd.read_csv(OUTPUT)

    print("Construction des stats buteurs (martj42 goalscorers)...")
    ref_date = pd.Timestamp.now()
    df = _build(wc_teams or [], ref_date)

    df = df.sort_values(["team", "goal_rate"], ascending=[True, False]).reset_index(drop=True)
    df.to_csv(OUTPUT, index=False)
    print(f"Donnees joueurs sauvegardees : {len(df)} joueurs -> {OUTPUT}")
    return df


if __name__ == "__main__":
    df = fetch_players(force=True)
    print("\nTop 25 buteurs (goal_rate x start_prob) :")
    df = df.copy()
    df["w"] = df["goal_rate"] * df["start_prob"] * (1 + 0.3 * df["is_pen_taker"])
    print(df.sort_values("w", ascending=False)
            .head(25)[["player", "team", "goals_recent", "caps_recent",
                       "goal_rate", "is_pen_taker", "start_prob"]]
            .to_string(index=False))
