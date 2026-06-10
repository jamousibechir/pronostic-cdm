"""
Extrait l'historique des grandes compétitions par équipe.

Source : results.csv (martj42) — champ `tournament`

Grandes compétitions couvertes :
  - FIFA World Cup (1930 → 2022)
  - UEFA European Championship (1960 → 2024)
  - Copa América (1916 → 2024)
  - Africa Cup of Nations (1957 → 2024)
  - AFC Asian Cup (1956 → 2023)
  - CONCACAF Gold Cup (1941 → 2023)

Sortie : data/competition_history.csv
Colonnes : team, competition, year, round, is_champion, is_finalist,
           is_semifinalist, is_quarterfinalist
"""
import pandas as pd
import numpy as np

from config import DATA_DIR, MAJOR_TOURNAMENTS
from ingestion.results_fetch import fetch_results

OUTPUT = DATA_DIR / "competition_history.csv"

# Mapping pour normaliser les noms de tournois dans results.csv
TOURNAMENT_LABELS = {
    "FIFA World Cup":               "World Cup",
    "UEFA Euro":                    "Euro",
    "Copa América":                 "Copa America",
    "Africa Cup of Nations":        "AFCON",
    "AFC Asian Cup":                "Asian Cup",
    "CONCACAF Gold Cup":            "Gold Cup",
    "FIFA Confederations Cup":      "Confederations Cup",
}

# Mots-clés de stade (pour détecter le round depuis le nom du tournoi absent)
ROUND_KW = {
    "final":        "Final",
    "semi":         "Semi-final",
    "quarter":      "Quarter-final",
    "round of 16":  "Round of 16",
    "round of 32":  "Round of 32",
    "group":        "Group stage",
}


def _detect_round(tournament: str) -> str:
    t = tournament.lower()
    for kw, label in ROUND_KW.items():
        if kw in t:
            return label
    return "Group stage"


def _process_competition(df_comp: pd.DataFrame,
                          comp_name: str) -> pd.DataFrame:
    """
    Pour une compétition donnée, détermine le round atteint par chaque équipe
    et par année d'édition.
    """
    # Identifie les éditions par année
    df_comp = df_comp.copy()
    df_comp["year"] = df_comp["date"].dt.year

    records = []
    for year, df_year in df_comp.groupby("year"):
        # Toutes les équipes qui ont joué cette édition
        teams_in = set(df_year["home_team"]) | set(df_year["away_team"])

        # Meilleur round atteint par équipe dans cette édition
        round_order = {
            "Group stage": 0, "Round of 32": 1, "Round of 16": 2,
            "Quarter-final": 3, "Semi-final": 4, "Final": 5,
        }

        team_rounds: dict[str, str] = {t: "Group stage" for t in teams_in}
        team_champion: dict[str, bool] = {t: False for t in teams_in}

        for _, row in df_year.iterrows():
            rnd   = _detect_round(row.get("tournament", ""))
            home  = row["home_team"]
            away  = row["away_team"]
            for team in [home, away]:
                if team in team_rounds:
                    if round_order.get(rnd, 0) > round_order.get(team_rounds[team], 0):
                        team_rounds[team] = rnd

        # Vainqueur = équipe qui a gagné la finale
        finals = df_year[df_year["tournament"].str.lower().str.contains("final", na=False)
                         & ~df_year["tournament"].str.lower().str.contains("semi", na=False)
                         & ~df_year["tournament"].str.lower().str.contains("third", na=False)
                         & ~df_year["tournament"].str.lower().str.contains("3rd", na=False)]

        if not finals.empty:
            last_final = finals.sort_values("date").iloc[-1]
            h, a = last_final["home_team"], last_final["away_team"]
            hs, as_ = last_final["home_score"], last_final["away_score"]
            if hs > as_:
                team_champion[h] = True
                team_rounds[h] = "Final"
                team_rounds[a] = "Final"
            elif as_ > hs:
                team_champion[a] = True
                team_rounds[h] = "Final"
                team_rounds[a] = "Final"

        for team, rnd in team_rounds.items():
            records.append({
                "team":              team,
                "competition":       TOURNAMENT_LABELS.get(comp_name, comp_name),
                "year":              year,
                "round":             rnd,
                "is_champion":       team_champion.get(team, False),
                "is_finalist":       rnd == "Final",
                "is_semifinalist":   rnd in ("Semi-final", "Final"),
                "is_quarterfinalist": rnd in ("Quarter-final", "Semi-final", "Final"),
            })

    return pd.DataFrame(records)


def build_competition_history(force: bool = False) -> pd.DataFrame:
    """
    Construit l'historique complet des grandes compétitions.

    Returns
    -------
    pd.DataFrame avec une ligne par (team, competition, year)
    """
    if OUTPUT.exists() and not force:
        return pd.read_csv(OUTPUT)

    results = fetch_results()
    frames  = []

    for comp in MAJOR_TOURNAMENTS:
        mask = results["tournament"].str.contains(comp, na=False, case=False)
        df_comp = results[mask]
        if df_comp.empty:
            continue
        print(f"  Traitement : {comp} ({len(df_comp)} matchs)")
        frames.append(_process_competition(df_comp, comp))

    history = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    history.to_csv(OUTPUT, index=False)
    print(f"Historique compétitions sauvegardé : {len(history)} lignes → {OUTPUT}")
    return history


def team_summary(team: str,
                 history: pd.DataFrame | None = None) -> dict:
    """
    Retourne un résumé de palmarès pour une équipe.

    Exemple de retour :
    {
      "World Cup titles": 2,
      "World Cup finals": 3,
      "Euro titles": 1,
      ...
    }
    """
    if history is None:
        history = build_competition_history()

    df = history[history["team"] == team]
    summary: dict = {}
    for comp in df["competition"].unique():
        dc = df[df["competition"] == comp]
        summary[f"{comp} titles"]      = int(dc["is_champion"].sum())
        summary[f"{comp} finals"]      = int(dc["is_finalist"].sum())
        summary[f"{comp} semi-finals"] = int(dc["is_semifinalist"].sum())
    return summary


def top_nations_by_wc_titles(n: int = 10,
                              history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Classement des nations par nombre de titres mondiaux."""
    if history is None:
        history = build_competition_history()
    wc = history[history["competition"] == "World Cup"]
    return (
        wc.groupby("team")["is_champion"]
        .sum()
        .astype(int)
        .nlargest(n)
        .reset_index(name="wc_titles")
    )


if __name__ == "__main__":
    hist = build_competition_history(force=True)
    print(top_nations_by_wc_titles(history=hist))
    print(team_summary("France", history=hist))
