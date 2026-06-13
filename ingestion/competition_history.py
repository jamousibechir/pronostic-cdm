"""
Palmarès historique des grandes compétitions, par équipe.

Coupe du Monde : piloté par les données jfjelstul/worldcup, qui fournit le
`stage_name` ('final', 'semi-finals', ...) absent de martj42. On détecte le
champion / finaliste / demi-finaliste de chaque édition à partir des matchs KO.

Euro & Copa América : tables de référence (palmarès public, stable) — utilisées
seulement pour enrichir l'affichage de champion.csv, pas pour la prédiction.

Sortie : data/competition_history.csv (WC détaillé) + team_summary() agrège tout.
"""
import io
import requests
import pandas as pd

from config import DATA_DIR
from ingestion.names import canonical

OUTPUT = DATA_DIR / "competition_history.csv"
JFJELSTUL_URL = "https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/matches.csv"
UA = {"User-Agent": "Mozilla/5.0"}

_STAGE_ORDER = {
    "group stage": 0, "first round": 0,
    "round of 16": 1, "second round": 1,
    "quarter-finals": 2, "semi-finals": 3,
    "third-place match": 3, "final": 4,
}

# ── Palmarès Euro (titres par nation, ères modernes regroupées) ───────────────
EURO_TITLES = {
    "Spain": 4, "Germany": 3, "France": 2, "Italy": 2,
    "Portugal": 1, "Netherlands": 1, "Denmark": 1, "Greece": 1,
    "Czech Republic": 1, "Russia": 1,
}
# ── Palmarès Copa América (titres historiques par nation) ─────────────────────
COPA_TITLES = {
    "Argentina": 16, "Uruguay": 15, "Brazil": 9, "Paraguay": 2,
    "Peru": 2, "Chile": 2, "Colombia": 1, "Bolivia": 1,
}


def _winner_of(row: pd.Series) -> str | None:
    """Vainqueur d'un match jfjelstul (gère les tirs au but)."""
    if bool(row.get("home_team_win", False)):
        return canonical(row["home_team_name"])
    if bool(row.get("away_team_win", False)):
        return canonical(row["away_team_name"])
    # Égalité réglée aux pénos
    hp = row.get("home_team_score_penalties", 0) or 0
    ap = row.get("away_team_score_penalties", 0) or 0
    if hp > ap:
        return canonical(row["home_team_name"])
    if ap > hp:
        return canonical(row["away_team_name"])
    return None


def build_competition_history(force: bool = False) -> pd.DataFrame:
    """
    Construit le palmarès WC détaillé (une ligne par équipe/édition).
    Colonnes : team, year, round, is_champion, is_finalist, is_semifinalist
    """
    if OUTPUT.exists() and not force:
        return pd.read_csv(OUTPUT)

    try:
        r = requests.get(JFJELSTUL_URL, headers=UA, timeout=30)
        r.raise_for_status()
        wc = pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        print(f"  Historique WC indisponible ({e})")
        empty = pd.DataFrame(columns=["team", "year", "round",
                                      "is_champion", "is_finalist", "is_semifinalist"])
        empty.to_csv(OUTPUT, index=False)
        return empty

    wc = wc[wc["tournament_id"].str.match(r"WC-\d{4}$")].copy()
    wc = wc[~wc["tournament_name"].str.contains("Women", na=False, case=False)]
    wc["year"]  = wc["tournament_id"].str.extract(r"WC-(\d{4})").astype(int)
    wc["stage"] = wc["stage_name"].str.lower().str.strip()

    records = []
    for year, ed in wc.groupby("year"):
        teams = set(ed["home_team_name"].map(canonical)) | set(ed["away_team_name"].map(canonical))
        best = {t: 0 for t in teams}
        for _, row in ed.iterrows():
            lvl = _STAGE_ORDER.get(row["stage"], 0)
            for t in (canonical(row["home_team_name"]), canonical(row["away_team_name"])):
                best[t] = max(best[t], lvl)

        # Champion = vainqueur de la finale
        finals = ed[ed["stage"] == "final"]
        champion = _winner_of(finals.iloc[-1]) if not finals.empty else None

        for t in teams:
            records.append({
                "team":            t,
                "year":            int(year),
                "round":           best[t],
                "is_champion":     (t == champion),
                "is_finalist":     best[t] >= 4,
                "is_semifinalist": best[t] >= 3,
            })

    hist = pd.DataFrame(records)
    hist.to_csv(OUTPUT, index=False)
    n_champs = hist["is_champion"].sum()
    print(f"Palmares WC sauvegarde : {len(hist)} lignes, {n_champs} titres -> {OUTPUT}")
    return hist


def team_summary(team: str, history: pd.DataFrame | None = None) -> dict:
    """Résumé de palmarès pour une équipe (clés attendues par predict.py)."""
    if history is None:
        history = build_competition_history()
    team = canonical(team)
    df = history[history["team"] == team]
    return {
        "World Cup titles":      int(df["is_champion"].sum()) if not df.empty else 0,
        "World Cup finals":      int(df["is_finalist"].sum()) if not df.empty else 0,
        "World Cup semi-finals": int(df["is_semifinalist"].sum()) if not df.empty else 0,
        "Euro titles":           EURO_TITLES.get(team, 0),
        "Copa America titles":   COPA_TITLES.get(team, 0),
    }


def top_nations_by_wc_titles(n: int = 10,
                             history: pd.DataFrame | None = None) -> pd.DataFrame:
    if history is None:
        history = build_competition_history()
    return (history.groupby("team")["is_champion"].sum().astype(int)
            .nlargest(n).reset_index(name="wc_titles"))


if __name__ == "__main__":
    hist = build_competition_history(force=True)
    print(top_nations_by_wc_titles(history=hist).to_string(index=False))
    print()
    for t in ["Brazil", "Germany", "Argentina", "France", "Italy", "Spain"]:
        print(t, team_summary(t, hist))
