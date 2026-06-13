"""
Assignation de chaque équipe à sa confédération FIFA (UEFA, CONMEBOL, CAF, AFC,
CONCACAF, OFC), pour le shrinkage hiérarchique à deux niveaux du modèle de force.

Méthode (automatique) : on regarde dans quelles COMPÉTITIONS CONTINENTALES chaque
équipe a joué (le champ `tournament` de martj42), et on prend la confédération
majoritaire — restreinte aux ~20 dernières années pour refléter l'appartenance
actuelle (ex. l'Australie a basculé OFC -> AFC en 2006).

Quelques cas particuliers (géographie != confédération) sont forcés manuellement.

Sortie : data/confederations.csv  (team, confederation)
"""
import pandas as pd

from config import DATA_DIR
from ingestion.results_fetch import fetch_results
from ingestion.names import canonical

OUTPUT = DATA_DIR / "confederations.csv"

CONFEDERATIONS = ["UEFA", "CONMEBOL", "CAF", "AFC", "CONCACAF", "OFC"]

# Mots-clés NON ambigus par confédération (minuscule). Les compétitions mixtes
# (Arab Cup, Afro-Asian Games, Confederations Cup...) sont volontairement exclues.
_CONF_KEYWORDS = {
    "UEFA":     ["uefa", "central european international"],
    "CONMEBOL": ["copa am", "conmebol"],
    "CAF":      ["african cup of nations", "cecafa", "cosafa", "all-african",
                 "african friendship", "west african", "caf ", "african games"],
    "AFC":      ["afc", "asian cup", "asian games", "gulf cup", "aff championship",
                 "saff", "eaff", "southeast asian", "south asian games",
                 "east asian games", "waff", "cafa", "asean"],
    "CONCACAF": ["concacaf", "gold cup", "caribbean", "cfu", "uncaf",
                 "central american and caribbean", "nafc"],
    "OFC":      ["oceania", "ofc"],
}

# Forçages manuels (appartenance != géographie, ou histoire ambiguë)
_OVERRIDES = {
    "Australia":   "AFC",   # OFC -> AFC depuis 2006
    "Israel":      "UEFA",  # joue en UEFA
    "Kazakhstan":  "UEFA",  # AFC -> UEFA en 2002
    "Guyana":      "CONCACAF",
    "Suriname":    "CONCACAF",
    "French Guiana": "CONCACAF",
    "United States": "CONCACAF",
    "Mexico":      "CONCACAF",
    "Canada":      "CONCACAF",
}

_RECENT_FROM = "2005-01-01"   # fenêtre pour refléter l'appartenance actuelle


def _conf_of_tournament(name: str) -> str | None:
    n = str(name).lower()
    for conf, kws in _CONF_KEYWORDS.items():
        if any(k in n for k in kws):
            return conf
    return None


def build_confederations(force: bool = False) -> pd.DataFrame:
    """Construit (et met en cache) la table team -> confederation."""
    if OUTPUT.exists() and not force:
        return pd.read_csv(OUTPUT)

    res = fetch_results()
    res = res[res["date"] >= _RECENT_FROM].copy()
    res["conf"] = res["tournament"].map(_conf_of_tournament)
    cont = res.dropna(subset=["conf"])

    # Compte par équipe et par confédération (domicile + extérieur)
    counts: dict[str, dict[str, int]] = {}
    for col in ("home_team", "away_team"):
        for team, conf in zip(cont[col], cont["conf"]):
            counts.setdefault(team, {}).setdefault(conf, 0)
            counts[team][conf] += 1

    rows = []
    for team, c in counts.items():
        rows.append({"team": team, "confederation": max(c, key=c.get)})
    df = pd.DataFrame(rows)

    # Applique les forçages
    for team, conf in _OVERRIDES.items():
        team = canonical(team)
        if (df["team"] == team).any():
            df.loc[df["team"] == team, "confederation"] = conf
        else:
            df = pd.concat([df, pd.DataFrame([{"team": team, "confederation": conf}])],
                           ignore_index=True)

    df = df.sort_values("team").reset_index(drop=True)
    df.to_csv(OUTPUT, index=False)
    print(f"  Confederations : {len(df)} equipes -> {OUTPUT}")
    print("  Repartition :",
          df["confederation"].value_counts().to_dict())
    return df


_CACHE: dict[str, str] | None = None

def confederation_map(force: bool = False) -> dict[str, str]:
    global _CACHE
    if _CACHE is None or force:
        df = build_confederations(force=force)
        _CACHE = dict(zip(df["team"], df["confederation"]))
    return _CACHE


def confederation_of(team: str, default: str = "UEFA") -> str:
    return confederation_map().get(canonical(team), default)


if __name__ == "__main__":
    df = build_confederations(force=True)
    for conf in CONFEDERATIONS:
        teams = df[df["confederation"] == conf]["team"].tolist()
        print(f"\n{conf} ({len(teams)}): {', '.join(teams[:12])}...")
