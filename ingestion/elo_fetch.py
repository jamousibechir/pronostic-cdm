"""
Récupère les ratings Elo internationaux depuis eloratings.net.

Source : https://www.eloratings.net/World.tsv (fichier TSV public)
Fallback : scrape HTML de la page principale si le TSV n'est pas disponible.

Sortie : data/elo_ratings.csv
  Colonnes : team, elo, rank
"""
import re
import requests
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup

from config import DATA_DIR, ELO_URL


HEADERS = {"User-Agent": "pronostic-cdm/1.0 (research project)"}
OUTPUT = DATA_DIR / "elo_ratings.csv"

# Mapping noms eloratings.net → noms FIFA standard (à compléter si besoin)
ELO_NAME_MAP = {
    "United States": "USA",
    "Iran": "IR Iran",
    "South Korea": "Korea Republic",
    "North Korea": "Korea DPR",
    "Ivory Coast": "Côte d'Ivoire",
    "DR Congo": "Congo DR",
}


def _fetch_tsv() -> pd.DataFrame | None:
    """Essaie de télécharger le TSV directement."""
    try:
        r = requests.get(ELO_URL, headers=HEADERS, timeout=15)
        r.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(r.text), sep="\t", header=None,
                         names=["rank", "team", "elo", "change"])
        df = df[["team", "elo", "rank"]].dropna(subset=["team", "elo"])
        df["elo"] = pd.to_numeric(df["elo"], errors="coerce")
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
        return df.dropna()
    except Exception:
        return None


def _fetch_html() -> pd.DataFrame:
    """Scrape la page HTML si le TSV échoue."""
    url = "https://www.eloratings.net/"
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    rows = []
    for tr in soup.select("table tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) >= 3:
            try:
                rank = int(cells[0])
                team = cells[1]
                elo  = float(re.sub(r"[^\d.]", "", cells[2]))
                rows.append({"rank": rank, "team": team, "elo": elo})
            except (ValueError, IndexError):
                continue
    if not rows:
        raise RuntimeError("Impossible de scraper eloratings.net")
    return pd.DataFrame(rows)


def fetch_elo(force: bool = False) -> pd.DataFrame:
    """
    Charge les ratings Elo. Utilise le cache si disponible et force=False.

    Returns
    -------
    pd.DataFrame : colonnes [team, elo, rank]
    """
    if OUTPUT.exists() and not force:
        return pd.read_csv(OUTPUT)

    df = _fetch_tsv()
    if df is None:
        print("TSV eloratings.net indisponible, fallback HTML...")
        df = _fetch_html()

    # Normalisation des noms
    df["team"] = df["team"].replace(ELO_NAME_MAP)
    df = df.sort_values("rank").reset_index(drop=True)
    df.to_csv(OUTPUT, index=False)
    print(f"Elo ratings sauvegardés : {len(df)} équipes → {OUTPUT}")
    return df


if __name__ == "__main__":
    print(fetch_elo(force=True).head(20))
