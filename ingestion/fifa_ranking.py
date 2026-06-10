"""
Récupère le classement FIFA officiel.

Source primaire : endpoint JSON non officiel de fifa.com
Source secondaire : scrape HTML de la page de classement FIFA

Sortie : data/fifa_rankings.csv
  Colonnes : team, fifa_rank, fifa_points
"""
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup

from config import DATA_DIR

HEADERS = {"User-Agent": "pronostic-cdm/1.0 (research project)"}
OUTPUT  = DATA_DIR / "fifa_rankings.csv"

# Endpoint JSON non officiel (extrait du JS de la page FIFA)
FIFA_JSON_URL = (
    "https://www.fifa.com/en/ranking/men?dateId=id14218"
)
FIFA_API_URL = (
    "https://www.fifa.com/en/ranking/men"
)


def _fetch_json() -> pd.DataFrame | None:
    """Essaie l'endpoint JSON de la page FIFA."""
    try:
        url = "https://www.fifa.com/en/ranking-api/men"
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
        rankings = data.get("rankings", data.get("data", []))
        rows = []
        for item in rankings:
            name   = item.get("countryName") or item.get("name", "")
            rank   = item.get("rank", item.get("ranking", 0))
            points = item.get("totalPoints", item.get("points", 0))
            rows.append({"team": name, "fifa_rank": rank, "fifa_points": points})
        if rows:
            return pd.DataFrame(rows)
    except Exception:
        pass
    return None


def _fetch_html() -> pd.DataFrame:
    """Scrape la page HTML de classement FIFA."""
    r = requests.get(FIFA_API_URL, headers=HEADERS, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    rows = []
    # Cherche le JSON embarqué dans la page (Next.js / React)
    for script in soup.find_all("script", {"id": "__NEXT_DATA__"}):
        data = json.loads(script.string)
        try:
            # Chemin variable selon la version du site FIFA
            ranking_data = (
                data["props"]["pageProps"]["pageData"]["rankings"]
            )
            for item in ranking_data:
                rows.append({
                    "team":        item["countryName"],
                    "fifa_rank":   item["rank"],
                    "fifa_points": item["totalPoints"],
                })
            break
        except (KeyError, TypeError):
            continue
    if not rows:
        # Fallback : parse le tableau HTML visible
        for tr in soup.select("table tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(cells) >= 3:
                try:
                    rows.append({
                        "team":        cells[1],
                        "fifa_rank":   int(cells[0]),
                        "fifa_points": float(cells[2].replace(",", ".")),
                    })
                except ValueError:
                    continue
    if not rows:
        raise RuntimeError("Impossible de récupérer le classement FIFA")
    return pd.DataFrame(rows)


def fetch_fifa_rankings(force: bool = False) -> pd.DataFrame:
    """
    Charge le classement FIFA. Utilise le cache si disponible et force=False.

    Returns
    -------
    pd.DataFrame : colonnes [team, fifa_rank, fifa_points]
    """
    if OUTPUT.exists() and not force:
        return pd.read_csv(OUTPUT)

    df = _fetch_json()
    if df is None:
        print("Endpoint JSON FIFA indisponible, fallback HTML...")
        df = _fetch_html()

    df = df.sort_values("fifa_rank").reset_index(drop=True)
    df.to_csv(OUTPUT, index=False)
    print(f"Classement FIFA sauvegardé : {len(df)} équipes → {OUTPUT}")
    return df


if __name__ == "__main__":
    print(fetch_fifa_rankings(force=True).head(20))
