"""
Pipeline de mise à jour QUOTIDIENNE pendant le tournoi (juin–juillet 2026).

À lancer chaque soir. Évite que le modèle a priori ne devienne obsolète dès le
premier match et gère la forte variance d'un tournoi court.

Étapes :
  1. Rafraîchit les résultats CdM (scores live football-data + martj42).
  2. Débruitage bayésien par les xG : remplace le score brut d'un match CdM par
     un mélange  XG_BLEND·xG + (1−XG_BLEND)·score. Une équipe malchanceuse
     (perd 1-0 avec 3.0 xG contre 0.5) est ainsi BEAUCOUP moins pénalisée.
  3. K-factor : poids fortement majoré pour les matchs CdM en cours, pour que le
     modèle apprenne vite la forme actuelle des équipes.
  4. Estime les forces, applique les ajustements manuels d'effectif
     (data/daily_adjustments.json : malus blessure/suspension en log-force),
     puis sauvegarde le modèle de production.
  5. Relance predict.py --update : fige les matchs déjà joués et re-simule le
     Monte-Carlo sur le bracket partiel -> nouveaux pourcentages de titre.

Fichiers d'entrée (optionnels, déposés manuellement) :
  data/xg_daily.csv            colonnes : date, home_team, away_team, home_xg, away_xg
  data/daily_adjustments.json  ex : {"France": -0.15, "Brazil": -0.05}

Usage : python update_daily.py
"""
import json
import subprocess
import sys
import pandas as pd

# Force UTF-8 (PowerShell cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import (DATA_DIR, RIDGE_TEAM, RIDGE_CONF, TRAIN_CUTOFF_YEARS,
                    XG_BLEND, WC_K_FACTOR, WC_SINCE)
from ingestion.results_fetch import recent_results
from ingestion.elo_fetch import fetch_elo
from ingestion.confederations import confederation_map
from ingestion.fixtures_fetch import update_live_results
from ingestion.names import canonical
from model.elo import build_elo_dict
from model.strength_poisson import estimate_strengths, save_params

XG_CSV   = DATA_DIR / "xg_daily.csv"
ADJ_JSON = DATA_DIR / "daily_adjustments.json"


def _load_xg() -> pd.DataFrame | None:
    if not XG_CSV.exists():
        print("  (pas de data/xg_daily.csv : debruitage xG ignore)")
        return None
    df = pd.read_csv(XG_CSV)
    df["date"]      = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["home_team"] = df["home_team"].map(canonical)
    df["away_team"] = df["away_team"].map(canonical)
    return df


def _denoise_with_xg(matches: pd.DataFrame, xg: pd.DataFrame | None,
                     blend: float) -> pd.DataFrame:
    """Remplace le score des matchs présents dans xg par un mélange xG/score réel."""
    if xg is None or xg.empty:
        return matches
    m = matches.copy()
    m["_d"] = pd.to_datetime(m["date"]).dt.normalize()
    xmap = {(r._d, r.home_team, r.away_team): (float(r.home_xg), float(r.away_xg))
            for r in xg.itertuples()}
    n = 0
    for i, row in m.iterrows():
        k = (row["_d"], row["home_team"], row["away_team"])
        if k in xmap:
            hx, ax = xmap[k]
            m.at[i, "home_score"] = blend * hx + (1 - blend) * float(row["home_score"])
            m.at[i, "away_score"] = blend * ax + (1 - blend) * float(row["away_score"])
            n += 1
    print(f"  Debruitage xG : {n} match(s) ajuste(s) (blend={blend})")
    return m.drop(columns=["_d"])


def _build_training(blend: float, k_factor: float, wc_since: str) -> pd.DataFrame:
    res = recent_results(years=TRAIN_CUTOFF_YEARS, force=True)   # données fraîches
    res = _denoise_with_xg(res, _load_xg(), blend)
    res["weight_mult"] = 1.0
    dt = pd.to_datetime(res["date"])
    wc_mask = (res["tournament"].str.contains("World Cup", case=False, na=False)
               & ~res["tournament"].str.contains("qualification", case=False, na=False)
               & (dt >= pd.Timestamp(wc_since)))
    res.loc[wc_mask, "weight_mult"] = k_factor
    print(f"  K-factor {k_factor} applique a {int(wc_mask.sum())} match(s) CdM en cours")
    return res


def _apply_adjustments(params: dict) -> dict:
    if not ADJ_JSON.exists():
        print("  (pas de data/daily_adjustments.json : aucun malus d'effectif)")
        return params
    adj = json.loads(ADJ_JSON.read_text(encoding="utf-8"))
    for team, delta in adj.items():
        t = canonical(team)
        if t in params["strength"]:
            params["strength"][t] += float(delta)
            print(f"  Ajustement effectif {t}: {float(delta):+.3f} log-force")
    return params


def main():
    print("=" * 60)
    print("MISE A JOUR QUOTIDIENNE - CdM 2026")
    print("=" * 60)

    print("\n[1/4] Resultats live...")
    update_live_results()

    print("\n[2/4] Construction du jeu d'entrainement (xG + K-factor)...")
    res = _build_training(XG_BLEND, WC_K_FACTOR, WC_SINCE)

    print("\n[3/4] Re-estimation des forces + ajustements d'effectif...")
    elo  = build_elo_dict(fetch_elo())
    conf = confederation_map()
    params = estimate_strengths(res, ridge_team=RIDGE_TEAM, ridge_conf=RIDGE_CONF,
                                elo_dict=elo, conf_map=conf)
    params = _apply_adjustments(params)
    save_params(params)

    print("\n[4/4] Re-simulation sur le bracket partiel (predict.py --update)...")
    # predict.py --update charge le modele en cache (le notre), fige les matchs
    # joues et rejoue le Monte-Carlo.
    subprocess.run([sys.executable, "predict.py", "--update"], check=False)
    print("\nMise a jour quotidienne terminee.")


if __name__ == "__main__":
    main()
