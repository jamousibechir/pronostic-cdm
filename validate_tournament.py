"""
Validation de la COUCHE SIMULATION (et pas seulement du modèle de match).

Un bon modèle de match peut quand même mal agréger en probabilités de tour /
de titre (variance, structure du bracket). On vérifie donc, sur les Mondiaux
2014/2018/2022 (format 32 équipes), que les probabilités d'accès par tour sont
CALIBRÉES : une équipe annoncée à ~70 % d'atteindre les quarts doit y parvenir
~70 % du temps.

Protocole (sans fuite) : pour chaque édition, on entraîne le modèle de force sur
les matchs ANTÉRIEURS au coup d'envoi, on reconstitue les 8 groupes depuis
jfjelstul/worldcup, on lance le Monte-Carlo (format 32 : 8 groupes -> R16 -> QF
-> SF -> finale), puis on compare P(atteindre le tour) prédit vs réalité.

Lancer : python validate_tournament.py
"""
import io
import requests
import numpy as np
import pandas as pd
from collections import defaultdict

from ingestion.results_fetch import fetch_results
from ingestion.elo_fetch import fetch_elo
from ingestion.confederations import confederation_map
from model.elo import build_elo_dict
from model.strength_poisson import estimate_strengths
from sim.tournament import _simulate_group, _ko_match
from ingestion.names import canonical

JFJELSTUL_URL = "https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/matches.csv"
UA = {"User-Agent": "Mozilla/5.0"}

WC_START = {2014: "2014-06-12", 2018: "2018-06-14", 2022: "2022-11-20"}
# Coûteux (boucles Python). Réglable en CLI : python validate_tournament.py --n-sims 2000
N_SIMS = 1000

# Niveaux atteints
LEVELS = ["R16", "QF", "SF", "Final", "Champion"]
_STAGE_LEVEL = {  # niveau ATTEINT en jouant ce stade
    "round of 16": 1, "quarter-finals": 2, "semi-finals": 3,
    "third-place match": 3, "final": 4,
}

# Bracket classique 32 équipes (croisement 1er/2e des groupes A..H)
_R16 = [("A", "B"), ("C", "D"), ("E", "F"), ("G", "H"),
        ("B", "A"), ("D", "C"), ("F", "E"), ("H", "G")]


def _load_edition(wc: pd.DataFrame, year: int):
    """Retourne (groups{grp:[teams]}, actual_reach{team:level}, champion)."""
    ed = wc[wc["tournament_id"] == f"WC-{year}"].copy()
    ed["stage"] = ed["stage_name"].str.lower().str.strip()

    # Groupes
    groups: dict[str, list[str]] = {}
    gs = ed[ed["group_name"].notna() & (ed["group_name"] != "")]
    for _, r in gs.iterrows():
        g = str(r["group_name"]).replace("Group ", "").strip()
        for t in (canonical(r["home_team_name"]), canonical(r["away_team_name"])):
            groups.setdefault(g, [])
            if t not in groups[g]:
                groups[g].append(t)
    groups = {g: t for g, t in groups.items() if len(t) == 4}

    # Niveau réellement atteint
    reach: dict[str, int] = defaultdict(int)
    for _, r in ed.iterrows():
        lvl = _STAGE_LEVEL.get(r["stage"], 0)
        for t in (canonical(r["home_team_name"]), canonical(r["away_team_name"])):
            reach[t] = max(reach[t], lvl)

    finals = ed[ed["stage"] == "final"]
    champion = None
    if not finals.empty:
        f = finals.iloc[-1]
        if bool(f.get("home_team_win", False)):
            champion = canonical(f["home_team_name"])
        elif bool(f.get("away_team_win", False)):
            champion = canonical(f["away_team_name"])
        else:
            hp = f.get("home_team_score_penalties", 0) or 0
            ap = f.get("away_team_score_penalties", 0) or 0
            champion = canonical(f["home_team_name"]) if hp >= ap else canonical(f["away_team_name"])
    return groups, dict(reach), champion


def _simulate_wc32(groups, params, n_sims, seed):
    """Monte-Carlo format 32. Retourne {team: {niveau: proba}}."""
    rng = np.random.default_rng(seed)
    reach_count = {lvl: defaultdict(int) for lvl in LEVELS}

    for _ in range(n_sims):
        winners, runners = {}, {}
        for g, teams in groups.items():
            standings = _simulate_group(teams, params, rng, {})
            winners[g] = standings.iloc[0]["team"]
            runners[g] = standings.iloc[1]["team"]
        # tout le monde en R16 = 16 qualifiés
        for g in groups:
            reach_count["R16"][winners[g]] += 1
            reach_count["R16"][runners[g]] += 1

        # R16
        r16 = []
        for gw, gr in _R16:
            if gw in winners and gr in runners:
                home, away = winners[gw], runners[gr]
                w, _, _ = _ko_match(home, away, params, rng)
                r16.append(w)
        for t in r16:
            reach_count["QF"][t] += 1
        # QF
        qf = [_ko_match(r16[i], r16[i+1], params, rng)[0] for i in range(0, len(r16)-1, 2)]
        for t in qf:
            reach_count["SF"][t] += 1
        # SF
        sf = [_ko_match(qf[i], qf[i+1], params, rng)[0] for i in range(0, len(qf)-1, 2)]
        for t in sf:
            reach_count["Final"][t] += 1
        # Finale
        if len(sf) >= 2:
            champ, _, _ = _ko_match(sf[0], sf[1], params, rng)
            reach_count["Champion"][champ] += 1

    teams = [t for tl in groups.values() for t in tl]
    out = {t: {lvl: reach_count[lvl][t] / n_sims for lvl in LEVELS} for t in teams}
    return out


def run_validation(n_sims: int = N_SIMS, years: list[int] | None = None):
    print("=" * 60)
    print(f"VALIDATION DE LA SIMULATION (WC, {n_sims} sims/edition)")
    print("=" * 60)
    use_years = years or list(WC_START)

    results = fetch_results()
    results["date"] = pd.to_datetime(results["date"])
    elo_dict = build_elo_dict(fetch_elo())
    conf_map = confederation_map()
    wc = pd.read_csv(io.StringIO(requests.get(JFJELSTUL_URL, headers=UA, timeout=30).text))
    wc = wc[~wc["tournament_name"].str.contains("Women", na=False, case=False)]

    # Collecte (proba prédite, réalité 0/1) par niveau
    pred = {lvl: [] for lvl in LEVELS}
    obs  = {lvl: [] for lvl in LEVELS}
    champ_loglosses = []

    for year in use_years:
        start = WC_START[year]
        groups, reach, champion = _load_edition(wc, year)
        if len(groups) != 8:
            print(f"  {year}: structure incomplete ({len(groups)} groupes) - saute")
            continue
        train = results[results["date"] < start]
        params = estimate_strengths(train, ref_date=pd.Timestamp(start),
                                    elo_dict=elo_dict, conf_map=conf_map, verbose=False)
        probs = _simulate_wc32(groups, params, n_sims, seed=42 + year)

        teams = [t for tl in groups.values() for t in tl]
        for t in teams:
            lv = reach.get(t, 0)
            for k, lvl in enumerate(LEVELS, start=1):
                pred[lvl].append(probs[t][lvl])
                obs[lvl].append(1.0 if lv >= k else 0.0)
        # log-loss du champion
        if champion:
            p_c = probs.get(champion, {}).get("Champion", 1e-4)
            champ_loglosses.append(-np.log(max(p_c, 1e-4)))
            top = max(teams, key=lambda t: probs[t]["Champion"])
            print(f"  {year}: champion reel={champion} "
                  f"(P_pred={p_c:.1%}) | favori modele={top} ({probs[top]['Champion']:.1%})")

    # ── Calibration par tour ──
    print("\n-- Calibration des probabilites d'acces par tour --")
    print(f"  {'Tour':9s} {'n':>4s} {'P_moy':>7s} {'Obs':>7s} {'|ecart|':>8s}")
    cal_errs = []
    for lvl in LEVELS:
        p = np.array(pred[lvl]); o = np.array(obs[lvl])
        if len(p) == 0:
            continue
        err = abs(p.mean() - o.mean())
        cal_errs.append(err)
        print(f"  {lvl:9s} {len(p):4d} {p.mean():7.3f} {o.mean():7.3f} {err:8.3f}")
    print(f"\n  Erreur de calibration moyenne (tous tours) : {np.mean(cal_errs):.3f}")
    if champ_loglosses:
        print(f"  Log-loss champion (moyenne 3 editions)     : {np.mean(champ_loglosses):.3f}")

    # ── Calibration fine par bin sur 'atteindre les quarts' ──
    print("\n-- Calibration fine : P(atteindre les QUARTS) --")
    p = np.array(pred["QF"]); o = np.array(obs["QF"])
    for lo, hi in [(0, .1), (.1, .25), (.25, .5), (.5, .75), (.75, 1.01)]:
        m = (p >= lo) & (p < hi)
        if m.sum():
            print(f"  [{lo:.2f},{hi:.2f})  pred={p[m].mean():.3f}  obs={o[m].mean():.3f}  n={m.sum()}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Validation de la simulation sur WC passes")
    ap.add_argument("--n-sims", type=int, default=N_SIMS)
    ap.add_argument("--years", type=int, nargs="*", default=None,
                    help="ex: --years 2022  (1 edition = plus rapide)")
    a = ap.parse_args()
    run_validation(n_sims=a.n_sims, years=a.years)
