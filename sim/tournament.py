"""
Simulation Monte-Carlo de la Coupe du Monde 2026.

Format du tournoi :
  48 équipes / 12 groupes de 4 / 3 journées de groupes
  → Top 2 de chaque groupe (24) + 8 meilleurs 3es = 32 équipes en seizièmes
  → Tableau à élimination directe : R32 → R16 → QF → SF → Finale
  Prolongations + tirs au but si égalité après 90 min (phase KO uniquement)

Bracket R32 (hypothèse documentée dans README) :
  Groupes A-H : 12 matchs GW vs RU cross-groupes (matches 1-12)
  Groupes I-L :  4 matchs GW vs meilleur 3e (matches 13-16)
               + 4 matchs RU  vs meilleur 3e (matches 13-16)
  Les 8 meilleurs 3es sont injectés dans les slots prédéfinis.
  NOTE : le bracket exact sera mis à jour depuis football-data.org quand
  les matchs KO seront publiés avec les slots officiels.
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from config import SEED, N_SIMULATIONS, N_THIRD_ADVANCE, MAX_GOALS
from model.poisson_dixoncoles import predict_match, score_matrix

# ── Ordre de préférence des critères de classement FIFA (phase de groupes) ───
# 1. Points  2. Diff buts  3. Buts marqués  4. H2H points  5. H2H diff buts
# 6. Fair-play  7. Tirage au sort (simulé aléatoirement)

# ── Structure du bracket R32 ─────────────────────────────────────────────────
# Format : ("source_type", "group", seed_rank)
# source_type : "W" = winner, "R" = runner-up, "T" = best third
# seed_rank   : pour "T", indice dans la liste des 8 meilleurs 3es (0-based)

R32_PAIRINGS = [
    # Groupe A-H : GW vs RU cross
    (("W", "A"), ("R", "C")),  # M1
    (("W", "B"), ("R", "D")),  # M2
    (("W", "C"), ("R", "A")),  # M3
    (("W", "D"), ("R", "B")),  # M4
    (("W", "E"), ("R", "G")),  # M5
    (("W", "F"), ("R", "H")),  # M6
    (("W", "G"), ("R", "E")),  # M7
    (("W", "H"), ("R", "F")),  # M8
    # Groupe I-L : GW vs meilleur 3e, RU vs meilleur 3e
    (("W", "I"), ("T", 0)),    # M9
    (("W", "J"), ("T", 1)),    # M10
    (("W", "K"), ("T", 2)),    # M11
    (("W", "L"), ("T", 3)),    # M12
    (("R", "I"), ("T", 4)),    # M13
    (("R", "J"), ("T", 5)),    # M14
    (("R", "K"), ("T", 6)),    # M15
    (("R", "L"), ("T", 7)),    # M16
]

# R16 : matches 2 par 2 dans l'ordre du bracket
# Winner M1 vs Winner M2, etc.
R16_PAIRINGS  = [(i, i+1) for i in range(0, 16, 2)]   # (0,1),(2,3),...,(14,15)
QF_PAIRINGS   = [(i, i+1) for i in range(0, 8,  2)]   # winners of R16
SF_PAIRINGS   = [(0, 1), (2, 3)]
FINAL_PAIRING = (0, 1)
THIRD_PAIRING = (0, 1)   # perdants SF1 vs perdants SF2

# Paramètres prolongations / tirs au but
ET_LAMBDA_FACTOR = 0.35   # les buts en prolongation ~ 35 % des buts sur 90 min
SHOOTOUT_P_WIN   = 0.50   # 50/50 aux t.a.b.


# ─────────────────────────────────────────────────────────────────────────────
# Simulation d'un match simple (90 min, renvoie les buts)
# ─────────────────────────────────────────────────────────────────────────────

def _sample_score(home: str, away: str, params: dict,
                  neutral: bool, rng: np.random.Generator) -> tuple[int, int]:
    """Tire un score depuis la matrice de probabilité Dixon-Coles."""
    alpha_h = params["alpha"].get(home, 1.0)
    beta_h  = params["beta"].get(home,  1.0)
    alpha_a = params["alpha"].get(away, 1.0)
    beta_a  = params["beta"].get(away,  1.0)
    rho     = params["rho"]
    gamma   = 1.0 if neutral else params["gamma"]

    mat = score_matrix(alpha_h, beta_h, alpha_a, beta_a, rho, gamma, MAX_GOALS)
    flat_probs = mat.ravel()
    flat_probs = np.maximum(flat_probs, 0)
    flat_probs /= flat_probs.sum()

    idx = rng.choice(len(flat_probs), p=flat_probs)
    h, a = divmod(idx, MAX_GOALS + 1)
    return int(h), int(a)


def _ko_match(home: str, away: str, params: dict,
              rng: np.random.Generator) -> tuple[str, int, int]:
    """
    Simule un match KO (terrain neutre). Prolongations + tirs au but si égalité.
    Retourne (winner, goals_home, goals_away) au score final.
    """
    h, a = _sample_score(home, away, params, neutral=True, rng=rng)

    if h != a:
        winner = home if h > a else away
        return winner, h, a

    # Prolongations (30 min ≈ 35 % des buts sur 90 min)
    lam_et = params["alpha"].get(home, 1.0) * params["beta"].get(away, 1.0) * ET_LAMBDA_FACTOR
    mu_et  = params["alpha"].get(away, 1.0) * params["beta"].get(home, 1.0) * ET_LAMBDA_FACTOR
    h_et = int(rng.poisson(lam_et))
    a_et = int(rng.poisson(mu_et))
    h += h_et
    a += a_et

    if h != a:
        winner = home if h > a else away
        return winner, h, a

    # Tirs au but (50/50)
    winner = home if rng.random() < SHOOTOUT_P_WIN else away
    return winner, h, a


# ─────────────────────────────────────────────────────────────────────────────
# Phase de groupes
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_group(teams: list[str], params: dict,
                    rng: np.random.Generator,
                    played: dict | None = None) -> pd.DataFrame:
    """
    Simule les matchs d'un groupe, en respectant les résultats déjà joués.

    played : dict {(home, away): (home_score, away_score)} pour les matchs fixés.

    Returns : DataFrame classement final du groupe
      colonnes : team, pts, gf, ga, gd, wins, draws, losses
    """
    if played is None:
        played = {}

    stats = {t: dict(pts=0, gf=0, ga=0, gd=0, wins=0, draws=0, losses=0)
             for t in teams}

    # Génère les 6 matchs du groupe (round-robin)
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            home, away = teams[i], teams[j]
            key = (home, away)
            if key in played:
                h, a = played[key]
            else:
                h, a = _sample_score(home, away, params, neutral=True, rng=rng)

            # Mise à jour des stats
            stats[home]["gf"] += h
            stats[home]["ga"] += a
            stats[away]["gf"] += a
            stats[away]["ga"] += h
            stats[home]["gd"] += h - a
            stats[away]["gd"] += a - h

            if h > a:
                stats[home]["pts"]   += 3
                stats[home]["wins"]  += 1
                stats[away]["losses"] += 1
            elif h < a:
                stats[away]["pts"]   += 3
                stats[away]["wins"]  += 1
                stats[home]["losses"] += 1
            else:
                stats[home]["pts"]    += 1
                stats[away]["pts"]    += 1
                stats[home]["draws"]  += 1
                stats[away]["draws"]  += 1

    df = pd.DataFrame.from_dict(stats, orient="index").reset_index()
    df = df.rename(columns={"index": "team"})

    # Classement : pts > gd > gf > aléatoire (simule tirage au sort)
    df["random_tiebreak"] = rng.random(len(df))
    df = df.sort_values(
        ["pts", "gd", "gf", "random_tiebreak"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    df["group_rank"] = df.index + 1

    return df.drop(columns=["random_tiebreak"])


def _rank_third_place_teams(thirds: list[dict]) -> list[dict]:
    """
    Classe les 12 troisièmes équipes, retourne les 8 meilleures.
    Critères FIFA : points > GD > GF > fair-play (non simulé → aléatoire).
    """
    return sorted(
        thirds,
        key=lambda x: (x["pts"], x["gd"], x["gf"]),
        reverse=True,
    )[:N_THIRD_ADVANCE]


# ─────────────────────────────────────────────────────────────────────────────
# Construction du bracket KO
# ─────────────────────────────────────────────────────────────────────────────

def _build_r32(group_results: dict[str, pd.DataFrame],
               rng: np.random.Generator) -> list[tuple[str, str]]:
    """
    Construit les 16 paires R32 à partir des résultats de groupes.
    group_results : {group_letter: DataFrame classement}
    """
    winners    = {g: df.iloc[0]["team"] for g, df in group_results.items()}
    runners_up = {g: df.iloc[1]["team"] for g, df in group_results.items()}

    # Construction de la liste des meilleurs 3es
    thirds = []
    for g, df in group_results.items():
        row = df.iloc[2]
        thirds.append({
            "team": row["team"],
            "group": g,
            "pts":  row["pts"],
            "gd":   row["gd"],
            "gf":   row["gf"],
        })
    best_thirds  = _rank_third_place_teams(thirds)
    thirds_teams = [t["team"] for t in best_thirds]
    while len(thirds_teams) < 8:
        thirds_teams.append("TBD")

    def _resolve(slot: tuple) -> str:
        slot_type, slot_val = slot
        if slot_type == "W":
            return winners.get(slot_val, "TBD")
        elif slot_type == "R":
            return runners_up.get(slot_val, "TBD")
        else:  # "T" — slot_val is the rank index (int)
            idx = int(slot_val)
            return thirds_teams[idx] if idx < len(thirds_teams) else "TBD"

    return [(_resolve(home_slot), _resolve(away_slot))
            for home_slot, away_slot in R32_PAIRINGS]


# ─────────────────────────────────────────────────────────────────────────────
# Un run complet du tournoi
# ─────────────────────────────────────────────────────────────────────────────

def _run_tournament(groups: dict[str, list[str]],
                    params: dict,
                    rng: np.random.Generator,
                    played_matches: dict | None = None,
                    player_weights: dict | None = None
                    ) -> tuple[str, dict[str, int]]:
    """
    Simule un tournoi complet.

    Parameters
    ----------
    groups        : {group_letter: [team1, team2, team3, team4]}
    params        : paramètres Dixon-Coles
    rng           : générateur aléatoire
    played_matches: {(home, away): (h_score, a_score)} résultats réels
    player_weights: {team: {player: weight}} pour la simulation Soulier d'or

    Returns
    -------
    (champion, player_goals_dict)
    player_goals_dict : {player: int} buts cumulés sur ce run
    """
    if played_matches is None:
        played_matches = {}
    if player_weights is None:
        player_weights = {}

    # ── Phase de groupes ──
    group_results: dict[str, pd.DataFrame] = {}
    for g, teams in groups.items():
        # Filtre les matchs déjà joués pour ce groupe
        played_g = {k: v for k, v in played_matches.items()
                    if k[0] in teams and k[1] in teams}
        group_results[g] = _simulate_group(teams, params, rng, played_g)

    # ── Bracket R32 ──
    r32_pairs = _build_r32(group_results, rng)

    # ── Phases KO ──
    current_round = r32_pairs
    losers_sf: list[str] = []
    player_goals: dict[str, int] = defaultdict(int)

    def _play_round(pairs, round_name=""):
        nonlocal losers_sf
        next_round = []
        for home, away in pairs:
            winner, gh, ga = _ko_match(home, away, params, rng)
            loser = away if winner == home else home

            # Accumule les buts pour Soulier d'or
            _distribute_goals(home, gh, player_weights, player_goals, rng)
            _distribute_goals(away, ga, player_weights, player_goals, rng)

            if round_name == "SF":
                losers_sf.append(loser)
            next_round.append(winner)
        return next_round

    r16_winners   = _play_round(current_round, "R32")
    r16_pairs     = [(r16_winners[i], r16_winners[i+1])
                     for i in range(0, len(r16_winners), 2)]
    qf_winners    = _play_round(r16_pairs, "R16")
    qf_pairs      = [(qf_winners[i], qf_winners[i+1])
                     for i in range(0, len(qf_winners), 2)]
    sf_winners    = _play_round(qf_pairs, "QF")
    sf_pairs      = [(sf_winners[i], sf_winners[i+1])
                     for i in range(0, len(sf_winners), 2)]
    finalists     = _play_round(sf_pairs, "SF")

    # Match pour la 3e place
    if len(losers_sf) >= 2:
        _play_round([(losers_sf[0], losers_sf[1])], "3P")

    # Finale
    final_pair = [(finalists[0], finalists[1])]
    champion_list = _play_round(final_pair, "F")
    champion = champion_list[0]

    return champion, dict(player_goals)


def _distribute_goals(team: str, goals: int,
                       player_weights: dict,
                       player_goals: dict,
                       rng: np.random.Generator) -> None:
    """
    Distribue `goals` buts entre les joueurs d'une équipe selon leurs poids.
    Si l'équipe est inconnue, tous les buts vont dans un bucket "_unknown".
    """
    if goals == 0:
        return
    weights_dict = player_weights.get(team)
    if not weights_dict:
        player_goals[f"_unknown_{team}"] += goals
        return

    players = list(weights_dict.keys())
    weights = np.array(list(weights_dict.values()), dtype=float)
    weights = np.maximum(weights, 0)
    total_w = weights.sum()
    if total_w == 0:
        player_goals[f"_unknown_{team}"] += goals
        return
    weights /= total_w

    for _ in range(goals):
        scorer = rng.choice(players, p=weights)
        player_goals[scorer] += 1


# ─────────────────────────────────────────────────────────────────────────────
# Monte-Carlo principal
# ─────────────────────────────────────────────────────────────────────────────

def run_monte_carlo(groups: dict[str, list[str]],
                    params: dict,
                    played_matches: dict | None = None,
                    player_weights: dict | None = None,
                    n_sims: int = N_SIMULATIONS,
                    seed: int = SEED,
                    ) -> dict:
    """
    Lance n_sims simulations du tournoi et agrège les probabilités.

    Returns
    -------
    dict avec :
      champion_probs  : {team: float} proba de titre
      finalist_probs  : {team: float} proba d'atteindre la finale
      sf_probs        : {team: float} proba de demi-finale
      qf_probs        : {team: float} proba de QF
      r16_probs       : {team: float} proba de R16
      golden_boot     : {player: float} proba Soulier d'or
    """
    rng = np.random.default_rng(seed)

    champion_count:  defaultdict[str, int] = defaultdict(int)
    finalist_count:  defaultdict[str, int] = defaultdict(int)
    sf_count:        defaultdict[str, int] = defaultdict(int)
    qf_count:        defaultdict[str, int] = defaultdict(int)
    r16_count:       defaultdict[str, int] = defaultdict(int)
    player_total_goals: defaultdict[str, int] = defaultdict(int)
    player_top_scorer:  defaultdict[str, int] = defaultdict(int)

    all_teams = [t for tl in groups.values() for t in tl]

    for _ in tqdm(range(n_sims), desc="Monte-Carlo simulations", unit="run"):
        champion, p_goals = _run_tournament(
            groups, params, rng, played_matches, player_weights
        )
        champion_count[champion] += 1

        # Soulier d'or : meilleur buteur du run
        if p_goals:
            real_goals = {k: v for k, v in p_goals.items()
                          if not k.startswith("_unknown")}
            if real_goals:
                top_scorer = max(real_goals, key=real_goals.get)
                player_top_scorer[top_scorer] += 1
            for player, g in real_goals.items():
                player_total_goals[player] += g

    def normalize(counter: dict) -> dict:
        total = sum(counter.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in
                sorted(counter.items(), key=lambda x: x[1], reverse=True)}

    return {
        "champion_probs":  normalize(champion_count),
        "golden_boot":     normalize(player_top_scorer),
        "avg_goals":       {k: v / n_sims for k, v in player_total_goals.items()},
        "n_sims":          n_sims,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers pour construire les structures d'entrée depuis les DataFrames
# ─────────────────────────────────────────────────────────────────────────────

def build_groups_dict(fixtures: pd.DataFrame,
                       wc_teams: pd.DataFrame) -> dict[str, list[str]]:
    """
    Construit {group_letter: [team1, team2, team3, team4]} depuis les DataFrames.
    """
    if "group" not in wc_teams.columns or wc_teams["group"].isna().all():
        # Fallback : extrait les groupes depuis les fixtures phase de groupes
        gs = fixtures[fixtures["stage"].str.contains("GROUP", na=False, case=False)]
        groups: dict[str, list[str]] = {}
        for _, row in gs.iterrows():
            grp = str(row.get("group", "")).replace("GROUP_", "").strip()
            if not grp:
                continue
            h, a = str(row["home_team"]), str(row["away_team"])
            if grp not in groups:
                groups[grp] = []
            for t in [h, a]:
                if t and t not in groups[grp]:
                    groups[grp].append(t)
        return groups

    groups = {}
    for _, row in wc_teams.iterrows():
        grp = str(row["group"]).replace("GROUP_", "").strip()
        if not grp or grp.lower() == "nan":
            continue
        if grp not in groups:
            groups[grp] = []
        team = str(row["team"])
        if team not in groups[grp]:
            groups[grp].append(team)
    return groups


def build_played_matches(fixtures: pd.DataFrame) -> dict:
    """
    Extrait les matchs déjà joués (status=FINISHED) pour les figer dans la simulation.
    """
    finished = fixtures[
        (fixtures["status"] == "FINISHED") &
        fixtures["home_score"].notna() &
        fixtures["away_score"].notna()
    ]
    return {
        (row["home_team"], row["away_team"]): (int(row["home_score"]), int(row["away_score"]))
        for _, row in finished.iterrows()
    }


def build_player_weights(players: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    Construit {team: {player: weight}} pour la distribution des buts.
    weight = goal_rate * start_prob (+ bonus pénaltys)
    """
    weights: dict[str, dict[str, float]] = {}
    for _, row in players.iterrows():
        team   = str(row["team"])
        player = str(row["player"])
        w      = float(row["goal_rate"]) * float(row["start_prob"])
        # Bonus de 30 % pour les tireurs de pénos
        if row.get("is_pen_taker", 0):
            w *= 1.3
        if team not in weights:
            weights[team] = {}
        weights[team][player] = max(w, 0.0)
    return weights
