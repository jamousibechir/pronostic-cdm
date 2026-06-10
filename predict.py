"""
Point d'entrée principal — génère les 3 livrables de la CdM 2026.

Usage :
  python predict.py                  # run complet (télécharge tout, simule)
  python predict.py --update         # met à jour les résultats live, re-simule
  python predict.py --no-sim         # paramètres seulement, pas de Monte-Carlo
  python predict.py --backtest       # lance le backtest avant la prédiction
  python predict.py --n-sims 5000    # override le nombre de simulations

Sorties dans outputs/ :
  matchs.csv    — score prédit + P(1)/P(N)/P(2) pour chaque match
  champion.csv  — probabilité de titre par équipe
  buteurs.csv   — probabilité Soulier d'or par joueur
"""
import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from config import (
    SEED, N_SIMULATIONS, OUTPUTS_DIR, DATA_DIR,
)
from ingestion.elo_fetch        import fetch_elo
from ingestion.fifa_ranking     import fetch_fifa_rankings
from ingestion.results_fetch    import recent_results
from ingestion.fixtures_fetch   import fetch_fixtures, fetch_wc_teams, update_live_results
from ingestion.players_fetch    import fetch_players
from ingestion.competition_history import build_competition_history, team_summary

from model.poisson_dixoncoles import (
    estimate_parameters, predict_match, save_params, load_params, apply_elo_prior,
)
from model.elo import build_elo_dict

from sim.tournament import (
    run_monte_carlo, build_groups_dict, build_played_matches, build_player_weights,
)


def parse_args():
    p = argparse.ArgumentParser(description="Prédiction CdM 2026")
    p.add_argument("--update",   action="store_true",
                   help="Met à jour les résultats live uniquement")
    p.add_argument("--no-sim",   action="store_true",
                   help="Pas de simulation Monte-Carlo")
    p.add_argument("--backtest", action="store_true",
                   help="Lance le backtest avant la prédiction")
    p.add_argument("--force",    action="store_true",
                   help="Re-télécharge toutes les données")
    p.add_argument("--n-sims",   type=int, default=N_SIMULATIONS,
                   help=f"Nombre de simulations (défaut: {N_SIMULATIONS})")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Livrable 1 : scores prédits par match
# ─────────────────────────────────────────────────────────────────────────────

def generate_match_predictions(fixtures: pd.DataFrame,
                                 params: dict,
                                 history: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Génère le CSV matchs.csv avec pour chaque match :
      score prédit (le plus probable) + P(1)/P(N)/P(2) + buts attendus
    Les matchs déjà joués ont leurs vrais scores figés.
    """
    rows = []
    for _, row in fixtures.iterrows():
        home  = str(row["home_team"]) if pd.notna(row["home_team"]) else None
        away  = str(row["away_team"]) if pd.notna(row["away_team"]) else None
        if not home or not away or home == "None" or away == "None":
            continue

        status = str(row.get("status", "SCHEDULED"))
        stage  = str(row.get("stage", ""))
        grp    = str(row.get("group", ""))

        if status == "FINISHED":
            # Match joué : on conserve le vrai score
            rows.append({
                "stage":        stage,
                "group":        grp,
                "date":         row.get("utc_date", ""),
                "home_team":    home,
                "away_team":    away,
                "pred_score":   f"{int(row['home_score'])}-{int(row['away_score'])}",
                "actual_score": f"{int(row['home_score'])}-{int(row['away_score'])}",
                "p_home_win":   None,
                "p_draw":       None,
                "p_away_win":   None,
                "exp_home":     None,
                "exp_away":     None,
                "status":       "FINISHED",
            })
            continue

        # Match à venir : prédiction DC
        neutral = True  # CdM = terrain neutre
        pred = predict_match(home, away, params, neutral=neutral)

        rows.append({
            "stage":        stage,
            "group":        grp,
            "date":         row.get("utc_date", ""),
            "home_team":    home,
            "away_team":    away,
            "pred_score":   f"{pred['most_likely_score'][0]}-{pred['most_likely_score'][1]}",
            "actual_score": None,
            "p_home_win":   round(pred["prob_home_win"], 4),
            "p_draw":       round(pred["prob_draw"],     4),
            "p_away_win":   round(pred["prob_away_win"], 4),
            "exp_home":     round(pred["expected_home"], 3),
            "exp_away":     round(pred["expected_away"], 3),
            "status":       "PREDICTED",
        })

    df = pd.DataFrame(rows)
    out = OUTPUTS_DIR / "matchs.csv"
    df.to_csv(out, index=False)
    print(f"Livrable 1 → {out}  ({len(df)} matchs)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Livrable 2 : probabilités de titre
# ─────────────────────────────────────────────────────────────────────────────

def generate_champion_csv(mc_results: dict,
                           history: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Génère champion.csv enrichi avec le palmarès historique.
    """
    probs = mc_results["champion_probs"]
    rows = []
    for team, prob in probs.items():
        row = {
            "team":         team,
            "win_prob":     round(prob, 5),
            "win_pct":      f"{prob * 100:.2f}%",
        }
        if history is not None:
            s = team_summary(team, history)
            row["wc_titles"]       = s.get("World Cup titles",      0)
            row["wc_finals"]       = s.get("World Cup finals",       0)
            row["wc_semi_finals"]  = s.get("World Cup semi-finals",  0)
            row["euro_titles"]     = s.get("Euro titles",            0)
            row["copa_titles"]     = s.get("Copa America titles",    0)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("win_prob", ascending=False)
    out = OUTPUTS_DIR / "champion.csv"
    df.to_csv(out, index=False)
    print(f"Livrable 2 → {out}  ({len(df)} équipes)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Livrable 3 : Soulier d'or
# ─────────────────────────────────────────────────────────────────────────────

def generate_golden_boot_csv(mc_results: dict,
                               players: pd.DataFrame) -> pd.DataFrame:
    """
    Génère buteurs.csv avec la probabilité de Soulier d'or par joueur.

    NOTE : Cette prédiction est la plus bruitée du système.
    La répartition des buts repose sur des taux historiques en sélection
    et ne modélise pas les blessures, suspensions ni le temps de jeu exact.
    """
    boot_probs  = mc_results["golden_boot"]
    avg_goals   = mc_results.get("avg_goals", {})

    # Enrichit avec les données joueurs
    player_info = {}
    if not players.empty:
        for _, row in players.iterrows():
            player_info[str(row["player"])] = {
                "team":         str(row["team"]),
                "goals_recent": int(row.get("goals_recent", 0)),
                "goal_rate":    round(float(row.get("goal_rate", 0)), 3),
                "is_pen_taker": bool(row.get("is_pen_taker", False)),
            }

    rows = []
    for player, prob in boot_probs.items():
        info = player_info.get(player, {})
        rows.append({
            "player":       player,
            "team":         info.get("team", "?"),
            "golden_boot_prob": round(prob, 5),
            "golden_boot_pct":  f"{prob * 100:.2f}%",
            "avg_goals_per_sim": round(avg_goals.get(player, 0), 2),
            "goals_last_2y":    info.get("goals_recent", "?"),
            "goal_rate":        info.get("goal_rate", "?"),
            "pen_taker":        info.get("is_pen_taker", "?"),
        })

    df = pd.DataFrame(rows).sort_values("golden_boot_prob", ascending=False)
    out = OUTPUTS_DIR / "buteurs.csv"
    df.to_csv(out, index=False)
    print(f"Livrable 3 → {out}  ({len(df)} joueurs)")
    print("  ⚠ Soulier d'or : prédiction très incertaine (voir README).")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration principale
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    np.random.seed(SEED)

    print("=" * 60)
    print("PRONOSTIC CdM 2026 — Démarrage")
    print("=" * 60)

    # ── 0. Backtest optionnel ──────────────────────────────────
    if args.backtest:
        from backtest import run_backtest
        run_backtest(verbose=True)
        print()

    # ── 1. Ingestion données ──────────────────────────────────
    force = args.force
    print("\n[1/5] Ingestion des données...")

    elo_df   = fetch_elo(force=force)
    fifa_df  = fetch_fifa_rankings(force=force)

    if args.update:
        fixtures = update_live_results()
    else:
        fixtures  = fetch_fixtures(force=force)
    wc_teams_df = fetch_wc_teams(force=force)

    # Liste des équipes qualifiées
    wc_team_names = wc_teams_df["team"].dropna().unique().tolist()
    if not wc_team_names:
        # Fallback depuis les fixtures
        gs = fixtures[fixtures["stage"].str.contains("GROUP", na=False, case=False)]
        wc_team_names = list(
            set(gs["home_team"].dropna().tolist() + gs["away_team"].dropna().tolist())
        )

    players = fetch_players(
        wc_teams=wc_team_names,
        enrich_fbref=True,
        force=force,
    )

    print(f"  {len(elo_df)} équipes Elo | {len(fifa_df)} équipes FIFA | "
          f"{len(fixtures)} matchs | {len(players)} joueurs")

    # ── 2. Estimation du modèle ───────────────────────────────
    print("\n[2/5] Estimation du modèle Dixon-Coles...")

    params = load_params()
    if params is None or force:
        train_df = recent_results(force=force)
        params   = estimate_parameters(train_df)

        # Enrichit avec prior Elo pour les équipes peu représentées
        elo_dict = build_elo_dict(elo_df)
        params   = apply_elo_prior(params, elo_dict, weight=0.25)

        save_params(params)
    else:
        print(f"  Paramètres chargés depuis le cache ({len(params['teams'])} équipes)")

    # ── 3. Construction des structures de tournoi ─────────────
    print("\n[3/5] Construction du bracket CdM 2026...")
    groups       = build_groups_dict(fixtures, wc_teams_df)
    played       = build_played_matches(fixtures)
    pl_weights   = build_player_weights(players)

    n_groups = len(groups)
    n_played = len(played)
    print(f"  {n_groups} groupes | {n_played} matchs déjà joués | "
          f"{sum(len(v) for v in pl_weights.values())} joueurs pondérés")

    if n_groups == 0:
        print("  ERREUR : aucun groupe trouvé. Vérifiez la réponse API football-data.org.")
        print("  Utilisez --force pour re-télécharger les fixtures.")
        sys.exit(1)

    # ── 4. Livrable 1 : scores par match ──────────────────────
    print("\n[4/5] Prédiction des scores...")
    history = build_competition_history(force=force)
    matches_df = generate_match_predictions(fixtures, params, history)

    # ── 5. Monte-Carlo ────────────────────────────────────────
    if not args.no_sim:
        print(f"\n[5/5] Monte-Carlo ({args.n_sims} simulations)...")
        mc = run_monte_carlo(
            groups=groups,
            params=params,
            played_matches=played,
            player_weights=pl_weights,
            n_sims=args.n_sims,
            seed=SEED,
        )

        # Livrables 2 & 3
        champion_df = generate_champion_csv(mc, history=history)
        boot_df     = generate_golden_boot_csv(mc, players=players)

        # ── Résumé console ─────────────────────────────────────
        print("\n" + "=" * 60)
        print("RÉSULTATS — Top 10 favoris au titre")
        print("=" * 60)
        print(champion_df[["team", "win_pct"]].head(10).to_string(index=False))

        print("\nTop 10 favoris Soulier d'or")
        print("-" * 40)
        print(boot_df[["player", "team", "golden_boot_pct",
                        "avg_goals_per_sim"]].head(10).to_string(index=False))
    else:
        print("\n[5/5] Mode --no-sim : simulation ignorée.")

    print("\nTerminé. Fichiers dans outputs/")


if __name__ == "__main__":
    main()
