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

# Force UTF-8 output on Windows (évite UnicodeEncodeError avec PowerShell cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from config import (
    SEED, N_SIMULATIONS, OUTPUTS_DIR, DATA_DIR,
    TRAIN_CUTOFF_YEARS, RIDGE_TEAM, RIDGE_CONF, N_BOOTSTRAP,
    BOOTSTRAP_BLOCK, HOST_TEAMS, PREDICTIONS_LOG,
)
from ingestion.elo_fetch        import fetch_elo
from ingestion.fifa_ranking     import fetch_fifa_rankings
from ingestion.results_fetch    import recent_results
from ingestion.fixtures_fetch   import fetch_fixtures, fetch_wc_teams, update_live_results
from ingestion.players_fetch    import fetch_players
from ingestion.competition_history import build_competition_history, team_summary
from ingestion.confederations import confederation_map

from model.strength_poisson import (
    estimate_strengths, predict_match, save_params, load_params, bootstrap_strengths,
)
from model.calibration import (
    load_calibrator, save_calibrator, train_temporal_calibrator,
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

def _load_pred_log() -> dict:
    """Journal des pronostics gelés, indexé par match_id."""
    if PREDICTIONS_LOG.exists():
        df = pd.read_csv(PREDICTIONS_LOG)
        return {int(r["match_id"]): dict(r) for _, r in df.iterrows()
                if pd.notna(r.get("match_id"))}
    return {}


def _save_pred_log(log: dict) -> None:
    if log:
        pd.DataFrame(list(log.values())).sort_values("match_id").to_csv(
            PREDICTIONS_LOG, index=False)


def generate_match_predictions(fixtures: pd.DataFrame,
                                 params: dict,
                                 history: pd.DataFrame | None = None,
                                 calibrator=None) -> pd.DataFrame:
    """
    Génère matchs.csv avec GEL des pronostics avant match (anti look-ahead).

    Pour chaque match :
      - À VENIR : on (ré)écrit le pronostic candidat dans le journal gelé.
      - JOUÉ : on affiche/évalue le pronostic GELÉ (celui d'avant le coup d'envoi),
        jamais un recalcul avec le modèle actuel (qui a déjà vu le résultat).
        Si aucun pronostic n'avait été figé à temps -> 'graded=False' (non évaluable).
    Seuls les matchs 'graded=True' entrent dans le bilan.
    """
    log = _load_pred_log()
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    rows = []
    for _, row in fixtures.iterrows():
        home  = str(row["home_team"]) if pd.notna(row["home_team"]) else None
        away  = str(row["away_team"]) if pd.notna(row["away_team"]) else None
        if not home or not away or home == "None" or away == "None":
            continue

        stage  = str(row.get("stage", ""))
        grp    = str(row.get("group", ""))
        mid    = int(row["match_id"]) if pd.notna(row.get("match_id")) else None
        has_score = pd.notna(row.get("home_score")) and pd.notna(row.get("away_score"))
        actual = (f"{int(row['home_score'])}-{int(row['away_score'])}"
                  if has_score else None)

        # Pronostic du modèle COURANT (sert pour les matchs à venir, ou l'affichage
        # des matchs joués non gelés ; JAMAIS pour noter un match joué).
        pred = predict_match(home, away, params, neutral=True, host_teams=HOST_TEAMS)
        cur_p1, cur_pn, cur_p2 = pred["prob_home_win"], pred["prob_draw"], pred["prob_away_win"]
        if calibrator is not None:
            cur_p1, cur_pn, cur_p2 = calibrator.transform_one(cur_p1, cur_pn, cur_p2)
        cur_score = f"{pred['most_likely_score'][0]}-{pred['most_likely_score'][1]}"

        if not has_score:
            # Match à venir : (re)gèle le pronostic candidat
            if mid is not None:
                prev_seen = log.get(mid, {}).get("first_seen") if mid in log else None
                log[mid] = {"match_id": mid, "home_team": home, "away_team": away,
                            "date": str(row.get("utc_date", "")),
                            "pred_score": cur_score,
                            "p_home_win": round(cur_p1, 4), "p_draw": round(cur_pn, 4),
                            "p_away_win": round(cur_p2, 4),
                            "first_seen": prev_seen or now, "updated": now}
            d_score, d1, dn, d2 = cur_score, cur_p1, cur_pn, cur_p2
            graded, st = False, "PREDICTED"
        elif mid is not None and mid in log:
            # Match joué AVEC pronostic gelé pré-match -> évaluable
            fz = log[mid]
            d_score = str(fz["pred_score"])
            d1, dn, d2 = float(fz["p_home_win"]), float(fz["p_draw"]), float(fz["p_away_win"])
            graded, st = True, "FINISHED"
        else:
            # Match joué SANS pronostic figé à temps (déployé après) -> non évaluable
            d_score, d1, dn, d2 = cur_score, cur_p1, cur_pn, cur_p2
            graded, st = False, "FINISHED_UNGRADED"

        rows.append({
            "stage": stage, "group": grp, "date": row.get("utc_date", ""),
            "home_team": home, "away_team": away,
            "pred_score": d_score, "actual_score": actual,
            "p_home_win": round(d1, 4), "p_draw": round(dn, 4), "p_away_win": round(d2, 4),
            "exp_home": round(pred["expected_home"], 3),
            "exp_away": round(pred["expected_away"], 3),
            "graded": graded, "status": st,
        })

    _save_pred_log(log)
    df = pd.DataFrame(rows)
    out = OUTPUTS_DIR / "matchs.csv"
    df.to_csv(out, index=False)
    n_frozen = sum(r["graded"] for r in rows)
    print(f"Livrable 1 -> {out}  ({len(df)} matchs, {n_frozen} evalues sur pronostic gele)")
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
    print(f"Livrable 2 -> {out}  ({len(df)} equipes)")
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

    if rows:
        df = pd.DataFrame(rows).sort_values("golden_boot_prob", ascending=False)
    else:
        df = pd.DataFrame(columns=["player", "team", "golden_boot_prob",
                                    "golden_boot_pct", "avg_goals_per_sim",
                                    "goals_last_2y", "goal_rate", "pen_taker"])
    out = OUTPUTS_DIR / "buteurs.csv"
    df.to_csv(out, index=False)
    print(f"Livrable 3 -> {out}  ({len(df)} joueurs)")
    print("  Avertissement : Soulier d'or - prediction tres incertaine (voir README).")
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
        force=force,
    )

    print(f"  {len(elo_df)} équipes Elo | {len(fifa_df)} équipes FIFA | "
          f"{len(fixtures)} matchs | {len(players)} joueurs")

    # ── 2. Estimation du modèle (force unique + bootstrap) ────
    print("\n[2/5] Estimation du modèle (Poisson force unique)...")

    elo_dict  = build_elo_dict(elo_df)
    conf_map  = confederation_map()
    train_df  = recent_results(years=TRAIN_CUTOFF_YEARS, force=force)

    params = load_params()
    if params is None or force:
        params = estimate_strengths(train_df, ridge_team=RIDGE_TEAM, ridge_conf=RIDGE_CONF,
                                    elo_dict=elo_dict, conf_map=conf_map)
        save_params(params)
    else:
        print(f"  Paramètres chargés depuis le cache ({len(params['teams'])} équipes)")

    # Calibrateur isotonique (entraîné sur une fenêtre de validation récente)
    calibrator = load_calibrator()
    if calibrator is None or force:
        print("  Calibration isotonique des probabilités 1/N/2...")
        calibrator = train_temporal_calibrator(train_df, elo_dict=elo_dict,
                                                ridge_team=RIDGE_TEAM, ridge_conf=RIDGE_CONF,
                                                conf_map=conf_map)
        save_calibrator(calibrator)

    # Réplicas bootstrap : propagent l'incertitude d'estimation dans le Monte-Carlo
    param_sets = bootstrap_strengths(train_df, params, B=N_BOOTSTRAP,
                                     ridge_team=RIDGE_TEAM, ridge_conf=RIDGE_CONF,
                                     elo_dict=elo_dict, conf_map=conf_map,
                                     block=BOOTSTRAP_BLOCK, seed=SEED)

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
    matches_df = generate_match_predictions(fixtures, params, history,
                                             calibrator=calibrator)

    # ── 5. Monte-Carlo ────────────────────────────────────────
    if not args.no_sim:
        print(f"\n[5/5] Monte-Carlo ({args.n_sims} simulations)...")
        mc = run_monte_carlo(
            groups=groups,
            params=param_sets,          # réplicas bootstrap -> incertitude propagée
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

    # Dashboard HTML (ne doit jamais faire échouer le run)
    try:
        from dashboard import main as build_dashboard
        build_dashboard()
    except Exception as e:
        print(f"  (dashboard non genere : {e})")

    print("\nTerminé. Fichiers dans outputs/")


if __name__ == "__main__":
    main()
