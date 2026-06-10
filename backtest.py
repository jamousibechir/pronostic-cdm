"""
Backtest et calibration du modèle Dixon-Coles.

Protocole :
  - Entraînement : matchs internationaux jusqu'à septembre 2024
  - Test         : matchs internationaux octobre 2024 – mai 2026
    (inclut UEFA Nations League, qualifications, amicaux, AFCON 2025,
     Copa América 2024, Euros 2024, etc.)
  - Métriques calculées :
      Brier score     : qualité de la prédiction 1/N/2 (plus bas = meilleur)
      Log-loss        : calibration probabiliste
      RMSE buts       : erreur sur le score prédit
      Accuracy 1/N/2  : taux de bonne prédiction de l'issue
      Calibration plot: intervalles de confiance vs fréquences observées

Lancer : python backtest.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

from config import DATA_DIR, OUTPUTS_DIR, SEED, HALF_LIFE_DAYS
from ingestion.results_fetch import fetch_results
from model.poisson_dixoncoles import (
    estimate_parameters, predict_match, save_params,
)

TRAIN_END   = "2024-09-30"
TEST_START  = "2024-10-01"
TEST_END    = "2026-05-31"
BACKTEST_OUT = OUTPUTS_DIR / "backtest_metrics.csv"


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Brier score multi-classe. probs.shape = (N, 3), outcomes.shape = (N,) ∈ {0,1,2}."""
    one_hot = np.zeros_like(probs)
    for i, o in enumerate(outcomes):
        one_hot[i, int(o)] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def log_loss(probs: np.ndarray, outcomes: np.ndarray,
             eps: float = 1e-9) -> float:
    """Log-loss multi-classe."""
    n = len(outcomes)
    ll = 0.0
    for i, o in enumerate(outcomes):
        ll += np.log(max(probs[i, int(o)], eps))
    return -ll / n


def accuracy(probs: np.ndarray, outcomes: np.ndarray) -> float:
    preds = np.argmax(probs, axis=1)
    return float(np.mean(preds == outcomes))


def run_backtest(train_years: int = 6,
                 verbose: bool = True) -> pd.DataFrame:
    """
    Lance le backtest complet.

    Returns
    -------
    pd.DataFrame avec les métriques globales + par compétition
    """
    print("=" * 60)
    print("BACKTEST — Calibration du modèle Dixon-Coles")
    print("=" * 60)

    results = fetch_results()

    train_df = results[results["date"] <= TRAIN_END].copy()
    test_df  = results[
        (results["date"] >= TEST_START) &
        (results["date"] <= TEST_END)
    ].copy()

    print(f"  Entraînement : {len(train_df)} matchs (jusqu'à {TRAIN_END})")
    print(f"  Test         : {len(test_df)} matchs ({TEST_START} – {TEST_END})")

    # Estimation des paramètres sur train
    print("\nEstimation des paramètres Dixon-Coles...")
    ref_date = pd.Timestamp(TRAIN_END)
    params = estimate_parameters(train_df, ref_date=ref_date)
    save_params(params)
    print(f"  {len(params['teams'])} équipes modélisées")
    print(f"  gamma (avantage domicile) = {params['gamma']:.3f}")
    print(f"  rho (correction DC)       = {params['rho']:.4f}")

    # Prédictions sur le jeu de test
    print("\nPrédictions sur le jeu de test...")
    rows = []
    probs_list = []
    outcomes_list = []
    home_pred_list = []
    away_pred_list = []
    home_actual_list = []
    away_actual_list = []

    skipped = 0
    for _, row in test_df.iterrows():
        home = str(row["home_team"])
        away = str(row["away_team"])

        # Saute si l'équipe est inconnue du modèle
        if (home not in params["alpha"] or away not in params["alpha"]):
            skipped += 1
            continue

        pred = predict_match(home, away, params,
                              neutral=bool(row["neutral"]))

        p_hw = pred["prob_home_win"]
        p_d  = pred["prob_draw"]
        p_aw = pred["prob_away_win"]

        hs = int(row["home_score"])
        as_ = int(row["away_score"])

        if hs > as_:
            outcome = 0   # home win
        elif hs == as_:
            outcome = 1   # draw
        else:
            outcome = 2   # away win

        probs_list.append([p_hw, p_d, p_aw])
        outcomes_list.append(outcome)
        home_pred_list.append(pred["expected_home"])
        away_pred_list.append(pred["expected_away"])
        home_actual_list.append(hs)
        away_actual_list.append(as_)

        rows.append({
            "date":       row["date"],
            "tournament": row.get("tournament", ""),
            "home_team":  home,
            "away_team":  away,
            "home_actual": hs,
            "away_actual": as_,
            "home_pred":  pred["expected_home"],
            "away_pred":  pred["expected_away"],
            "most_likely": f"{pred['most_likely_score'][0]}-{pred['most_likely_score'][1]}",
            "p_home_win": p_hw,
            "p_draw":     p_d,
            "p_away_win": p_aw,
            "outcome":    ["H", "D", "A"][outcome],
            "correct":    ["H", "D", "A"][int(np.argmax([p_hw, p_d, p_aw]))] == ["H", "D", "A"][outcome],
        })

    print(f"  Matchs prédits : {len(rows)}  |  ignorés (équipe inconnue) : {skipped}")

    probs_arr    = np.array(probs_list)
    outcomes_arr = np.array(outcomes_list)
    h_pred = np.array(home_pred_list)
    a_pred = np.array(away_pred_list)
    h_act  = np.array(home_actual_list)
    a_act  = np.array(away_actual_list)

    bs  = brier_score(probs_arr, outcomes_arr)
    ll  = log_loss(probs_arr, outcomes_arr)
    acc = accuracy(probs_arr, outcomes_arr)
    rmse_h = float(np.sqrt(np.mean((h_pred - h_act) ** 2)))
    rmse_a = float(np.sqrt(np.mean((a_pred - a_act) ** 2)))

    print("\n── Métriques globales ──────────────────────────────────")
    print(f"  Brier score      : {bs:.4f}  (référence naïf 1/3 = 0.667)")
    print(f"  Log-loss         : {ll:.4f}")
    print(f"  Accuracy 1/N/2   : {acc:.3f}  ({acc*100:.1f}%)")
    print(f"  RMSE buts dom.   : {rmse_h:.3f}")
    print(f"  RMSE buts ext.   : {rmse_a:.3f}")

    # ── Calibration par quantile de probabilité ──
    print("\n── Calibration P(victoire domicile) ────────────────────")
    calibration_rows = []
    p_home = probs_arr[:, 0]
    bins = np.linspace(0, 1, 11)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_home >= lo) & (p_home < hi)
        if mask.sum() == 0:
            continue
        mean_pred = float(p_home[mask].mean())
        mean_obs  = float((outcomes_arr[mask] == 0).mean())
        n         = int(mask.sum())
        calibration_rows.append({
            "bin": f"[{lo:.1f},{hi:.1f})",
            "mean_pred": mean_pred,
            "mean_obs":  mean_obs,
            "n":         n,
            "error":     abs(mean_pred - mean_obs),
        })
        print(f"  {f'[{lo:.1f},{hi:.1f})':<12}  prédit={mean_pred:.3f}  observé={mean_obs:.3f}  n={n}")

    cal_df = pd.DataFrame(calibration_rows)
    mean_cal_error = float(cal_df["error"].mean()) if not cal_df.empty else np.nan
    print(f"\n  Erreur de calibration moyenne : {mean_cal_error:.4f}")

    # ── Métriques par compétition ────────────────────────────────
    details_df = pd.DataFrame(rows)
    if not details_df.empty:
        print("\n── Accuracy par compétition (top 10) ───────────────────")
        comp_stats = (
            details_df.groupby("tournament")["correct"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "accuracy", "count": "n_matches"})
            .sort_values("n_matches", ascending=False)
            .head(10)
        )
        print(comp_stats.to_string())

    # ── Sauvegarde ─────────────────────────────────────────────
    metrics = pd.DataFrame([{
        "brier_score":    bs,
        "log_loss":       ll,
        "accuracy":       acc,
        "rmse_home":      rmse_h,
        "rmse_away":      rmse_a,
        "cal_error_mean": mean_cal_error,
        "n_matches":      len(rows),
        "train_cutoff":   TRAIN_END,
        "test_start":     TEST_START,
    }])
    metrics.to_csv(BACKTEST_OUT, index=False)
    print(f"\nMétriques sauvegardées → {BACKTEST_OUT}")

    if verbose and not details_df.empty:
        details_path = OUTPUTS_DIR / "backtest_details.csv"
        details_df.to_csv(details_path, index=False)
        print(f"Détails match par match → {details_path}")

    return metrics


if __name__ == "__main__":
    run_backtest(verbose=True)
