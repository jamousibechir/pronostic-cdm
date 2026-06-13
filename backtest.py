"""
Backtest et calibration du modèle de buts à force unique.

Protocole (sans fuite) :
  - Entraînement : matchs <= 2024-09-30 (fenêtre glissante de `train_years`)
  - Test         : 2024-10-01 -> 2026-05-31 (hors échantillon)
  - Le ridge (shrinkage) est réglé sur une sous-validation temporelle DU TRAIN,
    jamais sur le test.
  - La calibration isotonique est ajustée sur une fenêtre de validation du train,
    puis appliquée au test (métriques brutes ET calibrées rapportées).
  - AUCUNE équipe n'est ignorée (le modèle gère les inconnues par repli Elo / force 0),
    contrairement à l'ancien Dixon-Coles qui jetait ~23 % du test.

Métriques : Brier, log-loss, accuracy 1/N/2, RMSE buts, courbe de calibration.
"""
import numpy as np
import pandas as pd

from config import OUTPUTS_DIR, RIDGE_TEAM, RIDGE_CONF
from ingestion.results_fetch import fetch_results
from ingestion.elo_fetch import fetch_elo
from ingestion.confederations import confederation_map
from model.elo import build_elo_dict
from model.strength_poisson import estimate_strengths, predict_match
from model.calibration import MultiClassCalibrator

TRAIN_END   = "2024-09-30"
TEST_START  = "2024-10-01"
TEST_END    = "2026-05-31"
TRAIN_YEARS = 10
# Grille log-restreinte 2-D : (shrinkage équipe->confédération, confédération->global)
RIDGE_TEAM_GRID = [0.5, 2.0, 8.0]
RIDGE_CONF_GRID = [0.1, 1.0, 10.0]
BACKTEST_OUT = OUTPUTS_DIR / "backtest_metrics.csv"


# ── Métriques ─────────────────────────────────────────────────────────────────

def brier_score(probs, outcomes):
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(outcomes)), outcomes] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def log_loss(probs, outcomes, eps=1e-9):
    p = np.clip(probs[np.arange(len(outcomes)), outcomes], eps, 1)
    return float(-np.mean(np.log(p)))


def accuracy(probs, outcomes):
    return float(np.mean(np.argmax(probs, axis=1) == outcomes))


def _predict_set(matches, params):
    """Prédit un ensemble de matchs. Retourne (probs Nx3, outcomes, xg_home, xg_away, actual)."""
    probs, outs, xh, xa, ah, aa = [], [], [], [], [], []
    for _, r in matches.iterrows():
        pred = predict_match(str(r["home_team"]), str(r["away_team"]),
                             params, neutral=bool(r["neutral"]))
        probs.append([pred["prob_home_win"], pred["prob_draw"], pred["prob_away_win"]])
        hs, as_ = int(r["home_score"]), int(r["away_score"])
        outs.append(0 if hs > as_ else 1 if hs == as_ else 2)
        xh.append(pred["expected_home"]); xa.append(pred["expected_away"])
        ah.append(hs); aa.append(as_)
    return (np.array(probs), np.array(outs),
            np.array(xh), np.array(xa), np.array(ah), np.array(aa))


def _tune_ridges(train, elo_dict, conf_map, verbose=True):
    """
    Règle ridge_team par validation temporelle. ridge_conf est FIXÉ (config) :
    la validation est dominée par l'intra-confédération, où le niveau de
    confédération s'annule dans (r_i − r_j) ; elle ne peut donc pas le régler.
    On imprime quand même la sensibilité à ridge_conf pour transparence.
    """
    train = train.sort_values("date")
    cutoff = train["date"].max() - pd.DateOffset(months=18)
    sub_tr = train[train["date"] <= cutoff]
    sub_va = train[train["date"] > cutoff]
    if len(sub_va) < 100 or len(sub_tr) < 500:
        return RIDGE_TEAM, RIDGE_CONF

    best_rt, best_ll = RIDGE_TEAM, np.inf
    for rt in RIDGE_TEAM_GRID:
        p = estimate_strengths(sub_tr, ref_date=cutoff, ridge_team=rt,
                               ridge_conf=RIDGE_CONF, elo_dict=elo_dict,
                               conf_map=conf_map, verbose=False)
        probs, outs, *_ = _predict_set(sub_va, p)
        ll = log_loss(probs, outs)
        if verbose:
            print(f"    ridge_team={rt:4.1f} (ridge_conf={RIDGE_CONF}) -> val log-loss={ll:.4f}")
        if ll < best_ll:
            best_ll, best_rt = ll, rt

    # Sensibilité (informative) à ridge_conf, au meilleur ridge_team
    if verbose:
        for rc in RIDGE_CONF_GRID:
            p = estimate_strengths(sub_tr, ref_date=cutoff, ridge_team=best_rt,
                                   ridge_conf=rc, elo_dict=elo_dict,
                                   conf_map=conf_map, verbose=False)
            probs, outs, *_ = _predict_set(sub_va, p)
            print(f"    [info] ridge_conf={rc:5.1f} -> val log-loss={log_loss(probs, outs):.4f} "
                  f"(quasi insensible -> fixe a {RIDGE_CONF} par prior)")
    print(f"  Retenu : ridge_team={best_rt}  ridge_conf={RIDGE_CONF} (prior)")
    return best_rt, RIDGE_CONF


# ── Backtest ──────────────────────────────────────────────────────────────────

def run_backtest(verbose: bool = True) -> pd.DataFrame:
    print("=" * 60)
    print("BACKTEST - modele force unique")
    print("=" * 60)

    results = fetch_results()
    results["date"] = pd.to_datetime(results["date"])
    elo_dict = build_elo_dict(fetch_elo())
    conf_map = confederation_map()

    train_start = pd.Timestamp(TRAIN_END) - pd.DateOffset(years=TRAIN_YEARS)
    train = results[(results["date"] >= train_start) & (results["date"] <= TRAIN_END)].copy()
    test  = results[(results["date"] >= TEST_START) & (results["date"] <= TEST_END)].copy()
    print(f"  Entrainement : {len(train)} matchs ({train_start.date()} -> {TRAIN_END})")
    print(f"  Test         : {len(test)} matchs ({TEST_START} -> {TEST_END})")

    # ── Réglage des deux ridges sur validation interne ──
    print("\nReglage du shrinkage (2 niveaux) sur validation temporelle...")
    ridge_team, ridge_conf = _tune_ridges(train, elo_dict, conf_map, verbose=verbose)

    # ── Modèle final + calibrateur (tous deux sans toucher au test) ──
    print("\nEstimation du modele final...")
    # NB : le backtest est purement évaluatif — il n'écrit JAMAIS les caches de
    # production (strength_params.pkl / calibrators.pkl), sinon predict.py
    # rechargerait un modèle entraîné seulement jusqu'à TRAIN_END.
    ref_date = pd.Timestamp(TRAIN_END)
    params = estimate_strengths(train, ref_date=ref_date, ridge_team=ridge_team,
                                ridge_conf=ridge_conf, elo_dict=elo_dict, conf_map=conf_map)
    print(f"  beta0={params['beta0']:.3f}  home_adv={params['home_adv']:.3f}  "
          f"rho={params['rho']:.4f}  ({len(params['teams'])} equipes)")

    # Calibrateur : ajusté sur une fenêtre de validation du train
    cal_cut = train["date"].max() - pd.DateOffset(months=18)
    cal_tr  = train[train["date"] <= cal_cut]
    cal_va  = train[train["date"] > cal_cut]
    cal = MultiClassCalibrator()
    if len(cal_va) >= 100 and len(cal_tr) >= 500:
        cal_params = estimate_strengths(cal_tr, ref_date=cal_cut, ridge_team=ridge_team,
                                        ridge_conf=ridge_conf, elo_dict=elo_dict,
                                        conf_map=conf_map, verbose=False)
        cp, co, *_ = _predict_set(cal_va, cal_params)
        cal.fit(cp, co)

    # ── Évaluation sur le test (aucune équipe ignorée) ──
    print("\nPredictions sur le test (aucune equipe ignoree)...")
    probs, outs, xh, xa, ah, aa = _predict_set(test, params)
    probs_cal = cal.transform(probs)

    def block(tag, P):
        bs, ll, acc = brier_score(P, outs), log_loss(P, outs), accuracy(P, outs)
        print(f"  [{tag}] Brier={bs:.4f}  LogLoss={ll:.4f}  Acc={acc*100:.1f}%")
        return bs, ll, acc

    print("\n-- Metriques globales (reference naif 1/3 : Brier 0.667) --")
    bs_r, ll_r, acc_r = block("brut    ", probs)
    bs_c, ll_c, acc_c = block("calibre ", probs_cal)
    rmse_h = float(np.sqrt(np.mean((xh - ah) ** 2)))
    rmse_a = float(np.sqrt(np.mean((xa - aa) ** 2)))
    print(f"  RMSE buts : dom={rmse_h:.3f}  ext={rmse_a:.3f}")

    # ── Calibration P(victoire domicile) ──
    print("\n-- Calibration P(victoire domicile) [brut -> calibre] --")
    cal_err_raw, cal_err_cal = _calibration_report(probs, probs_cal, outs)

    metrics = pd.DataFrame([{
        "ridge_team": ridge_team, "ridge_conf": ridge_conf,
        "brier_raw": bs_r, "logloss_raw": ll_r, "acc_raw": acc_r,
        "brier_cal": bs_c, "logloss_cal": ll_c, "acc_cal": acc_c,
        "rmse_home": rmse_h, "rmse_away": rmse_a,
        "cal_err_raw": cal_err_raw, "cal_err_cal": cal_err_cal,
        "n_test": len(test), "n_skipped": 0,
    }])
    metrics.to_csv(BACKTEST_OUT, index=False)
    print(f"\nMetriques sauvegardees -> {BACKTEST_OUT}")
    return metrics


def _calibration_report(probs, probs_cal, outs):
    bins = np.linspace(0, 1, 11)
    errs_raw, errs_cal = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (probs[:, 0] >= lo) & (probs[:, 0] < hi)
        if m.sum() == 0:
            continue
        obs = (outs[m] == 0).mean()
        pr  = probs[m, 0].mean()
        mc = (probs_cal[:, 0] >= lo) & (probs_cal[:, 0] < hi)
        pc  = probs_cal[mc, 0].mean() if mc.sum() else np.nan
        oc  = (outs[mc] == 0).mean() if mc.sum() else np.nan
        errs_raw.append(abs(pr - obs))
        if mc.sum():
            errs_cal.append(abs(pc - oc))
        print(f"  [{lo:.1f},{hi:.1f})  brut pred={pr:.3f} obs={obs:.3f} (n={m.sum():3d})"
              f"   | calibre pred={pc:.3f} obs={oc:.3f}")
    er = float(np.mean(errs_raw)) if errs_raw else np.nan
    ec = float(np.mean(errs_cal)) if errs_cal else np.nan
    print(f"  Erreur de calibration moyenne : brut={er:.4f}  calibre={ec:.4f}")
    return er, ec


if __name__ == "__main__":
    run_backtest(verbose=True)
