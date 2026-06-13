"""
Calibration post-hoc des probabilités 1/N/2 par régression isotonique.

Le modèle peut être discriminant mais mal calibré (ex. backtest : prédit 0.04
quand l'observé est 0.13 sur les outsiders). La régression isotonique apprend
une transformation monotone p_prédit -> p_calibré sur un jeu de validation
temporellement séparé, puis on renormalise les 3 sorties à somme 1.

Implémentation PAVA (Pool Adjacent Violators) maison -> aucune dépendance sklearn.
Doit être ajustée sur des prédictions HORS échantillon d'entraînement du modèle.
"""
import pickle
import numpy as np
from pathlib import Path

from config import DATA_DIR

CALIB_CACHE = DATA_DIR / "calibrators.pkl"


def _pava(p: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Régression isotonique non décroissante (moindres carrés) via PAVA.
    Retourne (x_knots croissants, y_knots non décroissants) pour interpolation.
    """
    order = np.argsort(p, kind="mergesort")
    xs = p[order].astype(float)
    ys = target[order].astype(float)

    vals: list[float] = []
    wts:  list[float] = []
    cnts: list[int]   = []
    for v in ys:
        vals.append(v); wts.append(1.0); cnts.append(1)
        while len(vals) > 1 and vals[-2] >= vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnts.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnts.pop()
            nw = w1 + w2
            vals.append((v1 * w1 + v2 * w2) / nw)
            wts.append(nw); cnts.append(c1 + c2)

    yhat = np.empty_like(ys)
    i = 0
    for v, c in zip(vals, cnts):
        yhat[i:i + c] = v
        i += c

    # Dédoublonne les x (np.interp exige xp croissant) en moyennant les y
    ux, inv = np.unique(xs, return_inverse=True)
    uy = np.zeros_like(ux)
    cnt = np.zeros_like(ux)
    np.add.at(uy, inv, yhat)
    np.add.at(cnt, inv, 1.0)
    uy /= np.maximum(cnt, 1)
    uy = np.maximum.accumulate(uy)   # garantit la monotonie après moyennage
    return ux, uy


class IsotonicCalibrator:
    """Calibrateur isotonique 1-D (un par classe, en one-vs-rest)."""
    def __init__(self):
        self.x = np.array([0.0, 1.0])
        self.y = np.array([0.0, 1.0])

    def fit(self, p: np.ndarray, target: np.ndarray) -> "IsotonicCalibrator":
        if len(p) >= 5:
            self.x, self.y = _pava(np.asarray(p), np.asarray(target))
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return np.interp(np.clip(p, 0, 1), self.x, self.y, left=self.y[0], right=self.y[-1])


class MultiClassCalibrator:
    """Calibre (P_dom, P_nul, P_ext) par isotonique one-vs-rest + renormalisation."""
    def __init__(self):
        self.cal = [IsotonicCalibrator() for _ in range(3)]
        self.fitted = False

    def fit(self, probs: np.ndarray, outcomes: np.ndarray) -> "MultiClassCalibrator":
        """probs (N,3) ; outcomes (N,) dans {0,1,2}."""
        probs = np.asarray(probs)
        outcomes = np.asarray(outcomes)
        for k in range(3):
            self.cal[k].fit(probs[:, k], (outcomes == k).astype(float))
        self.fitted = True
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=float)
        if not self.fitted:
            return probs
        out = np.column_stack([self.cal[k].predict(probs[:, k]) for k in range(3)])
        out = np.clip(out, 1e-6, None)
        out /= out.sum(axis=1, keepdims=True)
        return out

    def transform_one(self, p_hw: float, p_d: float, p_aw: float) -> tuple[float, float, float]:
        c = self.transform(np.array([[p_hw, p_d, p_aw]]))[0]
        return float(c[0]), float(c[1]), float(c[2])


def _outcomes_from_scores(hs, as_) -> np.ndarray:
    """0 = victoire dom, 1 = nul, 2 = victoire ext."""
    hs = np.asarray(hs); as_ = np.asarray(as_)
    out = np.where(hs > as_, 0, np.where(hs == as_, 1, 2))
    return out.astype(int)


def train_temporal_calibrator(results, elo_dict=None, ridge_team: float = 2.0,
                              ridge_conf: float = 1.0, conf_map=None,
                              holdout_months: int = 18) -> "MultiClassCalibrator":
    """
    Construit un calibrateur de façon honnête : entraîne le modèle sur les données
    ANCIENNES, prédit la fenêtre de validation récente, ajuste l'isotonique dessus.
    """
    import pandas as pd
    from model.strength_poisson import estimate_strengths, predict_match

    results = results.dropna(subset=["home_score", "away_score"]).copy()
    results["date"] = pd.to_datetime(results["date"])
    cutoff = results["date"].max() - pd.DateOffset(months=holdout_months)

    train = results[results["date"] <= cutoff]
    hold  = results[results["date"] > cutoff]
    if len(hold) < 50 or len(train) < 200:
        return MultiClassCalibrator()   # pas assez de données -> identité

    params = estimate_strengths(train, ref_date=cutoff, ridge_team=ridge_team,
                                ridge_conf=ridge_conf, elo_dict=elo_dict,
                                conf_map=conf_map, verbose=False)
    probs, outs = [], []
    for _, r in hold.iterrows():
        pred = predict_match(str(r["home_team"]), str(r["away_team"]),
                             params, neutral=bool(r["neutral"]))
        probs.append([pred["prob_home_win"], pred["prob_draw"], pred["prob_away_win"]])
        outs.append(0 if r["home_score"] > r["away_score"]
                    else 1 if r["home_score"] == r["away_score"] else 2)
    probs = np.array(probs); outs = np.array(outs)

    # Auto-désactivation : on n'applique la calibration que si elle améliore la
    # log-loss sur une sous-coupe de contrôle. Le modèle force unique est souvent
    # déjà bien calibré -> l'isotonique sur peu de données peut nuire.
    n = len(outs)
    idx = np.arange(n)
    fit_idx, chk_idx = idx[: n // 2], idx[n // 2:]
    if len(chk_idx) < 30:
        return MultiClassCalibrator().fit(probs, outs)

    cal = MultiClassCalibrator().fit(probs[fit_idx], outs[fit_idx])
    def _ll(P):
        p = np.clip(P[np.arange(len(chk_idx)), outs[chk_idx]], 1e-9, 1)
        return -np.mean(np.log(p))
    raw_ll = _ll(probs[chk_idx])
    cal_ll = _ll(cal.transform(probs[chk_idx]))
    if cal_ll >= raw_ll:
        return MultiClassCalibrator()   # identité : la calibration n'aide pas
    return MultiClassCalibrator().fit(probs, outs)


def save_calibrator(cal: MultiClassCalibrator, path: Path = CALIB_CACHE) -> None:
    with open(path, "wb") as f:
        pickle.dump(cal, f)


def load_calibrator(path: Path = CALIB_CACHE) -> MultiClassCalibrator | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
