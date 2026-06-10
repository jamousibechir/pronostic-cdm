"""
Modèle de Poisson avec correction Dixon-Coles pour prédire les scores.

Référence :
  Dixon & Coles (1997) — "Modelling Association Football Scores and Inefficiencies
  in the Football Betting Market"

Paramètres estimés par MLE pondéré temporellement :
  alpha[i]  : force d'attaque de l'équipe i   (alpha > 1 = attaque forte)
  beta[i]   : faiblesse défensive de l'équipe i (beta > 1 = défense faible)
  gamma     : multiplicateur avantage domicile (gamma > 1)
  rho       : paramètre de correction pour les scores bas (0-0, 1-0, 0-1, 1-1)

Buts attendus :
  lambda (domicile) = alpha[home] * beta[away] * gamma
  mu     (extérieur) = alpha[away] * beta[home]

  Pour terrain neutre (CdM) : gamma = 1

Pondération temporelle : w = exp(-ln(2) * Δjours / demi_vie)
"""
import pickle
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from pathlib import Path

from config import DATA_DIR, HALF_LIFE_DAYS, MAX_GOALS, SEED

PARAMS_CACHE = DATA_DIR / "dc_params.pkl"


# ── Correction Dixon-Coles ────────────────────────────────────────────────────

def _tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """
    Facteur de correction DC pour les scores bas.
    Évite les probabilités négatives : rho doit être dans [-1/(lam*mu), 1/lam, 1/mu].
    """
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    elif x == 1 and y == 0:
        return 1.0 + mu * rho
    elif x == 0 and y == 1:
        return 1.0 + lam * rho
    elif x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


# ── Pondération temporelle ────────────────────────────────────────────────────

def time_weight(date: pd.Timestamp, reference: pd.Timestamp,
                half_life: int = HALF_LIFE_DAYS) -> float:
    delta_days = max((reference - date).days, 0)
    return float(np.exp(-np.log(2) * delta_days / half_life))


# ── Log-vraisemblance ─────────────────────────────────────────────────────────

def _neg_log_likelihood(params: np.ndarray,
                         matches: pd.DataFrame,
                         teams: list[str],
                         ref_date: pd.Timestamp) -> float:
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    log_alpha = params[:n]
    log_beta  = params[n:2*n]
    log_gamma = params[2*n]
    rho       = params[2*n + 1]

    alpha = np.exp(log_alpha)
    beta  = np.exp(log_beta)
    gamma = np.exp(log_gamma)

    ll = 0.0
    for row in matches.itertuples(index=False):
        i = idx.get(row.home_team)
        j = idx.get(row.away_team)
        if i is None or j is None:
            continue

        x = int(row.home_score)
        y = int(row.away_score)
        neutral = bool(row.neutral)

        g   = 1.0 if neutral else gamma
        lam = alpha[i] * beta[j] * g
        mu  = alpha[j] * beta[i]

        tau_val = _tau(x, y, lam, mu, rho)
        if tau_val <= 0:
            return 1e12   # paramètres invalides

        w = time_weight(pd.Timestamp(row.date), ref_date)
        log_p = (np.log(tau_val)
                 + poisson.logpmf(x, lam)
                 + poisson.logpmf(y, mu))
        ll += w * log_p

    return -ll


# ── Estimation des paramètres ─────────────────────────────────────────────────

def estimate_parameters(matches: pd.DataFrame,
                         ref_date: pd.Timestamp | None = None,
                         min_matches: int = 5) -> dict:
    """
    Estime les paramètres Dixon-Coles par MLE pondéré temporellement.

    Parameters
    ----------
    matches    : DataFrame avec colonnes [date, home_team, away_team,
                 home_score, away_score, neutral]
    ref_date   : date de référence pour les poids (défaut : aujourd'hui)
    min_matches: nb minimum de matchs récents pour inclure une équipe

    Returns
    -------
    dict avec clés 'alpha', 'beta' (dict team→float), 'gamma', 'rho', 'teams'
    """
    if ref_date is None:
        ref_date = pd.Timestamp.now()

    matches = matches.dropna(subset=["home_score", "away_score"]).copy()
    matches["date"] = pd.to_datetime(matches["date"])

    # Filtre les équipes avec peu de données
    team_counts = (
        pd.concat([matches["home_team"], matches["away_team"]])
        .value_counts()
    )
    valid_teams = team_counts[team_counts >= min_matches].index.tolist()
    matches = matches[
        matches["home_team"].isin(valid_teams) &
        matches["away_team"].isin(valid_teams)
    ]

    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    n = len(teams)
    print(f"  Estimation DC : {n} équipes, {len(matches)} matchs")

    # Paramètres initiaux
    x0 = np.zeros(2 * n + 2)
    x0[2*n]     = np.log(1.15)  # gamma ≈ 1.15 (avantage domicile modéré)
    x0[2*n + 1] = -0.10         # rho légèrement négatif

    # Bornes
    bounds = (
        [(-2.0, 2.0)] * n +    # log_alpha
        [(-2.0, 2.0)] * n +    # log_beta
        [(-0.5, 0.5)] +        # log_gamma
        [(-0.9, 0.9)]          # rho
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(
            _neg_log_likelihood,
            x0,
            args=(matches, teams, ref_date),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 3000, "ftol": 1e-9, "gtol": 1e-6},
        )

    if not result.success:
        print(f"  Avertissement optimisation : {result.message}")

    params_raw = result.x
    alpha_vals = np.exp(params_raw[:n])
    beta_vals  = np.exp(params_raw[n:2*n])

    # Normalisation : force que mean(alpha) = 1 (contrainte d'identification)
    alpha_mean = float(np.mean(alpha_vals))
    alpha_vals = alpha_vals / alpha_mean
    beta_vals  = beta_vals  * alpha_mean   # compense dans beta

    return {
        "teams": teams,
        "alpha": dict(zip(teams, alpha_vals.tolist())),
        "beta":  dict(zip(teams, beta_vals.tolist())),
        "gamma": float(np.exp(params_raw[2*n])),
        "rho":   float(params_raw[2*n + 1]),
        "log_likelihood": -result.fun,
    }


def save_params(params: dict, path: Path = PARAMS_CACHE) -> None:
    with open(path, "wb") as f:
        pickle.dump(params, f)
    print(f"Paramètres DC sauvegardés → {path}")


def load_params(path: Path = PARAMS_CACHE) -> dict | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Matrice de probabilité de scores ─────────────────────────────────────────

def score_matrix(alpha_h: float, beta_h: float,
                 alpha_a: float, beta_a: float,
                 rho: float, gamma: float = 1.0,
                 max_goals: int = MAX_GOALS) -> np.ndarray:
    """
    Calcule la matrice (max_goals+1) × (max_goals+1) des probabilités de scores.
    Element [x, y] = P(home scores x, away scores y).
    """
    lam = alpha_h * beta_a * gamma
    mu  = alpha_a * beta_h

    x_range = np.arange(max_goals + 1)
    # Distribution de Poisson tronquée
    p_home = poisson.pmf(x_range, lam)
    p_away = poisson.pmf(x_range, mu)

    matrix = np.outer(p_home, p_away)

    # Correction DC sur les scores bas
    for x in range(min(2, max_goals + 1)):
        for y in range(min(2, max_goals + 1)):
            matrix[x, y] *= _tau(x, y, lam, mu, rho)

    # Re-normalisation (la correction DC peut légèrement sortir de [0,1])
    total = matrix.sum()
    if total > 0:
        matrix /= total
    return matrix


def predict_match(home: str, away: str, params: dict,
                  neutral: bool = True,
                  max_goals: int = MAX_GOALS) -> dict:
    """
    Prédit le résultat d'un match.

    Returns
    -------
    dict avec :
      - matrix       : np.ndarray probabilités de scores
      - prob_home_win: float
      - prob_draw    : float
      - prob_away_win: float
      - expected_home: float (buts attendus domicile)
      - expected_away: float (buts attendus extérieur)
      - most_likely_score: tuple (h, a)
    """
    # Fallback Elo si l'équipe est inconnue du modèle DC
    alpha_h = params["alpha"].get(home, 1.0)
    beta_h  = params["beta"].get(home,  1.0)
    alpha_a = params["alpha"].get(away, 1.0)
    beta_a  = params["beta"].get(away,  1.0)
    rho     = params["rho"]
    gamma   = 1.0 if neutral else params["gamma"]

    mat = score_matrix(alpha_h, beta_h, alpha_a, beta_a, rho, gamma, max_goals)

    goals = np.arange(max_goals + 1)
    exp_h = float(mat.sum(axis=1) @ goals)
    exp_a = float(mat.sum(axis=0) @ goals)

    p_home_win = float(np.tril(mat, k=-1).sum())
    p_draw     = float(np.trace(mat))
    p_away_win = float(np.triu(mat, k=1).sum())

    # Score le plus probable
    idx = np.unravel_index(mat.argmax(), mat.shape)
    most_likely = (int(idx[0]), int(idx[1]))

    return {
        "matrix":            mat,
        "prob_home_win":     p_home_win,
        "prob_draw":         p_draw,
        "prob_away_win":     p_away_win,
        "expected_home":     exp_h,
        "expected_away":     exp_a,
        "most_likely_score": most_likely,
    }


def apply_elo_prior(params: dict, elo_dict: dict,
                    weight: float = 0.3) -> dict:
    """
    Enrichit les paramètres DC avec un prior Elo pour les équipes peu représentées.
    Mélange linéaire : params_final = (1-w)*DC + w*Elo_implied.

    weight : 0 = DC pur, 1 = Elo pur. Recommandé : 0.2–0.4.
    """
    from model.elo import expected_goals_from_elo

    if not elo_dict:
        return params

    # Elo moyen de référence
    elo_vals = list(elo_dict.values())
    elo_ref  = float(np.mean(elo_vals))

    new_alpha = dict(params["alpha"])
    new_beta  = dict(params["beta"])

    all_teams = set(elo_dict.keys())
    for team in all_teams:
        elo_t = elo_dict.get(team, elo_ref)
        # Buts attendus vs équipe moyenne
        lam_elo, _  = expected_goals_from_elo(elo_t, elo_ref)
        _, mu_elo   = expected_goals_from_elo(elo_ref, elo_t)
        # alpha ≈ attaque (lam quand beta_adverse = 1)
        # beta  ≈ faiblesse défensive
        alpha_elo = lam_elo
        beta_elo  = mu_elo   # buts encaissés quand alpha_adverse = 1

        if team in new_alpha:
            new_alpha[team] = (1 - weight) * new_alpha[team] + weight * alpha_elo
            new_beta[team]  = (1 - weight) * new_beta[team]  + weight * beta_elo
        else:
            # Équipe absente du modèle DC : on utilise le prior Elo seul
            new_alpha[team] = alpha_elo
            new_beta[team]  = beta_elo

    return {**params, "alpha": new_alpha, "beta": new_beta}


if __name__ == "__main__":
    # Test rapide avec des paramètres fictifs
    params_test = {
        "alpha": {"France": 1.8, "Argentine": 1.7},
        "beta":  {"France": 0.7, "Argentine": 0.75},
        "gamma": 1.15,
        "rho":   -0.10,
    }
    r = predict_match("France", "Argentine", params_test, neutral=True)
    print(f"France vs Argentine")
    print(f"  Score le plus probable : {r['most_likely_score']}")
    print(f"  P(France) = {r['prob_home_win']:.3f}  "
          f"P(nul) = {r['prob_draw']:.3f}  "
          f"P(Argentine) = {r['prob_away_win']:.3f}")
    print(f"  Buts attendus : {r['expected_home']:.2f} – {r['expected_away']:.2f}")
