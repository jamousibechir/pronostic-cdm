"""
Modèle de Poisson à FORCE UNIQUE par équipe (single-strength), avec :
  - estimation par maximum de vraisemblance pondéré temporellement,
  - GRADIENT ANALYTIQUE -> convergence garantie en quelques dizaines d'itérations
    (l'ancien Dixon-Coles attaque/défense ne convergeait pas : 2n params, pas de
     gradient, L-BFGS-B s'arrêtait sur EXCEEDS LIMIT),
  - SHRINKAGE HIÉRARCHIQUE (pénalité ridge = prior gaussien sur les forces)
    -> pooling partiel : les équipes peu vues sont rapprochées de la moyenne,
       et AUCUNE équipe n'est jamais ignorée (l'ancien modèle jetait 23% du test),
  - correction Dixon-Coles (rho) pour les scores bas, estimée après coup.

Référence : Ley, Van de Wiele & Van Eetvelde (2019), *Statistical Modelling* —
en sélection nationale, UNE force par équipe prédit mieux et plus calibré que le
couple attaque/défense, car bien plus robuste à la rareté des données.

Paramétrisation (échelle log) :
  log λ_dom = β0 + (r_i − r_j) + h · 1[pas neutre]
  log μ_ext = β0 + (r_j − r_i)
  r_i = force de l'équipe i (identifiée par le ridge : moyenne ≈ 0)
  β0  = log du taux de buts de base ; h = avantage domicile (additif en log)

Pondération temporelle : w = exp(−ln2 · Δjours / demi_vie)
"""
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

from config import DATA_DIR, HALF_LIFE_DAYS, MAX_GOALS, HOST_ADVANTAGE

PARAMS_CACHE = DATA_DIR / "strength_params.pkl"


# ── Correction Dixon-Coles (scores bas) ──────────────────────────────────────

def _tau(x, y, lam, mu, rho):
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    if x == 1 and y == 0:
        return 1.0 + mu * rho
    if x == 0 and y == 1:
        return 1.0 + lam * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


# ── Vraisemblance + gradient analytiques (Poisson + ridge) ───────────────────

def _nll_and_grad(theta, h_idx, a_idx, x, y, nn, w, n,
                  conf_idx, C, ridge_team, ridge_conf):
    """
    NLL Poisson pondérée + prior hiérarchique À DEUX NIVEAUX, et son gradient.

    theta = [β0, h, r_0..r_{n-1}, m_0..m_{C-1}]
      r_i = force de l'équipe i ; m_c = moyenne de la confédération c.

    Pénalité = λ_team·Σ_i (r_i − m_{c(i)})² + λ_conf·Σ_c m_c²
      -> rapproche chaque équipe de sa confédération (pooling intra-conf), et les
         confédérations de la moyenne globale. Les m_c sont identifiés surtout par
         les matchs inter-confédérations. Corrige le biais des équipes qui dominent
         des qualifs faibles (ex. AFC/CAF surévaluées).
    La vraisemblance ne dépend PAS de m_c (seulement de la pénalité).
    """
    beta0 = theta[0]
    hadv  = theta[1]
    r     = theta[2:2 + n]
    m     = theta[2 + n:2 + n + C]

    eta_h = beta0 + (r[h_idx] - r[a_idx]) + hadv * nn
    eta_a = beta0 + (r[a_idx] - r[h_idx])
    lam = np.exp(eta_h)
    mu  = np.exp(eta_a)

    diff = r - m[conf_idx]     # r_i − m_{c(i)}
    nll = (np.dot(w, (lam - x * eta_h) + (mu - y * eta_a))
           + ridge_team * np.dot(diff, diff)
           + ridge_conf * np.dot(m, m))

    A = w * (lam - x)          # ∂/∂η_h
    B = w * (mu - y)           # ∂/∂η_a
    Cres = A - B

    g_beta0 = np.sum(A + B)
    g_hadv  = np.sum(A * nn)
    g_r = (np.bincount(h_idx, weights=Cres, minlength=n)
           - np.bincount(a_idx, weights=Cres, minlength=n)
           + 2.0 * ridge_team * diff)
    g_m = (-2.0 * ridge_team * np.bincount(conf_idx, weights=diff, minlength=C)
           + 2.0 * ridge_conf * m)

    grad = np.empty_like(theta)
    grad[0] = g_beta0
    grad[1] = g_hadv
    grad[2:2 + n] = g_r
    grad[2 + n:2 + n + C] = g_m
    return nll, grad


def _fit_rho(lam, mu, x, y, w):
    """Estime rho (DC) par 1-D, à forces fixées. Ne touche que les scores bas."""
    low = (x <= 1) & (y <= 1)
    if not np.any(low):
        return 0.0
    xl, yl, ll, ml, wl = x[low], y[low], lam[low], mu[low], w[low]

    def neg_ll(rho):
        tau = np.ones_like(ll)
        m00 = (xl == 0) & (yl == 0)
        m10 = (xl == 1) & (yl == 0)
        m01 = (xl == 0) & (yl == 1)
        m11 = (xl == 1) & (yl == 1)
        tau[m00] = 1.0 - ll[m00] * ml[m00] * rho
        tau[m10] = 1.0 + ml[m10] * rho
        tau[m01] = 1.0 + ll[m01] * rho
        tau[m11] = 1.0 - rho
        if np.any(tau <= 1e-9):
            return 1e12
        return -np.dot(wl, np.log(tau))

    res = minimize_scalar(neg_ll, bounds=(-0.2, 0.2), method="bounded")
    return float(res.x)


# ── Estimation ───────────────────────────────────────────────────────────────

def estimate_strengths(matches: pd.DataFrame,
                       ref_date: pd.Timestamp | None = None,
                       ridge_team: float = 2.0,
                       ridge_conf: float = 1.0,
                       min_matches: int = 3,
                       elo_dict: dict[str, float] | None = None,
                       conf_map: dict[str, str] | None = None,
                       half_life: float = HALF_LIFE_DAYS,
                       verbose: bool = True) -> dict:
    """
    Estime les forces par MLE pondéré + prior hiérarchique à deux niveaux.

    Parameters
    ----------
    matches    : [date, home_team, away_team, home_score, away_score, neutral]
    ridge_team : shrinkage des équipes vers leur confédération.
    ridge_conf : shrinkage des confédérations vers la moyenne globale.
    elo_dict   : {team: elo} pour l'init et le fallback des inconnues.
    conf_map   : {team: confederation}. Si None, chargé via confederation_map().
    """
    if ref_date is None:
        ref_date = pd.Timestamp.now()
    if conf_map is None:
        from ingestion.confederations import confederation_map
        conf_map = confederation_map()

    m = matches.dropna(subset=["home_score", "away_score"]).copy()
    m["date"] = pd.to_datetime(m["date"])

    counts = pd.concat([m["home_team"], m["away_team"]]).value_counts()
    valid = counts[counts >= min_matches].index
    m = m[m["home_team"].isin(valid) & m["away_team"].isin(valid)].reset_index(drop=True)

    teams = sorted(set(m["home_team"]) | set(m["away_team"]))
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    # Confédération de chaque équipe -> indices
    team_conf = {t: conf_map.get(t, "OTHER") for t in teams}
    confs = sorted(set(team_conf.values()))
    conf_idx_map = {c: i for i, c in enumerate(confs)}
    C = len(confs)
    conf_idx = np.array([conf_idx_map[team_conf[t]] for t in teams], dtype=np.int64)

    if verbose:
        print(f"  Estimation force unique : {n} equipes, {len(m)} matchs, "
              f"{C} confederations, ridge_team={ridge_team} ridge_conf={ridge_conf}")

    h_idx = m["home_team"].map(idx).to_numpy(np.int64)
    a_idx = m["away_team"].map(idx).to_numpy(np.int64)
    x = m["home_score"].to_numpy(np.float64)
    y = m["away_score"].to_numpy(np.float64)
    nn = (~m["neutral"].fillna(False).to_numpy(bool)).astype(np.float64)  # 1 si domicile réel

    delta = (ref_date - m["date"]).dt.days.clip(lower=0).to_numpy(np.float64)
    w = np.exp(-np.log(2) * delta / half_life)
    # K-factor : un multiplicateur de poids par match (colonne optionnelle)
    # permet d'amplifier les matchs du Mondial en cours (apprentissage rapide
    # de la forme actuelle) — voir update_daily.py.
    if "weight_mult" in m.columns:
        w = w * m["weight_mult"].fillna(1.0).to_numpy(np.float64)

    # Initialisation : β0=log(moyenne buts), h léger, forces depuis Elo, m_c=0
    theta0 = np.zeros(n + 2 + C)
    theta0[0] = np.log(max(np.average((x + y) / 2, weights=w), 0.5))
    theta0[1] = 0.2
    if elo_dict:
        elo_vals = np.array([elo_dict.get(t, np.nan) for t in teams], float)
        if np.isfinite(elo_vals).sum() > 10:
            mean_elo = np.nanmean(elo_vals)
            elo_vals = np.where(np.isfinite(elo_vals), elo_vals, mean_elo)
            theta0[2:2 + n] = (elo_vals - mean_elo) / 400.0   # ~400 Elo ≈ 1 unité

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(
            _nll_and_grad, theta0,
            args=(h_idx, a_idx, x, y, nn, w, n, conf_idx, C, ridge_team, ridge_conf),
            method="L-BFGS-B", jac=True,
            options={"maxiter": 800, "ftol": 1e-9, "gtol": 1e-6},
        )
    if not res.success and verbose:
        print(f"  Avertissement optimisation : {res.message}")

    beta0 = float(res.x[0])
    hadv  = float(res.x[1])
    r     = res.x[2:2 + n].copy()
    mvec  = res.x[2 + n:2 + n + C].copy()
    shift = r.mean()
    r    -= shift          # recentrage (cosmétique : seules les différences comptent)
    mvec -= shift

    # rho post-hoc
    eta_h = beta0 + (r[h_idx] - r[a_idx]) + hadv * nn
    eta_a = beta0 + (r[a_idx] - r[h_idx])
    rho = _fit_rho(np.exp(eta_h), np.exp(eta_a), x, y, w)

    strength = dict(zip(teams, r.tolist()))
    conf_strength = {c: float(mvec[conf_idx_map[c]]) for c in confs}

    # Repli pour les équipes hors échantillon : Elo projeté, sinon moyenne de conf
    elo_strength = {}
    if elo_dict:
        known = np.array([strength[t] for t in teams])
        elo_known = np.array([elo_dict.get(t, np.nan) for t in teams], float)
        mask = np.isfinite(elo_known)
        if mask.sum() > 10:
            b = np.polyfit(elo_known[mask], known[mask], 1)
            for t, e in elo_dict.items():
                if t not in strength:
                    elo_strength[t] = float(np.polyval(b, e))

    return {
        "model":     "strength_poisson",
        "teams":     teams,
        "strength":  strength,
        "team_conf": team_conf,
        "conf_strength": conf_strength,
        "elo_fallback_strength": elo_strength,
        "beta0":     beta0,
        "home_adv":  hadv,
        "rho":       rho,
        "ridge_team": ridge_team,
        "ridge_conf": ridge_conf,
        "ref_date":  ref_date,
        "log_likelihood": -float(res.fun),
    }


# ── Bootstrap : propagation de l'incertitude d'estimation ────────────────────

def bootstrap_strengths(matches: pd.DataFrame,
                        base_params: dict,
                        B: int = 40,
                        ridge_team: float = 2.0,
                        ridge_conf: float = 1.0,
                        ref_date: pd.Timestamp | None = None,
                        elo_dict: dict[str, float] | None = None,
                        conf_map: dict[str, str] | None = None,
                        block: str = "tournament",
                        half_life: float = HALF_LIFE_DAYS,
                        seed: int = 42) -> list[dict]:
    """
    Ré-estime le modèle sur B rééchantillonnages bootstrap des matchs.

    BLOCK-BOOTSTRAP : au lieu de rééchantillonner les matchs en i.i.d (qui casse
    l'autocorrélation temporelle et sous-estime l'incertitude), on rééchantillonne
    des BLOCS entiers avec remise :
      - block="tournament" : par (compétition x année)  [défaut]
      - block="month6"     : par fenêtres de 6 mois
      - block="iid"        : ancien comportement (matchs individuels)

    Chaque réplica contient TOUTES les équipes de base_params (repli sur la force
    de base si une équipe manque d'un réplica). Le Monte-Carlo tire un réplica par
    simulation -> incertitude d'estimation propagée dans les probabilités de titre.
    """
    if conf_map is None:
        from ingestion.confederations import confederation_map
        conf_map = confederation_map()

    rng = np.random.default_rng(seed)
    base_strength = base_params["strength"]

    # Construit les identifiants de bloc
    mm = matches.copy()
    mm["date"] = pd.to_datetime(mm["date"])
    if block == "iid":
        mm["_block"] = np.arange(len(mm))
    elif block == "month6":
        mm["_block"] = (mm["date"].dt.year.astype(str) + "-"
                        + (mm["date"].dt.month > 6).astype(int).astype(str))
    else:  # tournament
        tour = mm["tournament"] if "tournament" in mm.columns else "all"
        mm["_block"] = (pd.Series(tour, index=mm.index).astype(str)
                        + "-" + mm["date"].dt.year.astype(str))

    groups = {b: g for b, g in mm.groupby("_block")}
    block_ids = list(groups.keys())
    replicas: list[dict] = []

    for _ in range(B):
        chosen = rng.choice(len(block_ids), size=len(block_ids), replace=True)
        sample = pd.concat([groups[block_ids[i]] for i in chosen], ignore_index=True)
        try:
            p = estimate_strengths(sample, ref_date=ref_date,
                                   ridge_team=ridge_team, ridge_conf=ridge_conf,
                                   elo_dict=elo_dict, conf_map=conf_map,
                                   half_life=half_life, verbose=False)
        except Exception:
            continue
        merged = dict(base_strength)
        merged.update(p["strength"])
        replicas.append({
            "model": "strength_poisson",
            "strength": merged,
            "elo_fallback_strength": base_params.get("elo_fallback_strength", {}),
            "beta0": p["beta0"], "home_adv": p["home_adv"], "rho": p["rho"],
        })

    if not replicas:
        replicas = [base_params]
    print(f"  Bootstrap ({block}) : {len(replicas)} replicas de parametres")
    return replicas


# ── Forces -> buts attendus ───────────────────────────────────────────────────

def _strength_of(team: str, params: dict) -> float:
    if team in params["strength"]:
        return params["strength"][team]
    return params.get("elo_fallback_strength", {}).get(team, 0.0)


def expected_goals(home: str, away: str, params: dict,
                   neutral: bool = True,
                   host_teams: set | None = None,
                   host_adv: float = HOST_ADVANTAGE) -> tuple[float, float]:
    """
    Buts attendus (λ_dom, μ_ext).

    host_teams : si fourni, une équipe hôte qui joue reçoit un bonus offensif
    `host_adv` (indépendamment du statut domicile/extérieur nominal, car en CdM
    le bracket attribue domicile/extérieur arbitrairement). Non passé en backtest
    -> aucun effet sur l'évaluation historique.
    """
    rh = _strength_of(home, params)
    ra = _strength_of(away, params)
    b  = params["beta0"]
    h  = 0.0 if neutral else params["home_adv"]
    host_h = host_adv if (host_teams and home in host_teams) else 0.0
    host_a = host_adv if (host_teams and away in host_teams) else 0.0
    lam = np.exp(b + (rh - ra) + h + host_h)
    mu  = np.exp(b + (ra - rh) + host_a)
    return float(lam), float(mu)


# ── Matrice de scores + prédiction (interface compatible) ─────────────────────

def score_matrix(lam: float, mu: float, rho: float,
                 max_goals: int = MAX_GOALS) -> np.ndarray:
    xr = np.arange(max_goals + 1)
    mat = np.outer(poisson.pmf(xr, lam), poisson.pmf(xr, mu))
    for xi in range(min(2, max_goals + 1)):
        for yi in range(min(2, max_goals + 1)):
            mat[xi, yi] *= _tau(xi, yi, lam, mu, rho)
    s = mat.sum()
    if s > 0:
        mat /= s
    return mat


def predict_match(home: str, away: str, params: dict,
                  neutral: bool = True, max_goals: int = MAX_GOALS,
                  host_teams: set | None = None) -> dict:
    lam, mu = expected_goals(home, away, params, neutral, host_teams=host_teams)
    mat = score_matrix(lam, mu, params["rho"], max_goals)
    goals = np.arange(max_goals + 1)
    exp_h = float(mat.sum(axis=1) @ goals)
    exp_a = float(mat.sum(axis=0) @ goals)
    p_hw = float(np.tril(mat, -1).sum())
    p_d  = float(np.trace(mat))
    p_aw = float(np.triu(mat,  1).sum())
    i, j = np.unravel_index(mat.argmax(), mat.shape)
    return {
        "matrix": mat,
        "prob_home_win": p_hw, "prob_draw": p_d, "prob_away_win": p_aw,
        "expected_home": exp_h, "expected_away": exp_a,
        "most_likely_score": (int(i), int(j)),
    }


# ── Persistance ───────────────────────────────────────────────────────────────

def save_params(params: dict, path: Path = PARAMS_CACHE) -> None:
    with open(path, "wb") as f:
        pickle.dump(params, f)
    print(f"Parametres force unique sauvegardes -> {path}")


def load_params(path: Path = PARAMS_CACHE) -> dict | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from ingestion.results_fetch import recent_results
    from model.elo import build_elo_dict
    from ingestion.elo_fetch import fetch_elo

    df = recent_results(years=8)
    elo = build_elo_dict(fetch_elo())
    p = estimate_strengths(df, ref_date=pd.Timestamp.now(),
                           ridge_team=2.0, ridge_conf=1.0, elo_dict=elo)
    print(f"  beta0={p['beta0']:.3f}  home_adv={p['home_adv']:.3f}  rho={p['rho']:.4f}")
    top = sorted(p["strength"].items(), key=lambda kv: kv[1], reverse=True)[:15]
    print("\n  Top 15 forces :")
    for t, s in top:
        print(f"    {t:25s} {s:+.3f}")
