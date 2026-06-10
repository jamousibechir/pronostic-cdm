"""
Modèle Elo pour les matchs internationaux.

Formule standard :
  P(victoire A) = 1 / (1 + 10^(-(Elo_A - Elo_B) / 400))

Terme de nul calibré empiriquement sur les matchs internationaux :
  P(nul) ≈ P_draw_max * exp(-( |ΔElo| / σ )^2)
  avec P_draw_max ≈ 0.27 et σ ≈ 200 (calibré sur résultats historiques)

Pour les matchs sur terrain neutre : pas d'ajustement avantage domicile.
Pour les qualifications / matchs à domicile : home_bonus = +100 Elo points.
"""
import numpy as np
import pandas as pd


# Paramètres calibrés sur les matchs internationaux 1990-2024
# (voir backtest.py pour la re-calibration)
P_DRAW_MAX  = 0.265   # probabilité de nul quand ΔElo = 0
DRAW_SIGMA  = 200.0   # Elo units de demi-décroissance
HOME_BONUS  = 100     # bonus Elo domicile (0 pour terrain neutre)


def win_probability(elo_a: float, elo_b: float,
                    neutral: bool = True) -> float:
    """
    P(A bat B) selon la formule Elo.

    Parameters
    ----------
    elo_a   : rating Elo de l'équipe A
    elo_b   : rating Elo de l'équipe B
    neutral : True = terrain neutre (pas d'avantage domicile)
    """
    bonus = 0 if neutral else HOME_BONUS
    delta = (elo_a - elo_b) + bonus
    return 1.0 / (1.0 + 10.0 ** (-delta / 400.0))


def draw_probability(elo_a: float, elo_b: float,
                     neutral: bool = True) -> float:
    """
    P(nul) en fonction de la différence de ratings Elo.
    Modèle gaussien centré sur ΔElo = 0.
    """
    bonus = 0 if neutral else HOME_BONUS
    delta = abs((elo_a - elo_b) + bonus)
    return P_DRAW_MAX * np.exp(-(delta / DRAW_SIGMA) ** 2)


def outcome_probabilities(elo_a: float, elo_b: float,
                           neutral: bool = True) -> tuple[float, float, float]:
    """
    Retourne (P(A gagne), P(nul), P(B gagne)).
    Les probabilités sont normalisées pour sommer à 1.
    """
    p_draw = draw_probability(elo_a, elo_b, neutral)
    p_raw_win = win_probability(elo_a, elo_b, neutral)
    # Distribue le reste (1 - p_draw) entre victoire et défaite
    remainder = 1.0 - p_draw
    p_win  = p_raw_win  * remainder
    p_loss = (1.0 - p_raw_win) * remainder
    return p_win, p_draw, p_loss


def expected_goals_from_elo(elo_a: float, elo_b: float,
                             avg_goals: float = 1.3) -> tuple[float, float]:
    """
    Estime les buts attendus à partir du ratio Elo.
    Utilisé comme prior pour le modèle Dixon-Coles quand les données sont rares.

    avg_goals : moyenne de buts par équipe par match en CdM (≈1.3)
    """
    p_win, _, _ = outcome_probabilities(elo_a, elo_b, neutral=True)
    # Force relative : ratio des chances de victoire
    strength_ratio = p_win / max(1 - p_win, 1e-6)
    # Scale pour que la moyenne des deux équipes = avg_goals
    lam = avg_goals * np.sqrt(strength_ratio)
    mu  = avg_goals / np.sqrt(strength_ratio)
    # Borne raisonnable [0.3, 3.5]
    lam = float(np.clip(lam, 0.3, 3.5))
    mu  = float(np.clip(mu,  0.3, 3.5))
    return lam, mu


def build_elo_dict(elo_df: pd.DataFrame) -> dict[str, float]:
    """Convertit le DataFrame Elo en dict {team_name: elo}."""
    return dict(zip(elo_df["team"], elo_df["elo"].astype(float)))


if __name__ == "__main__":
    # Exemples
    print("France vs Argentine (terrain neutre)")
    p_w, p_d, p_l = outcome_probabilities(2055, 1945, neutral=True)
    print(f"  France gagne : {p_w:.3f}  Nul : {p_d:.3f}  Argentine gagne : {p_l:.3f}")

    print("\nAngleterre vs San Marino")
    p_w, p_d, p_l = outcome_probabilities(1950, 1200, neutral=False)
    print(f"  Angleterre gagne : {p_w:.3f}  Nul : {p_d:.3f}  San Marino gagne : {p_l:.3f}")

    lam, mu = expected_goals_from_elo(2055, 1945)
    print(f"\nButs attendus France/Argentine : {lam:.2f} / {mu:.2f}")
