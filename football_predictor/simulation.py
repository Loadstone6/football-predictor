from __future__ import annotations

from math import exp, factorial

import numpy as np


def poisson_score_matrix(home_xg: float, away_xg: float, max_goals: int = 7) -> np.ndarray:
    home_xg = float(np.clip(home_xg, 0.05, 6.0))
    away_xg = float(np.clip(away_xg, 0.05, 6.0))
    home_probs = np.array([exp(-home_xg) * home_xg**k / factorial(k) for k in range(max_goals + 1)])
    away_probs = np.array([exp(-away_xg) * away_xg**k / factorial(k) for k in range(max_goals + 1)])
    matrix = np.outer(home_probs, away_probs)
    matrix = matrix / matrix.sum()
    return matrix


def dixon_coles_score_matrix(home_xg: float, away_xg: float, max_goals: int = 7, rho: float = -0.08) -> np.ndarray:
    matrix = poisson_score_matrix(home_xg, away_xg, max_goals=max_goals)
    home_xg = float(np.clip(home_xg, 0.05, 6.0))
    away_xg = float(np.clip(away_xg, 0.05, 6.0))
    adjustments = {
        (0, 0): 1.0 - home_xg * away_xg * rho,
        (0, 1): 1.0 + home_xg * rho,
        (1, 0): 1.0 + away_xg * rho,
        (1, 1): 1.0 - rho,
    }
    for (home_goals, away_goals), factor in adjustments.items():
        if home_goals <= max_goals and away_goals <= max_goals:
            matrix[home_goals, away_goals] *= max(0.05, factor)
    return matrix / matrix.sum()


def score_outcome_probabilities(score_matrix: np.ndarray) -> tuple[float, float, float]:
    home_win = float(np.tril(score_matrix, k=-1).sum())
    draw = float(np.trace(score_matrix))
    away_win = float(np.triu(score_matrix, k=1).sum())
    total = home_win + draw + away_win
    return home_win / total, draw / total, away_win / total


def top_scoreline(score_matrix: np.ndarray) -> tuple[int, int, float]:
    idx = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
    return int(idx[0]), int(idx[1]), float(score_matrix[idx])


def score_market_probabilities(score_matrix: np.ndarray) -> dict:
    max_goal = score_matrix.shape[0] - 1
    total_goals = np.fromfunction(lambda home, away: home + away, score_matrix.shape)
    home_goals = np.arange(max_goal + 1).reshape(-1, 1)
    away_goals = np.arange(max_goal + 1).reshape(1, -1)
    return {
        "over_0_5": float(score_matrix[total_goals > 0.5].sum()),
        "over_1_5": float(score_matrix[total_goals > 1.5].sum()),
        "over_2_5": float(score_matrix[total_goals > 2.5].sum()),
        "over_3_5": float(score_matrix[total_goals > 3.5].sum()),
        "btts": float(score_matrix[(home_goals > 0) & (away_goals > 0)].sum()),
        "home_clean_sheet": float(score_matrix[:, 0].sum()),
        "away_clean_sheet": float(score_matrix[0, :].sum()),
    }


def simulate_match_physics(
    home_attack: float,
    away_attack: float,
    home_defense: float,
    away_defense: float,
    tempo: float = 1.0,
    simulations: int = 5000,
    seed: int = 7,
) -> dict:
    rng = np.random.default_rng(seed)
    tempo = float(np.clip(tempo, 0.65, 1.35))
    home_shots = np.clip((8.0 + 2.4 * home_attack + 0.8 * away_defense) * tempo, 3.0, 22.0)
    away_shots = np.clip((8.0 + 2.4 * away_attack + 0.8 * home_defense) * tempo, 3.0, 22.0)
    home_quality = np.clip(0.08 + 0.025 * home_attack - 0.014 * away_defense, 0.035, 0.22)
    away_quality = np.clip(0.08 + 0.025 * away_attack - 0.014 * home_defense, 0.035, 0.22)

    home_goals = rng.binomial(rng.poisson(home_shots, simulations), home_quality)
    away_goals = rng.binomial(rng.poisson(away_shots, simulations), away_quality)
    home_wins = float(np.mean(home_goals > away_goals))
    draws = float(np.mean(home_goals == away_goals))
    away_wins = float(np.mean(home_goals < away_goals))
    return {
        "home_xg": float(np.mean(home_goals)),
        "away_xg": float(np.mean(away_goals)),
        "home_win": home_wins,
        "draw": draws,
        "away_win": away_wins,
    }
