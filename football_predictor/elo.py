from __future__ import annotations

from dataclasses import dataclass
from math import exp


@dataclass(frozen=True)
class EloConfig:
    base_rating: float = 1500.0
    k_factor: float = 24.0
    home_advantage: float = 55.0
    draw_base: float = 0.26
    draw_floor: float = 0.08
    draw_scale: float = 420.0


def expected_score(home_rating: float, away_rating: float, neutral_site: bool, config: EloConfig) -> float:
    home_adjustment = 0.0 if neutral_site else config.home_advantage
    diff = (home_rating + home_adjustment) - away_rating
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def result_score(home_goals: float, away_goals: float) -> float:
    if home_goals > away_goals:
        return 1.0
    if home_goals < away_goals:
        return 0.0
    return 0.5


def goal_multiplier(home_goals: float, away_goals: float) -> float:
    margin = abs(float(home_goals) - float(away_goals))
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11.0 + margin) / 8.0


def update_ratings(
    home_rating: float,
    away_rating: float,
    home_goals: float,
    away_goals: float,
    neutral_site: bool,
    config: EloConfig,
) -> tuple[float, float]:
    expected = expected_score(home_rating, away_rating, neutral_site, config)
    actual = result_score(home_goals, away_goals)
    change = config.k_factor * goal_multiplier(home_goals, away_goals) * (actual - expected)
    return home_rating + change, away_rating - change


def three_way_probabilities(
    home_rating: float,
    away_rating: float,
    neutral_site: bool,
    config: EloConfig,
) -> tuple[float, float, float]:
    home_binary = expected_score(home_rating, away_rating, neutral_site, config)
    home_adjustment = 0.0 if neutral_site else config.home_advantage
    diff = abs((home_rating + home_adjustment) - away_rating)
    draw_probability = config.draw_floor + config.draw_base * exp(-diff / config.draw_scale)
    draw_probability = min(0.34, max(config.draw_floor, draw_probability))
    remainder = 1.0 - draw_probability
    home_probability = remainder * home_binary
    away_probability = remainder * (1.0 - home_binary)
    total = home_probability + draw_probability + away_probability
    return home_probability / total, draw_probability / total, away_probability / total
