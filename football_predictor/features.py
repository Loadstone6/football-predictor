from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from football_predictor.betting import normalized_implied_probabilities
from football_predictor.elo import EloConfig, three_way_probabilities, update_ratings


FEATURE_COLUMNS = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "elo_home_prob",
    "elo_draw_prob",
    "elo_away_prob",
    "home_recent_points",
    "away_recent_points",
    "recent_points_diff",
    "home_goal_diff_form",
    "away_goal_diff_form",
    "goal_diff_form_diff",
    "home_attack_form",
    "away_attack_form",
    "home_defense_form",
    "away_defense_form",
    "home_xg_form",
    "away_xg_form",
    "xg_form_diff",
    "home_rest_days",
    "away_rest_days",
    "rest_diff",
    "neutral_site",
    "home_advantage_flag",
    "market_home_prob",
    "market_draw_prob",
    "market_away_prob",
    "poisson_home_xg",
    "poisson_away_xg",
]


@dataclass(frozen=True)
class FeatureConfig:
    form_window: int = 8
    default_rest_days: float = 10.0
    max_rest_days: float = 30.0
    default_goals: float = 1.25
    default_xg: float = 1.25


def _avg(values: deque, default: float) -> float:
    if not values:
        return default
    return float(np.mean(values))


def _rest_days(last_date, current_date, config: FeatureConfig) -> float:
    if last_date is None or pd.isna(last_date):
        return config.default_rest_days
    return float(min(config.max_rest_days, max(0, (current_date - last_date).days)))


def _team_snapshot(team: str, state: dict, current_date, config: FeatureConfig) -> dict:
    return {
        "points": _avg(state["points"][team], 1.2),
        "goal_diff": _avg(state["goal_diff"][team], 0.0),
        "attack": _avg(state["goals_for"][team], config.default_goals),
        "defense": _avg(state["goals_against"][team], config.default_goals),
        "xg_for": _avg(state["xg_for"][team], config.default_xg),
        "xg_against": _avg(state["xg_against"][team], config.default_xg),
        "rest": _rest_days(state["last_date"].get(team), current_date, config),
    }


def _expected_goals_from_snapshot(home: dict, away: dict, elo_diff: float, neutral_site: bool) -> tuple[float, float]:
    home_field = 0.12 if not neutral_site else 0.0
    home_lambda = (
        0.55 * home["attack"]
        + 0.25 * away["defense"]
        + 0.20 * home["xg_for"]
        + home_field
        + 0.0018 * elo_diff
    )
    away_lambda = (
        0.55 * away["attack"]
        + 0.25 * home["defense"]
        + 0.20 * away["xg_for"]
        - 0.0012 * elo_diff
    )
    return float(np.clip(home_lambda, 0.25, 3.75)), float(np.clip(away_lambda, 0.25, 3.75))


def _market_state(match: pd.Series, fallback: tuple[float, float, float]) -> tuple[tuple[float, float, float], bool, float | None]:
    odds = (match.get("home_odds"), match.get("draw_odds"), match.get("away_odds"))
    market = normalized_implied_probabilities(odds)
    has_market = not any(v is None for v in market)
    overround = None
    if has_market:
        overround = float(sum(1.0 / float(odd) for odd in odds))
        return (float(market[0]), float(market[1]), float(market[2])), True, overround
    return fallback, False, overround


def build_walk_forward_features(
    matches: pd.DataFrame,
    feature_config: FeatureConfig | None = None,
    elo_config: EloConfig | None = None,
) -> pd.DataFrame:
    feature_config = feature_config or FeatureConfig()
    elo_config = elo_config or EloConfig()
    matches = matches.sort_values(["date", "competition", "home_team", "away_team"]).reset_index(drop=True)

    ratings = defaultdict(lambda: elo_config.base_rating)
    state = {
        "points": defaultdict(lambda: deque(maxlen=feature_config.form_window)),
        "goal_diff": defaultdict(lambda: deque(maxlen=feature_config.form_window)),
        "goals_for": defaultdict(lambda: deque(maxlen=feature_config.form_window)),
        "goals_against": defaultdict(lambda: deque(maxlen=feature_config.form_window)),
        "xg_for": defaultdict(lambda: deque(maxlen=feature_config.form_window)),
        "xg_against": defaultdict(lambda: deque(maxlen=feature_config.form_window)),
        "last_date": {},
    }
    rows = []

    for _, match in matches.iterrows():
        home_team = match["home_team"]
        away_team = match["away_team"]
        current_date = match["date"]
        neutral_site = bool(match.get("neutral_site", False))

        home_elo = float(ratings[home_team])
        away_elo = float(ratings[away_team])
        elo_home, elo_draw, elo_away = three_way_probabilities(home_elo, away_elo, neutral_site, elo_config)
        elo_diff = home_elo - away_elo

        home = _team_snapshot(home_team, state, current_date, feature_config)
        away = _team_snapshot(away_team, state, current_date, feature_config)
        poisson_home_xg, poisson_away_xg = _expected_goals_from_snapshot(home, away, elo_diff, neutral_site)

        market, has_market_odds, market_overround = _market_state(match, (elo_home, elo_draw, elo_away))
        home_goals = float(match["home_goals"])
        away_goals = float(match["away_goals"])
        home_elo_post, away_elo_post = update_ratings(
            home_elo, away_elo, home_goals, away_goals, neutral_site, elo_config
        )

        rows.append(
            {
                **match.to_dict(),
                "home_elo": home_elo,
                "away_elo": away_elo,
                "home_elo_post": home_elo_post,
                "away_elo_post": away_elo_post,
                "elo_diff": elo_diff,
                "elo_home_prob": elo_home,
                "elo_draw_prob": elo_draw,
                "elo_away_prob": elo_away,
                "home_recent_points": home["points"],
                "away_recent_points": away["points"],
                "recent_points_diff": home["points"] - away["points"],
                "home_goal_diff_form": home["goal_diff"],
                "away_goal_diff_form": away["goal_diff"],
                "goal_diff_form_diff": home["goal_diff"] - away["goal_diff"],
                "home_attack_form": home["attack"],
                "away_attack_form": away["attack"],
                "home_defense_form": home["defense"],
                "away_defense_form": away["defense"],
                "home_xg_form": home["xg_for"],
                "away_xg_form": away["xg_for"],
                "xg_form_diff": home["xg_for"] - away["xg_for"],
                "home_rest_days": home["rest"],
                "away_rest_days": away["rest"],
                "rest_diff": home["rest"] - away["rest"],
                "neutral_site": float(neutral_site),
                "home_advantage_flag": 0.0 if neutral_site else 1.0,
                "market_home_prob": float(market[0]),
                "market_draw_prob": float(market[1]),
                "market_away_prob": float(market[2]),
                "has_market_odds": has_market_odds,
                "market_overround": market_overround,
                "poisson_home_xg": poisson_home_xg,
                "poisson_away_xg": poisson_away_xg,
            }
        )

        ratings[home_team], ratings[away_team] = home_elo_post, away_elo_post

        if home_goals > away_goals:
            home_points, away_points = 3.0, 0.0
        elif home_goals < away_goals:
            home_points, away_points = 0.0, 3.0
        else:
            home_points, away_points = 1.0, 1.0

        home_xg = float(match["home_xg"]) if not pd.isna(match.get("home_xg")) else home_goals
        away_xg = float(match["away_xg"]) if not pd.isna(match.get("away_xg")) else away_goals

        state["points"][home_team].append(home_points)
        state["points"][away_team].append(away_points)
        state["goal_diff"][home_team].append(home_goals - away_goals)
        state["goal_diff"][away_team].append(away_goals - home_goals)
        state["goals_for"][home_team].append(home_goals)
        state["goals_for"][away_team].append(away_goals)
        state["goals_against"][home_team].append(away_goals)
        state["goals_against"][away_team].append(home_goals)
        state["xg_for"][home_team].append(home_xg)
        state["xg_for"][away_team].append(away_xg)
        state["xg_against"][home_team].append(away_xg)
        state["xg_against"][away_team].append(home_xg)
        state["last_date"][home_team] = current_date
        state["last_date"][away_team] = current_date

    return pd.DataFrame(rows)
