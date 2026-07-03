from __future__ import annotations

import numpy as np
import pandas as pd


FRIENDLY_NAMES = {
    "home_elo": "Home Elo",
    "away_elo": "Away Elo",
    "elo_diff": "Elo gap",
    "elo_home_prob": "Elo home probability",
    "elo_draw_prob": "Elo draw probability",
    "elo_away_prob": "Elo away probability",
    "home_recent_points": "Home recent points",
    "away_recent_points": "Away recent points",
    "recent_points_diff": "Recent-points gap",
    "home_goal_diff_form": "Home goal-difference form",
    "away_goal_diff_form": "Away goal-difference form",
    "goal_diff_form_diff": "Goal-difference form gap",
    "home_attack_form": "Home attack form",
    "away_attack_form": "Away attack form",
    "home_defense_form": "Home defensive concession form",
    "away_defense_form": "Away defensive concession form",
    "home_xg_form": "Home xG form",
    "away_xg_form": "Away xG form",
    "xg_form_diff": "xG form gap",
    "home_rest_days": "Home rest",
    "away_rest_days": "Away rest",
    "rest_diff": "Rest gap",
    "neutral_site": "Neutral venue",
    "home_advantage_flag": "Home advantage",
    "market_home_prob": "Market home probability",
    "market_draw_prob": "Market draw probability",
    "market_away_prob": "Market away probability",
    "poisson_home_xg": "Home goal prior",
    "poisson_away_xg": "Away goal prior",
}


def friendly_feature_name(feature: str) -> str:
    return FRIENDLY_NAMES.get(feature, feature.replace("_", " ").title())


def local_factor_summary(
    row: pd.Series,
    reference_frame: pd.DataFrame,
    global_importance: list[dict],
    limit: int = 8,
) -> list[dict]:
    if reference_frame.empty:
        return []

    ranked = {item["feature"]: item["importance"] for item in global_importance}
    rows = []
    for feature, importance in ranked.items():
        if feature not in reference_frame.columns or feature not in row:
            continue
        values = pd.to_numeric(reference_frame[feature], errors="coerce")
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        if not np.isfinite(std) or std <= 1e-9:
            std = 1.0
        value = float(row[feature])
        score = ((value - mean) / std) * float(importance)
        rows.append(
            {
                "feature": feature,
                "label": friendly_feature_name(feature),
                "value": value,
                "direction_score": float(score),
                "importance": float(importance),
            }
        )

    rows.sort(key=lambda item: abs(item["direction_score"]), reverse=True)
    return rows[:limit]
