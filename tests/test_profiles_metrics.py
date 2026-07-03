from __future__ import annotations

from pathlib import Path

import pandas as pd

from football_predictor.data import load_matches
from football_predictor.features import build_walk_forward_features
from football_predictor.metrics import model_comparison
from football_predictor.profiles import build_team_profiles


ROOT = Path(__file__).resolve().parents[1]


def test_team_profiles_include_rank_and_recent_form():
    matches = load_matches(ROOT / "data" / "sample_matches.csv")
    featured = build_walk_forward_features(matches)
    profiles = build_team_profiles(featured)

    assert profiles
    assert profiles[0]["rank"] == 1
    assert profiles[0]["team"]
    assert set(profiles[0]["form"]).issubset({"W", "D", "L"})


def test_model_comparison_includes_ensemble_elo_and_market():
    predictions = pd.DataFrame(
        [
            {
                "actual_outcome_id": 0,
                "home_goals": 2,
                "away_goals": 1,
                "top_score": "2-1",
                "p_home": 0.55,
                "p_draw": 0.25,
                "p_away": 0.20,
                "elo_home_prob": 0.50,
                "elo_draw_prob": 0.27,
                "elo_away_prob": 0.23,
                "market_home_prob": 0.52,
                "market_draw_prob": 0.25,
                "market_away_prob": 0.23,
                "has_market_odds": True,
            },
            {
                "actual_outcome_id": 2,
                "home_goals": 0,
                "away_goals": 1,
                "top_score": "1-1",
                "p_home": 0.30,
                "p_draw": 0.25,
                "p_away": 0.45,
                "elo_home_prob": 0.35,
                "elo_draw_prob": 0.28,
                "elo_away_prob": 0.37,
                "market_home_prob": 0.33,
                "market_draw_prob": 0.29,
                "market_away_prob": 0.38,
                "has_market_odds": True,
            },
        ]
    )

    names = {row["name"] for row in model_comparison(predictions)}
    assert names == {"Ensemble", "Elo baseline", "Market no-vig"}
