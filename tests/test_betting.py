from __future__ import annotations

import pandas as pd

from football_predictor.betting import BetConfig, expected_value_per_unit, normalized_implied_probabilities, simulate_value_betting


def test_normalized_implied_probabilities_remove_margin():
    probs = normalized_implied_probabilities((2.0, 3.5, 4.0))
    assert probs[0] > probs[1] > probs[2]
    assert round(sum(probs), 10) == 1.0


def test_expected_value_positive_when_model_exceeds_price():
    assert expected_value_per_unit(0.70, 2.50) > 0
    assert expected_value_per_unit(0.40, 2.00) < 0


def test_value_betting_places_positive_edge_bet():
    predictions = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "home_team": "England",
                "away_team": "Spain",
                "actual_outcome_id": 0,
                "p_home": 0.70,
                "p_draw": 0.15,
                "p_away": 0.15,
                "home_odds": 2.50,
                "draw_odds": 3.40,
                "away_odds": 2.90,
            }
        ]
    )
    history, metrics = simulate_value_betting(predictions, BetConfig(edge_threshold=0.02))
    assert history.iloc[0]["placed"] == True
    assert metrics["bets"] == 1
    assert metrics["ending_bankroll"] > metrics["starting_bankroll"]
