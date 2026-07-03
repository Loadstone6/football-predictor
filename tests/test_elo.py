from __future__ import annotations

from football_predictor.elo import EloConfig, three_way_probabilities, update_ratings


def test_elo_probabilities_sum_to_one():
    probs = three_way_probabilities(1600, 1500, False, EloConfig())
    assert round(sum(probs), 10) == 1.0
    assert probs[0] > probs[2]


def test_elo_winner_gains_points():
    home, away = update_ratings(1500, 1500, 2, 0, False, EloConfig())
    assert home > 1500
    assert away < 1500
