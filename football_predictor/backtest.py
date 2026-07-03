from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from football_predictor.betting import BetConfig, simulate_value_betting
from football_predictor.explain import local_factor_summary
from football_predictor.features import FEATURE_COLUMNS, FeatureConfig, build_walk_forward_features
from football_predictor.metrics import calibration_bins, classification_metrics, model_comparison
from football_predictor.models import OutcomeModel, train_outcome_model
from football_predictor.simulation import poisson_score_matrix, top_scoreline


@dataclass(frozen=True)
class BacktestConfig:
    min_train_matches: int = 80
    refit_frequency: int = 10
    feature_config: FeatureConfig = FeatureConfig()
    bet_config: BetConfig = BetConfig()


def _blend_probabilities(model_probability: np.ndarray, poisson_probability: tuple[float, float, float]) -> np.ndarray:
    poisson = np.asarray(poisson_probability, dtype=float)
    blended = 0.72 * model_probability + 0.28 * poisson
    blended = np.clip(blended, 1e-6, 1.0)
    return blended / blended.sum()


def walk_forward_backtest(matches: pd.DataFrame, config: BacktestConfig | None = None) -> dict:
    config = config or BacktestConfig()
    featured = build_walk_forward_features(matches, feature_config=config.feature_config)

    predictions = []
    model: OutcomeModel | None = None
    model_last_fit_idx = -1
    model_name = "elo_warmup"

    for idx in range(len(featured)):
        row = featured.iloc[[idx]]
        if idx >= config.min_train_matches and (
            model is None or idx - model_last_fit_idx >= config.refit_frequency
        ):
            training = featured.iloc[:idx].copy()
            model = train_outcome_model(training, FEATURE_COLUMNS)
            model_last_fit_idx = idx
            model_name = model.model_name

        if model is None:
            model_probability = row[["elo_home_prob", "elo_draw_prob", "elo_away_prob"]].to_numpy(dtype=float)[0]
        else:
            model_probability = model.predict_proba(row)[0]

        score_matrix = poisson_score_matrix(
            row["poisson_home_xg"].iloc[0],
            row["poisson_away_xg"].iloc[0],
            max_goals=7,
        )
        home_win = float(np.tril(score_matrix, k=-1).sum())
        draw = float(np.trace(score_matrix))
        away_win = float(np.triu(score_matrix, k=1).sum())
        probabilities = _blend_probabilities(model_probability, (home_win, draw, away_win))
        home_goals_top, away_goals_top, top_probability = top_scoreline(score_matrix)

        original = featured.iloc[idx].to_dict()
        predictions.append(
            {
                **original,
                "model_name": model_name,
                "p_home": float(probabilities[0]),
                "p_draw": float(probabilities[1]),
                "p_away": float(probabilities[2]),
                "top_score": f"{home_goals_top}-{away_goals_top}",
                "top_score_probability": top_probability,
                "score_matrix": score_matrix.tolist(),
            }
        )

    predictions_frame = pd.DataFrame(predictions)
    scored = predictions_frame.iloc[config.min_train_matches :].reset_index(drop=True)
    metrics = classification_metrics(scored)
    comparison = model_comparison(scored)
    bet_history, bet_metrics = simulate_value_betting(scored, config.bet_config)

    final_model = model or train_outcome_model(featured, FEATURE_COLUMNS)
    feature_importance = final_model.feature_importance()
    latest_row = predictions_frame.iloc[-1]
    local_factors = local_factor_summary(latest_row, featured, feature_importance)

    return {
        "featured": featured,
        "predictions": predictions_frame,
        "scored_predictions": scored,
        "metrics": metrics,
        "model_comparison": comparison,
        "calibration": calibration_bins(scored),
        "bet_history": bet_history,
        "bet_metrics": bet_metrics,
        "feature_importance": feature_importance,
        "local_factors": local_factors,
        "model_name": final_model.model_name,
        "config": config,
    }
