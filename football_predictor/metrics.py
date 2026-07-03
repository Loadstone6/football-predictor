from __future__ import annotations

import numpy as np
import pandas as pd


def classification_metrics(
    predictions: pd.DataFrame,
    probability_columns: tuple[str, str, str] = ("p_home", "p_draw", "p_away"),
) -> dict:
    if predictions.empty:
        return {
            "matches": 0,
            "accuracy": 0.0,
            "log_loss": 0.0,
            "brier": 0.0,
            "exact_score_hit_rate": 0.0,
        }

    probabilities = predictions[list(probability_columns)].to_numpy(dtype=float)
    probabilities = np.clip(probabilities, 1e-9, 1.0)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    actual = predictions["actual_outcome_id"].to_numpy(dtype=int)
    pred = np.argmax(probabilities, axis=1)

    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(actual)), actual] = 1.0
    log_loss = -float(np.mean(np.log(probabilities[np.arange(len(actual)), actual])))
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    accuracy = float(np.mean(pred == actual))

    exact_hits = []
    if "top_score" in predictions.columns:
        for _, row in predictions.iterrows():
            exact_hits.append(f"{int(row['home_goals'])}-{int(row['away_goals'])}" == row.get("top_score"))

    return {
        "matches": int(len(predictions)),
        "accuracy": accuracy,
        "log_loss": log_loss,
        "brier": brier,
        "exact_score_hit_rate": float(np.mean(exact_hits)) if exact_hits else None,
    }


def calibration_bins(predictions: pd.DataFrame, bins: int = 8) -> list[dict]:
    if predictions.empty:
        return []

    probabilities = predictions[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    actual = predictions["actual_outcome_id"].to_numpy(dtype=int)
    confidence = probabilities.max(axis=1)
    correct = np.argmax(probabilities, axis=1) == actual
    cuts = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for left, right in zip(cuts[:-1], cuts[1:]):
        mask = (confidence >= left) & (confidence < right if right < 1.0 else confidence <= right)
        if mask.any():
            rows.append(
                {
                    "bin": f"{left:.2f}-{right:.2f}",
                    "count": int(mask.sum()),
                    "mean_confidence": float(confidence[mask].mean()),
                    "accuracy": float(correct[mask].mean()),
                }
            )
    return rows


def model_comparison(predictions: pd.DataFrame) -> list[dict]:
    rows = [
        {
            "name": "Ensemble",
            "kind": "model",
            **classification_metrics(predictions, ("p_home", "p_draw", "p_away")),
        },
        {
            "name": "Elo baseline",
            "kind": "baseline",
            **classification_metrics(predictions, ("elo_home_prob", "elo_draw_prob", "elo_away_prob")),
        },
    ]

    if "has_market_odds" in predictions.columns:
        market_predictions = predictions[predictions["has_market_odds"] == True].copy()
    else:
        market_predictions = predictions

    if not market_predictions.empty:
        rows.append(
            {
                "name": "Market no-vig",
                "kind": "market",
                **classification_metrics(
                    market_predictions,
                    ("market_home_prob", "market_draw_prob", "market_away_prob"),
                ),
            }
        )
    return rows
