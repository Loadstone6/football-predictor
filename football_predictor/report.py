from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from football_predictor.explain import friendly_feature_name
from football_predictor.profiles import build_team_profiles


def _clean(value: Any):
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if pd.isna(value) if not isinstance(value, (list, tuple, dict)) else False:
        return None
    return value


def _records(frame: pd.DataFrame, limit: int | None = None) -> list[dict]:
    if frame.empty:
        return []
    data = frame.tail(limit).to_dict(orient="records") if limit else frame.to_dict(orient="records")
    return [_clean(row) for row in data]


def _competition_breakdown(scored: pd.DataFrame) -> list[dict]:
    if scored.empty:
        return []
    rows = []
    for competition, group in scored.groupby("competition"):
        probabilities = group[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
        actual = group["actual_outcome_id"].to_numpy(dtype=int)
        predicted = probabilities.argmax(axis=1)
        rows.append(
            {
                "competition": competition,
                "matches": int(len(group)),
                "accuracy": float((predicted == actual).mean()),
                "avg_confidence": float(probabilities.max(axis=1).mean()),
                "bet_count": int(group.get("has_market_odds", pd.Series(dtype=bool)).fillna(False).sum()),
            }
        )
    return sorted(rows, key=lambda item: item["matches"], reverse=True)


def build_report(backtest: dict, source_name: str) -> dict:
    predictions = backtest["predictions"]
    scored = backtest["scored_predictions"]
    latest = predictions.iloc[-1].to_dict()
    score_matrix = latest.get("score_matrix", [])
    feature_importance = [
        {
            **item,
            "label": friendly_feature_name(item["feature"]),
        }
        for item in backtest["feature_importance"][:14]
    ]

    report = {
        "generated_at": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": source_name,
        "model_name": backtest["model_name"],
        "summary": {
            **backtest["metrics"],
            "betting": backtest["bet_metrics"],
        },
        "latest_match": _clean(latest),
        "score_matrix": _clean(score_matrix),
        "predictions": _records(scored, limit=60),
        "bankroll": _records(backtest["bet_history"], limit=None),
        "betting_ledger": _records(backtest["bet_history"], limit=80),
        "calibration": _clean(backtest["calibration"]),
        "model_comparison": _clean(backtest["model_comparison"]),
        "competition_breakdown": _clean(_competition_breakdown(scored)),
        "team_profiles": _clean(build_team_profiles(backtest["featured"])),
        "feature_importance": _clean(feature_importance),
        "local_factors": _clean(backtest["local_factors"]),
        "betting_config": _clean(
            {
                "starting_bankroll": backtest["config"].bet_config.starting_bankroll,
                "edge_threshold": backtest["config"].bet_config.edge_threshold,
                "fractional_kelly": backtest["config"].bet_config.fractional_kelly,
                "max_stake_fraction": backtest["config"].bet_config.max_stake_fraction,
            }
        ),
        "notes": [
            "Walk-forward features are generated before updating Elo and form with the current match.",
            "Model rows are trained only on matches earlier than the predicted row.",
            "Betting simulation is value betting, not true arbitrage.",
            "The bundled sample data is illustrative and too small for betting conclusions.",
        ],
        "data_sources": [
            {
                "name": "Football-Data.co.uk",
                "url": "https://www.football-data.co.uk/",
                "use": "Historical club results and bookmaker odds CSVs.",
            },
            {
                "name": "Club Elo",
                "url": "https://clubelo.com/",
                "use": "Historical club strength ratings.",
            },
            {
                "name": "World Football Elo Ratings",
                "url": "https://eloratings.net/",
                "use": "National-team Elo reference.",
            },
        ],
    }
    return _clean(report)


def write_report(backtest: dict, output_path: str | Path, source_name: str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(backtest, source_name)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path
