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


def _data_quality(featured: pd.DataFrame, scored: pd.DataFrame) -> dict:
    if featured.empty:
        return {
            "matches": 0,
            "scored_matches": 0,
            "teams": 0,
            "competitions": 0,
            "date_start": None,
            "date_end": None,
            "odds_coverage": 0.0,
            "xg_coverage": 0.0,
            "duplicate_matches": 0,
            "warnings": ["No match rows available."],
        }

    odds_columns = ["home_odds", "draw_odds", "away_odds"]
    has_odds = featured[odds_columns].notna().all(axis=1) if all(col in featured for col in odds_columns) else False
    has_xg = featured[["home_xg", "away_xg"]].notna().all(axis=1) if {"home_xg", "away_xg"}.issubset(featured.columns) else False
    duplicates = int(featured.duplicated(subset=["date", "competition", "home_team", "away_team"]).sum())
    teams = int(featured[["home_team", "away_team"]].stack().nunique())
    competitions = int(featured["competition"].nunique())

    odds_coverage = float(has_odds.mean()) if hasattr(has_odds, "mean") else 0.0
    xg_coverage = float(has_xg.mean()) if hasattr(has_xg, "mean") else 0.0
    warnings = []
    if odds_coverage < 0.85:
        warnings.append("Odds coverage is below 85%; betting simulation may be sparse.")
    if xg_coverage < 0.25:
        warnings.append("xG coverage is low; the goal model falls back to observed goals.")
    if duplicates:
        warnings.append(f"{duplicates} duplicate match rows were detected.")
    if len(scored) < 250:
        warnings.append("Backtest sample is small for betting conclusions.")

    return {
        "matches": int(len(featured)),
        "scored_matches": int(len(scored)),
        "teams": teams,
        "competitions": competitions,
        "date_start": featured["date"].min(),
        "date_end": featured["date"].max(),
        "odds_coverage": odds_coverage,
        "xg_coverage": xg_coverage,
        "duplicate_matches": duplicates,
        "warnings": warnings,
    }


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
        "data_quality": _clean(_data_quality(backtest["featured"], scored)),
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
