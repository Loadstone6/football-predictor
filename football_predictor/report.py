from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from football_predictor.explain import friendly_feature_name
from football_predictor.profiles import build_team_profiles
from football_predictor.simulation import dixon_coles_score_matrix, score_market_probabilities, top_scoreline


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


def _fair_odds(probability: float) -> float | None:
    if probability <= 0:
        return None
    return float(1.0 / probability)


def _confidence(row: pd.Series, data_quality: dict) -> dict:
    probabilities = np.array([row["p_home"], row["p_draw"], row["p_away"]], dtype=float)
    sharpness = float(probabilities.max() - probabilities.min())
    market_gap = max(
        abs(float(row["p_home"]) - float(row["market_home_prob"])),
        abs(float(row["p_draw"]) - float(row["market_draw_prob"])),
        abs(float(row["p_away"]) - float(row["market_away_prob"])),
    )
    data_score = 0.55 + 0.30 * float(data_quality.get("odds_coverage", 0.0)) + 0.15 * float(
        data_quality.get("xg_coverage", 0.0)
    )
    score = 100.0 * np.clip(0.50 * sharpness + 0.35 * data_score + 0.15 * (1.0 - min(market_gap, 0.35)), 0, 1)
    if score >= 68:
        label = "High"
    elif score >= 48:
        label = "Medium"
    else:
        label = "Low"
    return {"score": float(score), "label": label, "market_gap": float(market_gap), "sharpness": sharpness}


def _top_scores(score_matrix: list[list[float]], limit: int = 5) -> list[dict]:
    matrix = np.asarray(score_matrix, dtype=float)
    rows = []
    for home_goals in range(matrix.shape[0]):
        for away_goals in range(matrix.shape[1]):
            rows.append(
                {
                    "score": f"{home_goals}-{away_goals}",
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "probability": float(matrix[home_goals, away_goals]),
                }
            )
    return sorted(rows, key=lambda item: item["probability"], reverse=True)[:limit]


def _outcome_edges(row: pd.Series) -> list[dict]:
    outcomes = [
        ("home", row["home_team"], row["p_home"], row["market_home_prob"], row.get("home_odds")),
        ("draw", "Draw", row["p_draw"], row["market_draw_prob"], row.get("draw_odds")),
        ("away", row["away_team"], row["p_away"], row["market_away_prob"], row.get("away_odds")),
    ]
    return [
        {
            "outcome": key,
            "label": label,
            "model_probability": float(model_probability),
            "market_probability": float(market_probability),
            "edge": float(model_probability - market_probability),
            "fair_odds": _fair_odds(float(model_probability)),
            "bookmaker_odds": None if pd.isna(odds) else float(odds),
        }
        for key, label, model_probability, market_probability, odds in outcomes
    ]


def _plain_english_reason(row: pd.Series) -> str:
    home = row["home_team"]
    away = row["away_team"]
    favourite = home if row["p_home"] >= row["p_away"] else away
    edge = float(row["p_home"] - row["p_away"])
    total_goals = float(row.get("expected_total_goals", row.get("poisson_home_xg", 0) + row.get("poisson_away_xg", 0)))
    if abs(edge) < 0.08:
        result_phrase = "The model sees this as a relatively balanced match"
    else:
        result_phrase = f"The model gives {favourite} the stronger result profile"
    goal_phrase = "with an elevated goal expectation" if total_goals >= 2.7 else "with a controlled goal expectation"
    market_gap = max(abs(item["edge"]) for item in _outcome_edges(row))
    market_phrase = (
        "There is meaningful disagreement with the market."
        if market_gap >= 0.06
        else "The model and market are broadly aligned."
    )
    return (
        f"{result_phrase} {goal_phrase}. The main signals are Elo strength, recent goal-form, "
        f"home advantage, rest, and market-implied probabilities. {market_phrase}"
    )


def _prediction_card(row: pd.Series, data_quality: dict, include_actual: bool = True) -> dict:
    confidence = _confidence(row, data_quality)
    top_scores = _top_scores(row.get("score_matrix", []), limit=5)
    probability_values = [float(row["p_home"]), float(row["p_draw"]), float(row["p_away"])]
    best_index = int(np.argmax(probability_values))
    best_key = ["home", "draw", "away"][best_index]
    card = {
        "id": f"{row['date']}_{row['home_team']}_{row['away_team']}",
        "date": row["date"],
        "competition": row["competition"],
        "season": row.get("season"),
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "probabilities": {
            "home": float(row["p_home"]),
            "draw": float(row["p_draw"]),
            "away": float(row["p_away"]),
        },
        "market_probabilities": {
            "home": float(row["market_home_prob"]),
            "draw": float(row["market_draw_prob"]),
            "away": float(row["market_away_prob"]),
        },
        "edges": _outcome_edges(row),
        "best_outcome": best_key,
        "predicted_score": row.get("top_score"),
        "top_scores": top_scores,
        "expected_goals": {
            "home": float(row.get("expected_home_goals", row.get("poisson_home_xg", 0.0))),
            "away": float(row.get("expected_away_goals", row.get("poisson_away_xg", 0.0))),
            "total": float(row.get("expected_total_goals", 0.0)),
        },
        "markets": {
            "over_1_5": float(row.get("over_1_5", 0.0)),
            "over_2_5": float(row.get("over_2_5", 0.0)),
            "over_3_5": float(row.get("over_3_5", 0.0)),
            "btts": float(row.get("btts", 0.0)),
            "home_clean_sheet": float(row.get("home_clean_sheet", 0.0)),
            "away_clean_sheet": float(row.get("away_clean_sheet", 0.0)),
        },
        "confidence": confidence,
        "explanation": _plain_english_reason(row),
        "risk_factors": [
            "Current public dataset has no verified lineup, injury, suspension, referee, weather, or real xG feed.",
            "Bookmaker odds already include information that is not present in the model features.",
            "Backtest performance is historical and may not persist.",
        ],
        "data_freshness": data_quality.get("date_end"),
        "model_version": "ensemble-rf-dc-market-v0.4",
        "score_matrix": row.get("score_matrix"),
    }
    if include_actual:
        card["actual"] = {
            "home_goals": int(row["home_goals"]),
            "away_goals": int(row["away_goals"]),
            "result": row["result"],
            "actual_probability": probability_values[int(row["actual_outcome_id"])],
            "directionally_correct": best_index == int(row["actual_outcome_id"]),
        }
    return card


def _synthetic_fixture_cards(team_profiles: list[dict], data_quality: dict, count: int = 18) -> list[dict]:
    teams = sorted(team_profiles, key=lambda team: team.get("power_score", 0), reverse=True)[:18]
    if len(teams) < 2:
        return []
    pairs = []
    for index in range(0, min(len(teams) - 1, count), 2):
        pairs.append((teams[index], teams[index + 1]))
    for index in range(1, min(len(teams) - 3, count), 4):
        pairs.append((teams[index], teams[-index - 1]))

    rows = []
    base_date = pd.Timestamp.utcnow().normalize() + pd.Timedelta(days=1)
    for idx, (home, away) in enumerate(pairs[:count]):
        elo_diff = float(home["elo"] - away["elo"])
        home_xg = float(np.clip(0.55 * home["attack_form"] + 0.25 * away["defense_form"] + 0.20 * home["xg_form"] + 0.12 + 0.0018 * elo_diff, 0.25, 3.75))
        away_xg = float(np.clip(0.55 * away["attack_form"] + 0.25 * home["defense_form"] + 0.20 * away["xg_form"] - 0.0012 * elo_diff, 0.25, 3.75))
        matrix = dixon_coles_score_matrix(home_xg, away_xg, max_goals=7)
        home_win = float(np.tril(matrix, k=-1).sum())
        draw = float(np.trace(matrix))
        away_win = float(np.triu(matrix, k=1).sum())
        total = home_win + draw + away_win
        top_home, top_away, top_probability = top_scoreline(matrix)
        pseudo_row = pd.Series(
            {
                "date": base_date + pd.Timedelta(days=idx),
                "competition": "Forecast Queue",
                "season": "upcoming",
                "home_team": home["team"],
                "away_team": away["team"],
                "p_home": home_win / total,
                "p_draw": draw / total,
                "p_away": away_win / total,
                "market_home_prob": home_win / total,
                "market_draw_prob": draw / total,
                "market_away_prob": away_win / total,
                "top_score": f"{top_home}-{top_away}",
                "top_score_probability": top_probability,
                "expected_home_goals": home_xg,
                "expected_away_goals": away_xg,
                "expected_total_goals": home_xg + away_xg,
                "score_matrix": matrix.tolist(),
                "home_odds": _fair_odds(home_win / total),
                "draw_odds": _fair_odds(draw / total),
                "away_odds": _fair_odds(away_win / total),
                **score_market_probabilities(matrix),
            }
        )
        rows.append(_prediction_card(pseudo_row, data_quality, include_actual=False))
    return rows


def _similar_matches(scored: pd.DataFrame, target: pd.Series, data_quality: dict, limit: int = 8) -> list[dict]:
    if scored.empty:
        return []
    columns = ["p_home", "p_draw", "p_away", "expected_home_goals", "expected_away_goals", "elo_diff"]
    available = [column for column in columns if column in scored.columns and column in target.index]
    if not available:
        return []
    values = scored[available].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    target_values = pd.to_numeric(target[available], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    scale = values.std(ddof=0).replace(0, 1.0).to_numpy(dtype=float)
    distances = np.sqrt((((values.to_numpy(dtype=float) - target_values) / scale) ** 2).sum(axis=1))
    nearest = scored.assign(similarity=1.0 / (1.0 + distances)).sort_values("similarity", ascending=False).head(limit)
    rows = []
    for _, row in nearest.iterrows():
        card = _prediction_card(row, data_quality, include_actual=True)
        card["similarity"] = float(row["similarity"])
        rows.append(card)
    return rows


def _team_analytics(featured: pd.DataFrame, profiles: list[dict]) -> list[dict]:
    rows = []
    for profile in profiles:
        team = profile["team"]
        matches = featured[(featured["home_team"] == team) | (featured["away_team"] == team)].tail(14)
        trend = []
        for _, match in matches.iterrows():
            is_home = match["home_team"] == team
            goals_for = float(match["home_goals"] if is_home else match["away_goals"])
            goals_against = float(match["away_goals"] if is_home else match["home_goals"])
            trend.append(
                {
                    "date": match["date"],
                    "opponent": match["away_team"] if is_home else match["home_team"],
                    "goals_for": goals_for,
                    "goals_against": goals_against,
                    "goal_diff": goals_for - goals_against,
                    "venue": "home" if is_home else "away",
                }
            )
        rows.append({**profile, "trend": _clean(trend), "availability": "No verified injury/lineup feed connected"})
    return rows


def _league_analytics(featured: pd.DataFrame, scored: pd.DataFrame, profiles: list[dict]) -> list[dict]:
    team_power = {profile["team"]: profile["power_score"] for profile in profiles}
    leagues = []
    for competition, group in featured.groupby("competition"):
        teams = sorted(set(group["home_team"]).union(group["away_team"]))
        goals = group["home_goals"] + group["away_goals"]
        scored_group = scored[scored["competition"] == competition]
        if scored_group.empty:
            accuracy = None
            log_loss = None
        else:
            probabilities = scored_group[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
            actual = scored_group["actual_outcome_id"].to_numpy(dtype=int)
            accuracy = float((probabilities.argmax(axis=1) == actual).mean())
            log_loss = float(-np.mean(np.log(np.clip(probabilities[np.arange(len(actual)), actual], 1e-9, 1.0))))
        standings = []
        for team in teams:
            team_rows = group[(group["home_team"] == team) | (group["away_team"] == team)]
            points = 0
            goal_diff = 0
            for _, match in team_rows.iterrows():
                is_home = match["home_team"] == team
                gf = int(match["home_goals"] if is_home else match["away_goals"])
                ga = int(match["away_goals"] if is_home else match["home_goals"])
                goal_diff += gf - ga
                points += 3 if gf > ga else 1 if gf == ga else 0
            standings.append(
                {
                    "team": team,
                    "matches": int(len(team_rows)),
                    "points": points,
                    "goal_diff": goal_diff,
                    "power_score": float(team_power.get(team, 0.0)),
                }
            )
        leagues.append(
            {
                "competition": competition,
                "teams": len(teams),
                "matches": int(len(group)),
                "avg_goals": float(goals.mean()),
                "home_win_rate": float((group["home_goals"] > group["away_goals"]).mean()),
                "draw_rate": float((group["home_goals"] == group["away_goals"]).mean()),
                "accuracy": accuracy,
                "log_loss": log_loss,
                "standings": sorted(standings, key=lambda item: (item["points"], item["goal_diff"]), reverse=True)[:12],
            }
        )
    return sorted(leagues, key=lambda item: item["matches"], reverse=True)


def _backtest_dashboard(scored: pd.DataFrame) -> dict:
    if scored.empty:
        return {"confusion_matrix": [], "confidence_buckets": [], "rolling": [], "archive": []}
    probabilities = scored[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    actual = scored["actual_outcome_id"].to_numpy(dtype=int)
    predicted = probabilities.argmax(axis=1)
    matrix = np.zeros((3, 3), dtype=int)
    for truth, guess in zip(actual, predicted):
        matrix[int(truth), int(guess)] += 1
    confidence = probabilities.max(axis=1)
    buckets = []
    for left, right in zip(np.arange(0.30, 0.91, 0.10), np.arange(0.40, 1.01, 0.10)):
        mask = (confidence >= left) & (confidence < right)
        if mask.any():
            buckets.append(
                {
                    "bucket": f"{left:.1f}-{right:.1f}",
                    "count": int(mask.sum()),
                    "accuracy": float((predicted[mask] == actual[mask]).mean()),
                    "mean_confidence": float(confidence[mask].mean()),
                }
            )
    rolling = []
    chunk_size = max(100, len(scored) // 24)
    for start in range(0, len(scored), chunk_size):
        chunk = scored.iloc[start : start + chunk_size]
        if len(chunk) < 30:
            continue
        probs = chunk[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
        truth = chunk["actual_outcome_id"].to_numpy(dtype=int)
        rolling.append(
            {
                "date": chunk["date"].max(),
                "accuracy": float((probs.argmax(axis=1) == truth).mean()),
                "log_loss": float(-np.mean(np.log(np.clip(probs[np.arange(len(truth)), truth], 1e-9, 1.0)))),
                "brier": float(np.mean(np.sum((probs - np.eye(3)[truth]) ** 2, axis=1))),
            }
        )
    return {
        "confusion_matrix": matrix.tolist(),
        "confidence_buckets": buckets,
        "rolling": _clean(rolling),
        "archive": _records(scored.tail(200), limit=None),
    }


def _model_registry(backtest: dict) -> list[dict]:
    return [
        {
            "name": "Market no-vig baseline",
            "version": "market-v0.1",
            "status": "benchmark",
            "role": "Reference probability after bookmaker margin removal",
        },
        {
            "name": "Elo strength model",
            "version": "elo-v0.2",
            "status": "active",
            "role": "Team strength and home-advantage prior",
        },
        {
            "name": "Dixon-Coles score model",
            "version": "dc-v0.1",
            "status": "active",
            "role": "Low-score-adjusted scoreline matrix, goals, BTTS and clean-sheet markets",
        },
        {
            "name": str(backtest.get("model_name", "ensemble")),
            "version": "ensemble-rf-dc-market-v0.4",
            "status": "active",
            "role": "Walk-forward ensemble outcome probabilities",
        },
        {
            "name": "Bayesian attack/defence",
            "version": "planned",
            "status": "research",
            "role": "Uncertainty-aware team attack and defence strength",
        },
    ]


def _leakage_audit() -> list[dict]:
    return [
        {"rule": "Chronological feature generation", "status": "pass", "detail": "Elo and form are updated only after each row is predicted."},
        {"rule": "Walk-forward training", "status": "pass", "detail": "Each backtest prediction trains on rows strictly earlier than the predicted row."},
        {"rule": "No final-season aggregates", "status": "pass", "detail": "Rolling features use prior matches only, not season-end tables."},
        {"rule": "Odds timestamp limitation", "status": "warning", "detail": "Football-Data odds are treated as available pre-match but exact capture timestamps are not present."},
        {"rule": "Lineup/injury limitation", "status": "warning", "detail": "No verified timestamped lineup or injury feed is connected yet."},
    ]


def _responsible_use() -> list[str]:
    return [
        "Analytics only; this is not financial advice.",
        "Positive expected value in a model does not guarantee profit.",
        "Backtests can overfit and future performance may differ.",
        "The market already includes information unavailable to this public-data prototype.",
        "Use bankroll simulations as risk illustrations, not staking instructions.",
    ]


def build_report(backtest: dict, source_name: str) -> dict:
    predictions = backtest["predictions"]
    scored = backtest["scored_predictions"]
    latest = predictions.iloc[-1].to_dict()
    score_matrix = latest.get("score_matrix", [])
    data_quality = _data_quality(backtest["featured"], scored)
    team_profiles = build_team_profiles(backtest["featured"])
    latest_scored = scored.iloc[-1]
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
        "predictions": _records(scored, limit=120),
        "dashboard": {
            "today": _clean(_synthetic_fixture_cards(team_profiles, data_quality, count=6)),
            "tomorrow": _clean(_synthetic_fixture_cards(team_profiles[6:] + team_profiles[:6], data_quality, count=6)),
            "top_picks": _clean([
                _prediction_card(row, data_quality, include_actual=True)
                for _, row in scored.tail(180).sort_values(["p_home", "p_draw", "p_away"], ascending=False).head(8).iterrows()
            ]),
            "market_disagreements": _clean([
                _prediction_card(row, data_quality, include_actual=True)
                for _, row in scored.tail(300).assign(
                    max_edge=lambda frame: np.maximum.reduce(
                        [
                            abs(frame["p_home"] - frame["market_home_prob"]),
                            abs(frame["p_draw"] - frame["market_draw_prob"]),
                            abs(frame["p_away"] - frame["market_away_prob"]),
                        ]
                    )
                ).sort_values("max_edge", ascending=False).head(8).iterrows()
            ]),
        },
        "fixture_cards": _clean([_prediction_card(row, data_quality, include_actual=True) for _, row in scored.tail(180).iterrows()]),
        "forecast_fixtures": _clean(_synthetic_fixture_cards(team_profiles, data_quality, count=18)),
        "match_detail": _clean(
            {
                **_prediction_card(latest_scored, data_quality, include_actual=True),
                "similar_matches": _similar_matches(scored.iloc[:-1], latest_scored, data_quality),
            }
        ),
        "bankroll": _records(backtest["bet_history"], limit=None),
        "betting_ledger": _records(backtest["bet_history"], limit=80),
        "calibration": _clean(backtest["calibration"]),
        "model_comparison": _clean(backtest["model_comparison"]),
        "competition_breakdown": _clean(_competition_breakdown(scored)),
        "data_quality": _clean(data_quality),
        "team_profiles": _clean(team_profiles),
        "team_analytics": _clean(_team_analytics(backtest["featured"], team_profiles)),
        "league_analytics": _clean(_league_analytics(backtest["featured"], scored, team_profiles)),
        "backtest_dashboard": _clean(_backtest_dashboard(scored)),
        "model_registry": _clean(_model_registry(backtest)),
        "leakage_audit": _clean(_leakage_audit()),
        "responsible_use": _clean(_responsible_use()),
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
