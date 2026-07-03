from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football_predictor.betting import BetConfig, simulate_value_betting
from football_predictor.data import load_matches
from football_predictor.features import FEATURE_COLUMNS, build_walk_forward_features
from football_predictor.metrics import classification_metrics
from football_predictor.simulation import poisson_score_matrix


MARKET_FEATURES = {"market_home_prob", "market_draw_prob", "market_away_prob"}


@dataclass(frozen=True)
class Candidate:
    name: str
    factory: Callable[[], object]
    feature_columns: list[str]


def _fill_values(frame: pd.DataFrame, feature_columns: list[str]) -> pd.Series:
    fill = frame[feature_columns].apply(pd.to_numeric, errors="coerce").median(numeric_only=True)
    return fill.fillna(0.0)


def _prepare(frame: pd.DataFrame, feature_columns: list[str], fill_values: pd.Series) -> pd.DataFrame:
    return frame[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(fill_values)


def _full_probabilities(estimator: object, x: pd.DataFrame) -> np.ndarray:
    raw = estimator.predict_proba(x)
    full = np.zeros((len(x), 3), dtype=float)
    classes = getattr(estimator, "classes_", np.array([0, 1, 2]))
    for pos, cls in enumerate(classes):
        full[:, int(cls)] = raw[:, pos]
    full = np.clip(full, 1e-9, 1.0)
    return full / full.sum(axis=1, keepdims=True)


def _poisson_probabilities(frame: pd.DataFrame) -> np.ndarray:
    rows = []
    for _, row in frame.iterrows():
        matrix = poisson_score_matrix(row["poisson_home_xg"], row["poisson_away_xg"], max_goals=7)
        home = float(np.tril(matrix, k=-1).sum())
        draw = float(np.trace(matrix))
        away = float(np.triu(matrix, k=1).sum())
        total = home + draw + away
        rows.append([home / total, draw / total, away / total])
    return np.asarray(rows, dtype=float)


def _blend(model_probabilities: np.ndarray, poisson_probabilities: np.ndarray, weight: float = 0.72) -> np.ndarray:
    blended = weight * model_probabilities + (1.0 - weight) * poisson_probabilities
    blended = np.clip(blended, 1e-9, 1.0)
    return blended / blended.sum(axis=1, keepdims=True)


def _prediction_frame(scored: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    out = scored.copy()
    out["p_home"] = probabilities[:, 0]
    out["p_draw"] = probabilities[:, 1]
    out["p_away"] = probabilities[:, 2]
    return out


def _score(name: str, scored: pd.DataFrame, probabilities: np.ndarray, bet_config: BetConfig) -> dict:
    frame = _prediction_frame(scored, probabilities)
    metrics = classification_metrics(frame)
    _, bet_metrics = simulate_value_betting(frame, bet_config)
    return {
        "model": name,
        **metrics,
        "bets": bet_metrics["bets"],
        "roi": bet_metrics["roi"],
        "profit": bet_metrics["total_profit"],
        "max_drawdown": bet_metrics["max_drawdown"],
        "ending_bankroll": bet_metrics["ending_bankroll"],
    }


def _walk_forward_candidate(
    featured: pd.DataFrame,
    candidate: Candidate,
    min_train: int,
    refit_frequency: int,
) -> np.ndarray:
    probabilities = np.full((len(featured), 3), np.nan, dtype=float)
    for start in range(min_train, len(featured), refit_frequency):
        end = min(len(featured), start + refit_frequency)
        training = featured.iloc[:start]
        testing = featured.iloc[start:end]
        y = training["actual_outcome_id"].astype(int)
        fill = _fill_values(training, candidate.feature_columns)
        x_train = _prepare(training, candidate.feature_columns, fill)
        x_test = _prepare(testing, candidate.feature_columns, fill)
        estimator = candidate.factory()
        estimator.fit(x_train, y)
        probabilities[start:end] = _full_probabilities(estimator, x_test)
    return probabilities[min_train:]


def _candidate_factories(feature_columns: list[str], no_market_columns: list[str]) -> list[Candidate]:
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    candidates = [
        Candidate(
            "logistic_all",
            lambda: make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, C=0.6, class_weight="balanced", random_state=42),
            ),
            feature_columns,
        ),
        Candidate(
            "logistic_no_market",
            lambda: make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, C=0.6, class_weight="balanced", random_state=42),
            ),
            no_market_columns,
        ),
        Candidate(
            "random_forest_all",
            lambda: RandomForestClassifier(
                n_estimators=120,
                max_depth=6,
                min_samples_leaf=4,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
            feature_columns,
        ),
        Candidate(
            "extra_trees_all",
            lambda: ExtraTreesClassifier(
                n_estimators=180,
                max_depth=6,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            feature_columns,
        ),
    ]

    try:
        from sklearn.ensemble import HistGradientBoostingClassifier

        candidates.extend(
            [
                Candidate(
                    "hist_gradient_all",
                    lambda: HistGradientBoostingClassifier(
                        max_iter=90,
                        learning_rate=0.045,
                        max_leaf_nodes=15,
                        l2_regularization=0.12,
                        random_state=42,
                    ),
                    feature_columns,
                ),
                Candidate(
                    "hist_gradient_no_market",
                    lambda: HistGradientBoostingClassifier(
                        max_iter=90,
                        learning_rate=0.045,
                        max_leaf_nodes=15,
                        l2_regularization=0.12,
                        random_state=42,
                    ),
                    no_market_columns,
                ),
            ]
        )
    except Exception:
        pass

    try:
        from xgboost import XGBClassifier

        candidates.extend(
            [
                Candidate(
                    "xgboost_current_all",
                    lambda: XGBClassifier(
                        n_estimators=160,
                        max_depth=3,
                        learning_rate=0.045,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        random_state=42,
                        n_jobs=-1,
                        tree_method="hist",
                    ),
                    feature_columns,
                ),
                Candidate(
                    "xgboost_conservative_all",
                    lambda: XGBClassifier(
                        n_estimators=80,
                        max_depth=2,
                        learning_rate=0.035,
                        subsample=0.75,
                        colsample_bytree=0.75,
                        reg_alpha=0.2,
                        reg_lambda=4.0,
                        min_child_weight=8.0,
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        random_state=42,
                        n_jobs=-1,
                        tree_method="hist",
                    ),
                    feature_columns,
                ),
                Candidate(
                    "xgboost_conservative_no_market",
                    lambda: XGBClassifier(
                        n_estimators=80,
                        max_depth=2,
                        learning_rate=0.035,
                        subsample=0.75,
                        colsample_bytree=0.75,
                        reg_alpha=0.2,
                        reg_lambda=4.0,
                        min_child_weight=8.0,
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        random_state=42,
                        n_jobs=-1,
                        tree_method="hist",
                    ),
                    no_market_columns,
                ),
            ]
        )
    except Exception:
        pass

    return candidates


def compare_models(matches_path: Path, output_dir: Path, min_train: int, refit_frequency: int) -> pd.DataFrame:
    matches = load_matches(matches_path)
    featured = build_walk_forward_features(matches)
    scored = featured.iloc[min_train:].reset_index(drop=True)
    poisson = _poisson_probabilities(scored)
    bet_config = BetConfig(edge_threshold=0.05, fractional_kelly=0.2, max_stake_fraction=0.02)

    rows = [
        _score(
            "market_no_vig",
            scored,
            scored[["market_home_prob", "market_draw_prob", "market_away_prob"]].to_numpy(dtype=float),
            bet_config,
        ),
        _score(
            "elo_baseline",
            scored,
            scored[["elo_home_prob", "elo_draw_prob", "elo_away_prob"]].to_numpy(dtype=float),
            bet_config,
        ),
        _score("poisson_goals", scored, poisson, bet_config),
    ]
    rows.append(
        _score(
            "elo_poisson_blend",
            scored,
            _blend(scored[["elo_home_prob", "elo_draw_prob", "elo_away_prob"]].to_numpy(dtype=float), poisson),
            bet_config,
        )
    )

    feature_columns = FEATURE_COLUMNS
    no_market_columns = [name for name in FEATURE_COLUMNS if name not in MARKET_FEATURES]
    for candidate in _candidate_factories(feature_columns, no_market_columns):
        started = time.perf_counter()
        print(f"running {candidate.name}...")
        model_probabilities = _walk_forward_candidate(featured, candidate, min_train, refit_frequency)
        rows.append(_score(candidate.name, scored, model_probabilities, bet_config))
        rows.append(_score(f"{candidate.name}_poisson_blend", scored, _blend(model_probabilities, poisson), bet_config))
        print(f"finished {candidate.name} in {time.perf_counter() - started:.1f}s")

    results = pd.DataFrame(rows).sort_values(["log_loss", "brier", "accuracy"], ascending=[True, True, False])
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "model_comparison.csv", index=False)
    (output_dir / "model_comparison.json").write_text(
        json.dumps(results.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare football predictor models with walk-forward backtests")
    parser.add_argument("--matches", type=Path, default=Path("data/football_data_matches.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--min-train", type=int, default=500)
    parser.add_argument("--refit-frequency", type=int, default=500)
    args = parser.parse_args()
    results = compare_models(args.matches, args.output_dir, args.min_train, args.refit_frequency)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
