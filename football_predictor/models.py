from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from football_predictor.features import FEATURE_COLUMNS


@dataclass
class OutcomeModel:
    estimator: object | None
    feature_columns: list[str]
    fill_values: pd.Series
    model_name: str

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame[self.feature_columns].apply(pd.to_numeric, errors="coerce").fillna(self.fill_values)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.estimator is None:
            return frame[["elo_home_prob", "elo_draw_prob", "elo_away_prob"]].to_numpy(dtype=float)

        raw = self.estimator.predict_proba(self._prepare(frame))
        full = np.zeros((len(frame), 3), dtype=float)
        classes = getattr(self.estimator, "classes_", np.array([0, 1, 2]))
        for pos, cls in enumerate(classes):
            full[:, int(cls)] = raw[:, pos]
        full = np.clip(full, 1e-6, 1.0)
        full = full / full.sum(axis=1, keepdims=True)
        return full

    def feature_importance(self) -> list[dict]:
        if self.estimator is None:
            weights = np.zeros(len(self.feature_columns))
            for name, value in {
                "elo_diff": 0.32,
                "elo_home_prob": 0.22,
                "recent_points_diff": 0.16,
                "goal_diff_form_diff": 0.14,
                "xg_form_diff": 0.10,
                "home_advantage_flag": 0.06,
            }.items():
                if name in self.feature_columns:
                    weights[self.feature_columns.index(name)] = value
        elif hasattr(self.estimator, "feature_importances_"):
            weights = np.asarray(self.estimator.feature_importances_, dtype=float)
        else:
            weights = np.ones(len(self.feature_columns), dtype=float)

        total = float(weights.sum())
        if total <= 0:
            weights = np.ones(len(self.feature_columns), dtype=float)
            total = float(weights.sum())
        items = [
            {"feature": name, "importance": float(value / total)}
            for name, value in zip(self.feature_columns, weights)
        ]
        return sorted(items, key=lambda item: item["importance"], reverse=True)


def train_outcome_model(
    training_frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
    random_state: int = 42,
) -> OutcomeModel:
    feature_columns = feature_columns or FEATURE_COLUMNS
    y = training_frame["actual_outcome_id"].astype(int)
    fill_values = training_frame[feature_columns].apply(pd.to_numeric, errors="coerce").median(numeric_only=True)
    fill_values = fill_values.fillna(0.0)

    if y.nunique() < 2:
        return OutcomeModel(None, feature_columns, fill_values, "elo_fallback")

    x = training_frame[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(fill_values)

    try:
        from xgboost import XGBClassifier

        estimator = XGBClassifier(
            n_estimators=160,
            max_depth=3,
            learning_rate=0.045,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=random_state,
        )
        model_name = "xgboost"
    except Exception:
        from sklearn.ensemble import RandomForestClassifier

        estimator = RandomForestClassifier(
            n_estimators=240,
            max_depth=5,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=random_state,
        )
        model_name = "random_forest"

    estimator.fit(x, y)
    return OutcomeModel(estimator, feature_columns, fill_values, model_name)
