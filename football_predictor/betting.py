from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable

import numpy as np
import pandas as pd


OUTCOME_NAMES = ("home", "draw", "away")


@dataclass(frozen=True)
class BetConfig:
    starting_bankroll: float = 1000.0
    edge_threshold: float = 0.04
    fractional_kelly: float = 0.25
    max_stake_fraction: float = 0.03


def normalized_implied_probabilities(decimal_odds: Iterable[float | None]) -> tuple[float | None, ...]:
    implied = []
    for odd in decimal_odds:
        if odd is None or pd.isna(odd) or float(odd) <= 1.0:
            implied.append(np.nan)
        else:
            implied.append(1.0 / float(odd))

    if any(pd.isna(v) for v in implied):
        return tuple(None for _ in implied)

    margin_sum = float(sum(implied))
    if margin_sum <= 0:
        return tuple(None for _ in implied)
    return tuple(float(v / margin_sum) for v in implied)


def expected_value_per_unit(model_probability: float, decimal_odds: float) -> float:
    return (model_probability * decimal_odds) - 1.0


def kelly_fraction(model_probability: float, decimal_odds: float) -> float:
    if decimal_odds <= 1.0:
        return 0.0
    b = decimal_odds - 1.0
    q = 1.0 - model_probability
    fraction = ((b * model_probability) - q) / b
    return max(0.0, fraction)


def choose_value_bet(row: pd.Series, config: BetConfig) -> dict | None:
    odds = [row.get("home_odds"), row.get("draw_odds"), row.get("away_odds")]
    market = normalized_implied_probabilities(odds)
    if any(v is None for v in market):
        return None

    candidates = []
    for idx, name in enumerate(OUTCOME_NAMES):
        model_probability = float(row[f"p_{name}"])
        market_probability = float(market[idx])
        odd = float(odds[idx])
        edge = model_probability - market_probability
        ev = expected_value_per_unit(model_probability, odd)
        if edge >= config.edge_threshold and ev > 0:
            candidates.append(
                {
                    "outcome_id": idx,
                    "outcome": name,
                    "model_probability": model_probability,
                    "market_probability": market_probability,
                    "odds": odd,
                    "edge": edge,
                    "ev_per_unit": ev,
                    "kelly": kelly_fraction(model_probability, odd),
                }
            )

    if not candidates:
        return None
    return max(candidates, key=lambda item: (item["ev_per_unit"], item["edge"]))


def simulate_value_betting(predictions: pd.DataFrame, config: BetConfig | None = None) -> tuple[pd.DataFrame, dict]:
    config = config or BetConfig()
    bankroll = float(config.starting_bankroll)
    peak = bankroll
    rows = []

    for _, row in predictions.iterrows():
        bet = choose_value_bet(row, config)
        before = bankroll
        stake = 0.0
        profit = 0.0
        placed = False

        if bet is not None:
            stake_fraction = min(config.max_stake_fraction, config.fractional_kelly * bet["kelly"])
            stake = max(0.0, before * stake_fraction)
            if stake > 0:
                placed = True
                won = int(row["actual_outcome_id"]) == int(bet["outcome_id"])
                profit = stake * (bet["odds"] - 1.0) if won else -stake
                bankroll = before + profit
                bet["won"] = won

        peak = max(peak, bankroll)
        drawdown = (bankroll - peak) / peak if peak > 0 else 0.0
        rows.append(
            {
                "date": row["date"],
                "match": f"{row['home_team']} vs {row['away_team']}",
                "bankroll_before": before,
                "bankroll_after": bankroll,
                "stake": stake,
                "profit": profit,
                "placed": placed,
                "drawdown": drawdown,
                **(bet or {}),
            }
        )

    history = pd.DataFrame(rows)
    placed = history[history["placed"] == True] if not history.empty else history
    total_staked = float(placed["stake"].sum()) if not placed.empty else 0.0
    total_profit = float(placed["profit"].sum()) if not placed.empty else 0.0
    returns = placed["profit"] / placed["stake"].replace(0, np.nan) if not placed.empty else pd.Series(dtype=float)
    volatility = float(returns.std(ddof=0)) if len(returns) > 1 else 0.0
    mean_return = float(returns.mean()) if len(returns) > 0 else 0.0
    sharpe_like = float(mean_return / volatility * sqrt(len(returns))) if volatility > 0 else 0.0
    metrics = {
        "starting_bankroll": config.starting_bankroll,
        "ending_bankroll": bankroll,
        "bets": int(len(placed)),
        "total_staked": total_staked,
        "total_profit": total_profit,
        "roi": total_profit / total_staked if total_staked else 0.0,
        "hit_rate": float(placed["won"].mean()) if not placed.empty and "won" in placed else 0.0,
        "max_drawdown": float(history["drawdown"].min()) if not history.empty else 0.0,
        "sharpe_like": sharpe_like,
    }
    return history, metrics
