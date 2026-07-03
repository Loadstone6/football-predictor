from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd


def _empty_profile() -> dict:
    return {
        "team": "",
        "matches": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "points": 0.0,
        "goals_for": 0.0,
        "goals_against": 0.0,
        "xg_for": 0.0,
        "xg_against": 0.0,
        "elo": 1500.0,
        "attack_form": 1.25,
        "defense_form": 1.25,
        "xg_form": 1.25,
        "recent_points": 1.2,
        "last_date": None,
        "form": deque(maxlen=6),
    }


def _apply_result(profile: dict, goals_for: float, goals_against: float) -> None:
    if goals_for > goals_against:
        profile["wins"] += 1
        profile["points"] += 3
        profile["form"].append("W")
    elif goals_for < goals_against:
        profile["losses"] += 1
        profile["form"].append("L")
    else:
        profile["draws"] += 1
        profile["points"] += 1
        profile["form"].append("D")


def build_team_profiles(featured: pd.DataFrame) -> list[dict]:
    profiles = defaultdict(_empty_profile)
    if featured.empty:
        return []

    for _, row in featured.sort_values("date").iterrows():
        for side, opponent_side in (("home", "away"), ("away", "home")):
            team = row[f"{side}_team"]
            profile = profiles[team]
            profile["team"] = team

            goals_for = float(row[f"{side}_goals"])
            goals_against = float(row[f"{opponent_side}_goals"])
            xg_for = row.get(f"{side}_xg")
            xg_against = row.get(f"{opponent_side}_xg")
            xg_for = goals_for if pd.isna(xg_for) else float(xg_for)
            xg_against = goals_against if pd.isna(xg_against) else float(xg_against)

            profile["matches"] += 1
            profile["goals_for"] += goals_for
            profile["goals_against"] += goals_against
            profile["xg_for"] += xg_for
            profile["xg_against"] += xg_against
            profile["elo"] = float(row.get(f"{side}_elo_post", row.get(f"{side}_elo", 1500.0)))
            profile["attack_form"] = float(row.get(f"{side}_attack_form", profile["attack_form"]))
            profile["defense_form"] = float(row.get(f"{side}_defense_form", profile["defense_form"]))
            profile["xg_form"] = float(row.get(f"{side}_xg_form", profile["xg_form"]))
            profile["recent_points"] = float(row.get(f"{side}_recent_points", profile["recent_points"]))
            profile["last_date"] = row["date"]
            _apply_result(profile, goals_for, goals_against)

    max_date = featured["date"].max()
    rows = []
    for profile in profiles.values():
        matches = max(1, int(profile["matches"]))
        last_date = profile["last_date"]
        rest_days = int((max_date - last_date).days) if last_date is not None else None
        rows.append(
            {
                "team": profile["team"],
                "matches": int(profile["matches"]),
                "wins": int(profile["wins"]),
                "draws": int(profile["draws"]),
                "losses": int(profile["losses"]),
                "points_per_match": float(profile["points"] / matches),
                "win_rate": float(profile["wins"] / matches),
                "goals_for_per_match": float(profile["goals_for"] / matches),
                "goals_against_per_match": float(profile["goals_against"] / matches),
                "goal_diff_per_match": float((profile["goals_for"] - profile["goals_against"]) / matches),
                "xg_for_per_match": float(profile["xg_for"] / matches),
                "xg_against_per_match": float(profile["xg_against"] / matches),
                "xg_diff_per_match": float((profile["xg_for"] - profile["xg_against"]) / matches),
                "elo": float(profile["elo"]),
                "attack_form": float(profile["attack_form"]),
                "defense_form": float(profile["defense_form"]),
                "xg_form": float(profile["xg_form"]),
                "recent_points": float(profile["recent_points"]),
                "rest_days": rest_days,
                "form": "".join(profile["form"]),
                "last_date": last_date,
            }
        )

    rows.sort(key=lambda item: (item["elo"], item["xg_diff_per_match"], item["points_per_match"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["power_score"] = float(
            np.clip(
                50
                + (row["elo"] - 1500.0) / 8.0
                + 8.0 * row["xg_diff_per_match"]
                + 5.0 * (row["points_per_match"] - 1.2),
                0,
                100,
            )
        )
    return rows
