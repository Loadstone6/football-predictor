from __future__ import annotations

from pathlib import Path

import pandas as pd


FOOTBALL_DATA_ODDS_PRIORITY = (
    ("home_odds", "draw_odds", "away_odds"),
    ("AvgH", "AvgD", "AvgA"),
    ("B365H", "B365D", "B365A"),
    ("PSH", "PSD", "PSA"),
    ("WHH", "WHD", "WHA"),
    ("IWH", "IWD", "IWA"),
)


def _first_column(df: pd.DataFrame, names: tuple[str, ...], default=None):
    for name in names:
        if name in df.columns:
            return df[name]
    return default


def _coerce_bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes", "y", "neutral", "n"}
    )


def _select_odds(df: pd.DataFrame) -> tuple[pd.Series | None, pd.Series | None, pd.Series | None]:
    for home, draw, away in FOOTBALL_DATA_ODDS_PRIORITY:
        if home in df.columns and draw in df.columns and away in df.columns:
            return df[home], df[draw], df[away]
    return None, None, None


def _parse_dates(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, format="mixed", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(values.loc[missing], dayfirst=True, errors="coerce")
    return parsed


def standardize_matches(df: pd.DataFrame, include_unplayed: bool = False) -> pd.DataFrame:
    out = pd.DataFrame()
    out["date"] = _first_column(df, ("date", "Date", "match_date", "MatchDate"))
    out["competition"] = _first_column(df, ("competition", "Competition", "Div", "league"), "Unknown")
    out["season"] = _first_column(df, ("season", "Season"), None)
    out["home_team"] = _first_column(df, ("home_team", "HomeTeam", "home", "Home"))
    out["away_team"] = _first_column(df, ("away_team", "AwayTeam", "away", "Away"))
    out["home_goals"] = _first_column(df, ("home_goals", "FTHG", "HG", "home_score"))
    out["away_goals"] = _first_column(df, ("away_goals", "FTAG", "AG", "away_score"))

    neutral = _first_column(df, ("neutral_site", "neutral", "Neutral"), False)
    out["neutral_site"] = _coerce_bool(pd.Series(neutral, index=df.index))

    home_odds, draw_odds, away_odds = _select_odds(df)
    out["home_odds"] = home_odds
    out["draw_odds"] = draw_odds
    out["away_odds"] = away_odds

    out["home_xg"] = _first_column(df, ("home_xg", "HomeXG", "xG_home", "HxG"), None)
    out["away_xg"] = _first_column(df, ("away_xg", "AwayXG", "xG_away", "AxG"), None)

    out["date"] = _parse_dates(out["date"])
    for column in ("home_goals", "away_goals", "home_odds", "draw_odds", "away_odds", "home_xg", "away_xg"):
        out[column] = pd.to_numeric(out[column], errors="coerce")

    if out["season"].isna().all():
        out["season"] = out["date"].dt.year.astype("Int64").astype(str)
    else:
        out["season"] = out["season"].astype(str)

    out = out.dropna(subset=["date", "home_team", "away_team"])
    if not include_unplayed:
        out = out.dropna(subset=["home_goals", "away_goals"])

    out["result"] = "D"
    out.loc[out["home_goals"] > out["away_goals"], "result"] = "H"
    out.loc[out["home_goals"] < out["away_goals"], "result"] = "A"
    out["actual_outcome_id"] = out["result"].map({"H": 0, "D": 1, "A": 2}).astype(int)
    out = out.sort_values(["date", "competition", "home_team", "away_team"]).reset_index(drop=True)
    return out


def load_matches(path: str | Path, include_unplayed: bool = False) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    return standardize_matches(df, include_unplayed=include_unplayed)


def load_many(paths: list[str | Path], include_unplayed: bool = False) -> pd.DataFrame:
    frames = [load_matches(path, include_unplayed=include_unplayed) for path in paths]
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
