from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football_predictor.data import standardize_matches


BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

LEAGUE_NAMES = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
}

SEASON_LABELS = {
    "2223": "2022-23",
    "2324": "2023-24",
    "2425": "2024-25",
    "2526": "2025-26",
}


def _download_csv(season: str, league: str, raw_dir: Path, refresh: bool) -> Path | None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / f"{season}_{league}.csv"
    if target.exists() and not refresh:
        return target

    url = BASE_URL.format(season=season, league=league)
    try:
        urlretrieve(url, target)
        return target
    except Exception as exc:
        print(f"skipped {url}: {exc}")
        return None


def build_dataset(seasons: list[str], leagues: list[str], raw_dir: Path, output: Path, refresh: bool) -> Path:
    frames = []
    for season in seasons:
        for league in leagues:
            csv_path = _download_csv(season, league, raw_dir, refresh)
            if csv_path is None:
                continue

            raw = pd.read_csv(csv_path)
            raw["season"] = SEASON_LABELS.get(season, season)
            raw["competition"] = f"{LEAGUE_NAMES.get(league, league)} {SEASON_LABELS.get(season, season)}"
            standardized = standardize_matches(raw)
            standardized["source_file"] = csv_path.name
            frames.append(standardized)

    if not frames:
        raise SystemExit("No Football-Data CSVs were downloaded or parsed.")

    combined = pd.concat(frames, ignore_index=True).sort_values(["date", "competition", "home_team"])
    combined = combined.drop_duplicates(subset=["date", "competition", "home_team", "away_team"])
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    print(f"wrote {output} rows={len(combined)} teams={combined[['home_team', 'away_team']].stack().nunique()}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a combined Football-Data.co.uk match dataset")
    parser.add_argument("--seasons", nargs="+", default=["2223", "2324", "2425", "2526"])
    parser.add_argument("--leagues", nargs="+", default=["E0", "SP1", "D1", "I1", "F1"])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/football_data"))
    parser.add_argument("--output", type=Path, default=Path("data/football_data_matches.csv"))
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    build_dataset(args.seasons, args.leagues, args.raw_dir, args.output, args.refresh)


if __name__ == "__main__":
    main()
