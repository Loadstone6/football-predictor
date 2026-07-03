from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve


BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Football-Data.co.uk league CSVs")
    parser.add_argument("--seasons", nargs="+", default=["2324", "2425", "2526"])
    parser.add_argument("--leagues", nargs="+", default=["E0", "SP1", "D1", "I1", "F1"])
    parser.add_argument("--out", type=Path, default=Path("data/raw/football_data"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for season in args.seasons:
        for league in args.leagues:
            url = BASE_URL.format(season=season, league=league)
            target = args.out / f"{season}_{league}.csv"
            try:
                urlretrieve(url, target)
                print(f"downloaded {url} -> {target}")
            except Exception as exc:
                print(f"skipped {url}: {exc}")


if __name__ == "__main__":
    main()
