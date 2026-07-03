from __future__ import annotations

import argparse
from pathlib import Path

from football_predictor.backtest import BacktestConfig, walk_forward_backtest
from football_predictor.betting import BetConfig
from football_predictor.data import load_matches
from football_predictor.report import write_report


ROOT = Path(__file__).resolve().parents[1]


def _run(matches_path: Path, args) -> Path:
    matches = load_matches(matches_path)
    if args.competition_filter:
        matches = matches[matches["competition"].str.contains(args.competition_filter, case=False, na=False)]
        if matches.empty:
            raise SystemExit(f"No matches found for competition filter: {args.competition_filter}")

    config = BacktestConfig(
        min_train_matches=args.min_train,
        refit_frequency=args.refit_frequency,
        bet_config=BetConfig(
            starting_bankroll=args.bankroll,
            edge_threshold=args.edge_threshold,
            fractional_kelly=args.fractional_kelly,
            max_stake_fraction=args.max_stake_fraction,
        ),
    )
    backtest = walk_forward_backtest(matches, config)
    output = write_report(backtest, args.output, source_name=str(matches_path))
    metrics = backtest["metrics"]
    bet_metrics = backtest["bet_metrics"]
    print(f"Wrote {output}")
    print(
        "Matches={matches} Accuracy={accuracy:.3f} LogLoss={log_loss:.3f} "
        "Brier={brier:.3f} Bets={bets} ROI={roi:.3f}".format(
            **metrics,
            bets=bet_metrics["bets"],
            roi=bet_metrics["roi"],
        )
    )
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Football match predictor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output", type=Path, default=ROOT / "web" / "data" / "report.json")
    common.add_argument("--min-train", type=int, default=18)
    common.add_argument("--refit-frequency", type=int, default=4)
    common.add_argument("--competition-filter", default=None)
    common.add_argument("--bankroll", type=float, default=1000.0)
    common.add_argument("--edge-threshold", type=float, default=0.04)
    common.add_argument("--fractional-kelly", type=float, default=0.25)
    common.add_argument("--max-stake-fraction", type=float, default=0.03)

    demo = subparsers.add_parser("demo", parents=[common], help="Run against bundled sample data")
    demo.set_defaults(matches=ROOT / "data" / "sample_matches.csv")

    backtest = subparsers.add_parser("backtest", parents=[common], help="Run against a match CSV")
    backtest.add_argument("--matches", type=Path, required=True)

    args = parser.parse_args(argv)
    _run(Path(args.matches), args)


if __name__ == "__main__":
    main()
