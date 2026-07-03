# Football Predictor

Professional football prediction and backtesting console.

This implementation turns the product prompt in [PROMPT.md](PROMPT.md) into a hosted static analytics app backed by a local Python modelling pipeline:

- chronological feature generation with pre-match Elo, recent form, rest, market probabilities, and xG fallbacks
- strict walk-forward backtesting with no future rows used for model training
- model comparison across market baseline, Elo, Poisson, tree models, and XGBoost when installed
- Dixon-Coles adjusted scoreline matrices, expected goals, over/under, BTTS, and clean-sheet markets
- fixture browser with filters, confidence sorting, model-vs-market edge sorting, and search
- match detail page with probability bars, fair odds, score heatmap, xG, explainability, risk factors, and similar-match history
- team and league analytics pages with power rankings, form, standings, trend charts, and league performance
- backtest dashboard with accuracy, log loss, Brier score, calibration, confidence buckets, confusion matrix, bankroll simulation, and archive rows
- model lab with scenario simulator, odds lab, model registry, leakage audit, data quality checks, and watchlists

## Quick Start

```powershell
python scripts\build_football_data_dataset.py --output data\football_data_matches.csv
python -m football_predictor.cli backtest --matches data\football_data_matches.csv --output web\data\report.json --min-train 500 --refit-frequency 200 --edge-threshold 0.05 --fractional-kelly 0.20 --max-stake-fraction 0.02
python -m http.server 8000 --directory web
```

Open `http://localhost:8000`.

For a tiny smoke-test report, run:

```powershell
python -m football_predictor.cli demo --output web\data\report.json
```

## Public Deployment

The public dashboard is published at:

<https://loadstone6.github.io/>

The source project is in `Loadstone6/football-predictor`. The live static site is served from the root GitHub Pages repository `Loadstone6/Loadstone6.github.io`, using the generated contents of the local `web/` directory.

To publish a fresh build, regenerate `web/data/report.json`, copy the contents of `web/` into the Pages repository root, then commit and push that repository.

## Use Your Own Data

Use a CSV with these preferred columns:

```text
date,competition,season,home_team,away_team,home_goals,away_goals,neutral_site,home_odds,draw_odds,away_odds,home_xg,away_xg
```

Then run:

```powershell
python -m football_predictor.cli backtest --matches path\to\matches.csv --output web\data\report.json --min-train 80 --refit-frequency 10
```

To isolate the 2026 World Cup once you have a timestamped results/odds file:

```powershell
python -m football_predictor.cli backtest --matches path\to\international_matches.csv --competition-filter "FIFA World Cup 2026" --output web\data\report.json --min-train 500 --refit-frequency 1
```

For a true out-of-sample World Cup test, append each match only after it has finished, keep pre-kickoff odds/features timestamped, and rerun the backtest. The engine trains on rows strictly before each predicted match.

## Public Data Starting Points

- [Football-Data.co.uk](https://www.football-data.co.uk/) publishes historical club results and bookmaker odds in CSV format.
- [Football-Data results/odds index](https://www.football-data.co.uk/data.php) lists available country and league files.
- [Club Elo](https://clubelo.com/) provides club Elo ratings and historical strength estimates.
- [World Football Elo Ratings](https://eloratings.net/) publishes national-team Elo ratings.

The bundled `data/sample_matches.csv` is a small illustrative seed dataset for smoke testing the pipeline. The richer `data/football_data_matches.csv` dataset is built from Football-Data.co.uk match and odds CSVs.

## Betting Note

The betting module implements value betting, not true arbitrage. It compares model probability with bookmaker implied probability after removing margin, then sizes the best positive-edge bet with fractional Kelly. Profit in a backtest is not evidence of real-world profitability unless the historical odds are timestamped, available at bet time, and adjusted for limits, commission, rejection, slippage, and overfitting.

The current public dataset has strong results and odds coverage but no verified injuries, lineups, referees, weather, or real xG feed. Those are represented as data-quality warnings and adapter targets rather than silently inferred facts.

## Tests

```powershell
python -m pytest
node --check web\app.js
```

## Model Comparison

Run walk-forward comparisons across Elo, Poisson, market no-vig, logistic regression, tree ensembles, and XGBoost when installed:

```powershell
python -m pip install xgboost
python scripts\compare_models.py --matches data\football_data_matches.csv --output-dir results --min-train 500 --refit-frequency 500
```

Latest comparison summary: [results/model_comparison.md](results/model_comparison.md).
