# Football Predictor

Walk-forward football score and betting-value predictor.

This first implementation turns the project prompt in [PROMPT.md](PROMPT.md) into a runnable scaffold:

- chronological feature generation with pre-match Elo, recent form, rest, market probabilities, and xG fallbacks
- strict walk-forward backtesting with no future rows used for model training
- model selection that tries XGBoost when installed and otherwise uses scikit-learn
- Poisson exact-score distributions and a lightweight match simulator module
- value-betting bankroll simulation using no-vig implied probabilities and fractional Kelly sizing
- static dashboard generated from a JSON backtest artifact

## Quick Start

```powershell
python -m football_predictor.cli demo --output web/data/report.json
python -m http.server 8000 --directory web
```

Open `http://localhost:8000`.

## Public Deployment

The app is ready for GitHub Pages. The workflow in `.github/workflows/deploy-pages.yml` runs tests, regenerates `web/data/report.json`, and publishes the static `web/` directory.

After the repository is pushed to GitHub:

1. Open the repository on GitHub.
2. Go to `Settings` -> `Pages`.
3. Set `Source` to `GitHub Actions`.
4. Push to `main` or `master`, or run `Deploy GitHub Pages` manually from the `Actions` tab.

For a repository named `football-predictor` under the `Loadstone6` account, the public URL should be:

```text
https://loadstone6.github.io/football-predictor/
```

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

The bundled `data/sample_matches.csv` is a small illustrative seed dataset for smoke testing the pipeline. It is not enough to make research-grade betting claims.

## Betting Note

The betting module implements value betting, not true arbitrage. It compares model probability with bookmaker implied probability after removing margin, then sizes the best positive-edge bet with fractional Kelly. Profit in a backtest is not evidence of real-world profitability unless the historical odds are timestamped, available at bet time, and adjusted for limits, commission, rejection, slippage, and overfitting.

## Tests

```powershell
python -m pytest
```
