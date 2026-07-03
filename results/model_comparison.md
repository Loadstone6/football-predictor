# Model Comparison

Dataset: `data/football_data_matches.csv`

Walk-forward setup:
- Total rows: 7,082
- Scored rows: 6,582
- Training warm-up: 500 matches
- Refit frequency: 500 matches
- Date range: 2022-01-09 to 2026-12-05
- Odds coverage: 100%
- xG coverage: 0%; goal-model features fall back to observed goals/form proxies

Best results by log loss:

| Rank | Model | Accuracy | Log Loss | Brier | Betting ROI |
|---:|---|---:|---:|---:|---:|
| 1 | Market no-vig | 0.540 | 0.970 | 0.577 | 0.000 |
| 2 | XGBoost conservative all features | 0.534 | 0.978 | 0.581 | -0.045 |
| 3 | XGBoost conservative + Poisson blend | 0.533 | 0.979 | 0.583 | -0.071 |
| 4 | XGBoost current + Poisson blend | 0.530 | 0.984 | 0.585 | -0.042 |
| 5 | XGBoost current all features | 0.531 | 0.988 | 0.588 | -0.019 |
| 6 | Logistic + Poisson blend | 0.518 | 0.991 | 0.590 | -0.064 |
| 7 | Random forest + Poisson blend | 0.519 | 0.991 | 0.590 | -0.105 |
| 8 | Elo + Poisson blend | 0.518 | 0.996 | 0.594 | -0.118 |

Conclusion:
The best predictive model is the bookmaker market no-vig baseline. Among trainable models, conservative XGBoost with all features is best. It still does not beat the market on log loss and does not produce profitable value-betting ROI in this backtest.

Main reasons:
- Bookmaker odds already encode injuries, lineup expectations, tactical context, public/private information, and expert price discovery that the current feature set does not include.
- The public dataset has no real xG, shot, lineup, injury, or player-strength features.
- Many model features are derived from goals, Elo, recent form, and market probabilities, so the signal is limited and partly redundant.
- XGBoost can overfit noisy football outcomes unless aggressively regularized.
- Optimizing prediction log loss is not the same as finding positive expected value after bookmaker margin.
