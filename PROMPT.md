Build a production-quality football match score predictor with a polished web UI.

Goal:
Create a model that predicts match outcomes, exact scores, win/draw/loss probabilities, expected goals, and betting value. The system should use historical football data, Elo-style ratings, team/player features, xG-style metrics, simulated match physics, and machine learning models such as XGBoost, LightGBM, CatBoost, Poisson models, Bayesian models, or ensembles.

Core requirements:

1. Data collection
Use as much reliable historical football data as possible, including:
- International and club match results
- FIFA World Cup, Euros, Copa America, AFCON, Asian Cup, qualifiers, Nations League, friendlies
- Team Elo ratings over time
- Goals scored/conceded
- xG and xGA where available
- Shots, possession, corners, cards, fouls, set pieces
- Squad strength, player ratings, injuries, suspensions, lineups
- Rest days, travel distance, venue, home advantage, weather, altitude
- Manager tenure and recent form
- Betting odds and historical payout data where available

Clearly document all data sources and handle missing data carefully.

2. Modelling
Build several predictive approaches and compare them:
- Baseline Elo model
- Poisson or Dixon-Coles goal model
- XGBoost/LightGBM/CatBoost classifier/regressor
- Bayesian hierarchical model if useful
- Monte Carlo match simulator using attack/defence strength, tempo, shot quality, fatigue, red-card probability, substitutions, and randomness
- Ensemble model combining the best approaches

The model should output:
- Probability of home/team A win, draw, away/team B win
- Probability distribution over exact scores
- Expected goals for each team
- Most likely scorelines
- Confidence intervals
- Model uncertainty

3. Explainability
Show which factors explain the prediction most.
Use SHAP, permutation importance, feature importance, partial dependence, or similar techniques.

For each match, display:
- Top positive factors for Team A
- Top positive factors for Team B
- Factors increasing draw probability
- Global model feature importance
- Local explanation for the specific prediction

4. Historical walk-forward backtesting
Historical walk-forward backtesting is feasible, but only if the data is timestamped cleanly.

The correct setup:
- Sort matches chronologically.
- For each match date, train only on data available before kickoff.
- Generate features as they would have existed then: Elo before match, recent form before match, squad information known pre-match, and odds available before kickoff.
- Predict the match.
- Record the actual result, update ratings/data, and move forward.
- Evaluate log loss, Brier score, calibration, accuracy, ROI, drawdown, and bankroll growth.

The main feasibility issue is not modelling. It is data integrity. Avoid leakage from final Elo ratings, post-match xG, final tournament squad strength, closing odds unavailable at bet time, or injury/lineup data that was only published later.

For football outcomes, this is feasible with historical results, Elo, odds, and basic team features. It becomes harder with richer features like player availability, injuries, weather, tactical style, and xG because those datasets are patchier, often paid, and harder to reconstruct as-of-date.

Betting backtests are feasible but fragile. Historical odds must include timestamps, bookmaker margin should be removed, staking rules must be defined before testing, and practical frictions such as limits, slippage, commission, and bet rejection should be considered. Otherwise it is easy to produce a fake profitable strategy.

5. Out-of-sample testing
Perform strict out-of-sample testing with no data leakage.
Use time-based splits only.

Specifically test against the 2026 FIFA World Cup:
- Train only on data available before each match date.
- Predict each 2026 World Cup match before seeing the result.
- Track accuracy, log loss, Brier score, calibration, ROI, and exact-score performance.
- Compare against bookmaker implied probabilities where odds are available.
- Show where the model beat or underperformed the market.

If full 2026 results are not available yet, design the backtest so it can update automatically as results arrive.

6. Betting value and bankroll simulation
This is not true arbitrage unless risk-free odds exist across bookmakers. Treat this as value betting based on model edge.

Collect historical bookmaker odds and payouts if possible.
Convert odds into implied probabilities after removing bookmaker margin.

For each match:
- Compare bookmaker implied probability vs model probability.
- Identify value bets where model probability exceeds market probability by a configurable edge threshold.
- Size bets using fractional Kelly criterion, flat staking, and conservative staking.
- Simulate bankroll growth over historical matches.
- Report ROI, max drawdown, volatility, Sharpe-like metrics, hit rate, expected value, and calibration.
- Include transaction costs, limits, slippage, and bookmaker margin.
- Warn clearly about overfitting and gambling risk.

Example:
If England vs Spain market odds imply England 40% and Spain 60%, but the model predicts England 70% and Spain 30%, calculate whether England is a positive-EV bet and size the stake accordingly.

7. UI requirements
Create a clean, modern, responsive web UI.

The UI should include:
- Match predictor page
- Team comparison dashboard
- Score probability heatmap
- Win/draw/loss probability bars
- Feature explanation panel
- Elo/rating history chart
- Betting value dashboard
- Backtesting results page
- Bankroll simulation chart
- Calibration plots
- Model comparison table
- Data source/status page

The UI should feel like a professional sports analytics tool, not a toy demo.

8. Technical expectations
Use a clear, maintainable architecture.

Recommended stack:
- Python for modelling
- pandas/polars, scikit-learn, xgboost/lightgbm/catboost where available
- FastAPI, Flask, Streamlit, or static UI generated from model artifacts
- SQLite/Postgres for persistent data storage when the data volume grows
- Plotly/Recharts/ECharts/SVG for visualizations
- SHAP for explanations when available

Include:
- Reproducible data pipeline
- Model training scripts
- Evaluation scripts
- Backtesting engine
- Betting simulation module
- Tests for key calculations
- Clear README with setup instructions

9. Guardrails
Avoid data leakage.
Do not train on future information.
Separate training, validation, and test data chronologically.
Report uncertainty honestly.
Do not claim guaranteed betting profits.
Distinguish between:
- Prediction accuracy
- Calibration
- Positive expected value
- Real-world betting profitability

Deliverables:
- Working app
- Source code
- Data pipeline
- Trained model artifacts
- Backtest report
- Betting ROI simulation
- UI screenshots
- Explanation of model assumptions, limitations, and next improvements
