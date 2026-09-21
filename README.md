# Fantasy Football Trade Valuation

A two-stage machine learning system that scores whether a proposed trade in a
10-team keeper fantasy football league is fair to both sides, with a full
explanation of why.

**Live app:** https://fantasy-football-trade-valuation-gtown.streamlit.app/

## The problem

Keeper leagues complicate trade evaluation beyond raw player talent: a
player's value depends on what it costs to *keep* them next season (a
function of the draft round they were acquired in), not just their
projected production. A trade that looks lopsided on talent alone can be
fair once keeper cost is priced in, and vice versa. This project builds a
model-backed second opinion for that judgment call.

## Architecture

**Stage 1 — Keeper Value Score (KVS).** A separate XGBoost regressor per
position (QB, RB, WR, TE) predicts a player's next-season Value Over
Replacement Player (VORP) from their current season's performance,
opportunity, efficiency, and positional-scarcity metrics. Kicker and Team
Defense are explicitly excluded — they are never traded in this league, and
DEF's own validation numbers underperformed a naive baseline, so no model
was shipped for either position rather than presenting an untrustworthy one.

**Stage 2 — Trade Fairness Engine.** For each player in a trade, KVS is
combined with their keeper cost (derived from a real draft-capital curve
built from this league's own draft history) and an adjustment for how much
the *receiving* team specifically needs that position. The two sides' net
values are compared and converted into a fairness percentage, with a
dedicated fallback formula for the case where both sides are net-negative
(a naive percentage-of-total calculation badly distorts that case).

## Methodology

- **Walk-forward validation only.** No K-fold, no random splits, no
  shuffling. Each model is validated on chronologically rolling windows —
  training on all seasons up to a cutoff, testing on the next season, then
  advancing — to prevent temporal leakage. 8 folds per position, test
  seasons 2017–2024.
- **RFECV feature selection**, wrapped in the same walk-forward CV
  generator, optimized against XGBoost's gain metric. Feature sets are
  locked per position (4–9 features each) rather than re-derived on every
  run, to prevent silent drift.
- **Hyperparameter tuning** via nested Optuna search (TPE sampler, 40
  trials per fold), scored on an inner validation split carved from each
  fold's own training window — never on the fold's held-out test season.
- **SHAP (TreeExplainer)** for every prediction, surfaced in the app as the
  top drivers behind each player's number, not just an aggregate score.

## Key findings

- All four models beat a naive "assume next season repeats this season"
  baseline, by 2.7%–6.6% MAE.
- A player's most extreme recent value swing (`vorp_delta_yoy`) is a real,
  useful feature but degrades in its own extreme tail (confirmed at all
  four positions) — flagged per-prediction as `low_confidence_extreme_delta`
  rather than silently trusted.
- A player's top-decile standing at their position is consistently the
  *hardest* thing to predict continuation of, at every position — elite
  seasons are harder to project forward than replacement-level ones.

## Known limitations

- **Season-level, not real-time.** Predictions are built from each
  player's most recently completed full season. The model has no knowledge
  of the current season's games, injuries, or performance in progress.
- **K/DEF unsupported.** Selecting a kicker or defense in a trade returns a
  clear error rather than a fabricated number.
- **No backtest yet.** A true walk-forward backtest requires trades that
  happened outside a model's training window. This league currently has
  only 2 complete seasons, and every historical trade falls inside that
  window — running a "backtest" against them would just be re-grading the
  model on data it already saw. Revisit after the 2026 season closes.
- **Rookies and injury-shortened seasons** are flagged (`no_delta_history`,
  `low_confidence_extreme_delta`) rather than silently treated with full
  confidence.

## Tech stack

Python, pandas, XGBoost, scikit-learn, SHAP, Optuna, Streamlit. Trade
history logged to SQLite.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Repo structure

```
config/          league rules, locked feature sets
data/            raw pulls, processed labels, trained model artifacts
notebooks/       exploratory + modeling notebooks (Phases 2-6)
src/             production modules: Sleeper ingestion, league scoring, keeper ledger, trade engine, explainability
tests/           pytest suite (106 tests)
app.py           Streamlit trade-evaluation app
roadmap.md       full build log, phase by phase, including decisions and their reasoning
RETRAINING.md    runbook for retraining ahead of a new season
```
