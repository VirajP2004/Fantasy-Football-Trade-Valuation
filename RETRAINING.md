# Retraining for a new season

Run this once a season, after the prior season completes.

1. **Pull fresh data.** Re-run the Sleeper ingestion (`src/ingestion/`) to
   pull the completed season's stats, rosters, and draft results.
2. **Rebuild the keeper ledger.** Re-run `src/keeper_ledger/build_ledger.py`
   so keeper costs reflect the new season's draft.
3. **Append a new walk-forward fold.** Re-run each position's model
   notebook (`06a`–`06d`) — the walk-forward CV generator automatically
   extends to include the new season as the latest test fold. Do not
   manually edit fold boundaries.
4. **Re-tune and re-lock artifacts.** Confirm the naive-baseline comparison
   still holds (each model should still beat naive on both MAE and
   Spearman on the newly appended fold). Save the updated `.json` model
   files to `data/models/`.
5. **Bump `MODEL_VERSION`** in `src/trade_engine/config.py` — this is the
   single source of truth for which model version produced a given trade
   log entry, so old and new predictions are never confused.
6. **Redeploy.** Push to `main`; Streamlit Community Cloud picks up the
   change automatically on the next app restart.
