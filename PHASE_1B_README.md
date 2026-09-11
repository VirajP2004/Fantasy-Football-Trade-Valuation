# Phase 1B — Keeper Ledger & Draft Capital Curve

League configured: `1371612837679464448` (see `config/keeper_rules.yaml`)

## Status

- ✅ Escalation logic implemented and unit-tested (6/6 passing) against synthetic data.
- ⚠️ **Live pull not yet run.** This sandbox has no network access to `api.sleeper.app`
  (confirmed: 403 from the egress proxy on this environment). You need to run the
  pull step somewhere with internet access.

## To run it for real

```bash
pip install pandas pyyaml requests
mkdir -p data/raw data/processed
python -m src.keeper_ledger.build_ledger
```

This walks the league's `previous_league_id` chain back through every prior
season, pulls each season's draft picks, and writes:
- `data/raw/draft_history.parquet` — flattened raw picks, one row per
  (season, owner_id, player_id)
- `data/processed/keeper_ledger.parquet` — the escalation ledger with
  `next_season_keeper_round` and a `formula_mismatch` QA flag

## Before you trust the output

1. **Check `formula_mismatch` rows manually.** These are cases where the
   round Sleeper recorded doesn't match what the escalation formula
   predicts — usually a missed `is_keeper` flag (see below) or a genuine
   league rule exception (trade-deadline keeper swap, commissioner override, etc.).
2. **Sleeper's `is_keeper` field is often null.** If your commissioner never
   set it when starting drafts, most/all picks will read as "fresh," which
   silently breaks the escalation chain. Check the mismatch count printed
   by the script — a high count is a strong signal the flag isn't populated.
   Fix it via `data/raw/manual_keeper_overrides.csv` (columns: `season,
   owner_id, player_id, is_keeper`), keyed by the same `owner_id` you'll
   see in the ledger output (pull `/league/<id>/users` to map `owner_id` ->
   display name for building that file by hand).
3. **`draft_capital_curve.py` won't run yet** — it needs `vorp_labels.parquet`,
   which is a Phase 2 deliverable that doesn't exist in this repo yet. The
   function is ready; nothing else blocks it.

## Next step

Either:
- Connect Claude for Chrome in this session and ask me to pull it live, or
- Run `python -m src.keeper_ledger.build_ledger` yourself and drop the two
  output parquet files back here so I can sanity-check the escalation
  output against your actual league before we move to Phase 2.
