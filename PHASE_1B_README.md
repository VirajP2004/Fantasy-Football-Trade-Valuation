# Phase 1B — Keeper Ledger & Draft Capital Curve

League configured: `1371612837679464448` (see `config/keeper_rules.yaml`)

## How to (re)run it

```bash
pip install pandas pyyaml requests
mkdir -p data/raw data/processed
python -m src.keeper_ledger.build_ledger
```

This walks the league's `previous_league_id` chain back through every prior
season, pulls each season's draft picks, and writes:
- `data/raw/draft_history.parquet` — flattened raw picks, one row per
  (season, owner_id, player_id)
- `data/processed/keeper_ledger.parquet` — the escalation ledger, with an
  `expected_round_formula`/`formula_mismatch` QA flag pair (see below)

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
3. **This ledger does not itself project *next* season's keeper cost** —
   it only covers rows that already went through an actual draft pick, by
   design (see `build_keeper_ledger`'s docstring). The single source of
   truth for "what would it cost to keep this player next year" is
   `scripts/project_roster_keeper_costs.py`, which reads this ledger's
   `keeps_since_rule_start`/`is_keeper` columns and runs the full
   three-tier anchor resolution on top of them.
