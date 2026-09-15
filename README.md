# Keeper League Trade Fairness Engine

*Placeholder README — to be expanded into full setup/usage docs once the project is closer to done. See `roadmap.md` for current phase status and progress.*

## A note on data freshness

This project intentionally uses live, continuously-updated data sources (`nflreadpy`) rather than a frozen snapshot — on purpose, not as an oversight: the end goal is a tool meant to inform real, current trade decisions in an active league, so it needs to reflect the league's actual, present-day reality rather than a benchmark frozen at some arbitrary past date.

One consequence worth knowing before trusting any specific number out of this repo:

- **Reported metrics reflect the data available at the time each notebook last ran, not a fixed benchmark.** Upstream corrections to `nflreadpy`'s data (player records, team stats, injury reports — even for already-completed past seasons) can and do shift results between runs of the exact same code — sometimes only in the third decimal place, occasionally enough to change which features get selected.
- Every notebook that pulls `nflreadpy` data prints a `Results as of [run date] ..., pulling live nflreadpy data` line near the top, computed at execution time — check that timestamp before treating any number in that notebook as current, especially if it's been a while since the notebook was last run.
- If two notebooks (or a notebook and this README/`roadmap.md`) disagree on a specific decimal, the more recently-run one reflects more current data — neither is "wrong," they're just snapshots from different points in time.

**A concrete example, not a hypothetical**: re-running `05_feature_selection.ipynb` on 2026-09-15 picked a different feature set for RB than the run it was originally built and documented against — `receiving_epa` newly selected, growing RB's feature count from 4 to 5 — and flipped RB's naive-baseline verdict from "MARGINAL LIFT" to "DOES NOT CLEARLY BEAT NAIVE." This is exactly why feature selection for any position with a shipped model is **locked** in [`config/selected_features.yaml`](config/selected_features.yaml) rather than re-derived from whatever `05` happens to output on a given day — see that file, and the note near the top of `05_feature_selection.ipynb`, for how the two are meant to relate.