# Project Roadmap: Keeper League Trade Fairness Engine

## Phase 0 — Repo & Environment Setup — ✅ COMPLETE
## Phase 1 — Data Acquisition & League-Scoring Normalization — ✅ COMPLETE
(Fumbles and kicker scoring bugs found and fixed, both verified exact against real Sleeper output.)

## Phase 1B — Keeper Ledger & Draft Capital Curve — ✅ COMPLETE
(League-wide draft anchors, compounding escalation confirmed with commissioner, three-tier fallback logic, validated against real named examples — Rice, Irving, Watson, Wicks.)

## Phase 2 — Target Variable Construction: Next-Season VORP — ✅ COMPLETE
(Full 2008/2009–2025 range, all 6 league positions including DEF via a second Sleeper endpoint, FLEX split measured from real 2025 lineup data, a silent inner-join bug and a DEF duplication bug both found and fixed at the source with permanent regression guards. `vorp_labels.parquet`: 11,417 rows.)

---

## Phase 3 — Feature Engineering — ✅ COMPLETE

**Notebooks:** `03_feature_investigation.ipynb` (data availability triage) → `04_feature_creation.ipynb` (feature construction) → `05_feature_selection.ipynb` (leakage audit, RFECV, naive-baseline check)

- [x] **Step 1 — Data availability triage**, all four categories tested directly (not assumed): age (2008–2025, 100%), injury (2009–2025 only, 2008 hard-floor), draft capital (2008–2025, 69.2% + real UDFAs), OL quality (no free named metric exists — two self-built proxies instead: pass-protection via sack rate, run-blocking via stuff rate, confirmed non-redundant at r=-0.517)
- [x] **Step 2 — Confirmed-data features**: opportunity (target_share, air_yards_share, wopr), efficiency (EPA trio), positional scarcity (`scarcity_z`, standardized within position/season), multi-year trend (`vorp_delta_yoy`)
- [x] **Step 3 — Gap-filling features**: age + age², injury designation count (2008 correctly NaN not 0), both OL proxies, draft capital tier + inverse-pick
- [x] **Step 4 — Statistical selection**: 16-feature leakage audit (per-feature, individually confirmed), RFECV wrapped in walk-forward CV (7–8 real folds) per position, XGBoost gain-importance validated against football sense (not trusted blindly — caught K's `draft_tier_Round 2-3` as a 5-player noise artifact vs. TE's genuinely-populated 97-player version of the same feature)
- [x] **Naive baseline reality check** (added after real pushback on whether `scarcity_z`'s dominance was actually meaningful): XGBoost vs. "assume next season repeats this season" across all 6 positions, walk-forward. Result: marginal lift everywhere (0.9%–11.9% MAE improvement), and **DEF came back with a worse Spearman than naive (0.188 vs. 0.307)** — a genuine red flag, not just a low ceiling.

**Selected feature sets (RFECV-validated, football-sense-checked):**
- QB (6 features): scarcity_z, draft_pick_inverse, vorp_delta_yoy, passing_epa, age, ol_pass_protection_proxy
- RB (4 features): scarcity_z, draft_pick_inverse, age, vorp_delta_yoy — entire usage trio + OL run-block proxy eliminated (consistent, 0% selection across every fold)
- WR (8 features): scarcity_z, wopr, vorp_delta_yoy, receiving_epa, air_yards_share, age, draft_pick_inverse, target_share (target_share selection unstable — flip-flops 75% of folds, collinear with wopr)
- TE (9 features): scarcity_z, air_yards_share, receiving_epa, wopr, target_share, draft_tier_Round 2-3, draft_pick_inverse, vorp_delta_yoy, injury_designations_count

**Exit criterion:** Met — reduced, validated feature matrix per position, written record of eliminations, every retained feature checked for football sense, and an honest reality check against a naive baseline rather than trusting feature-importance rankings alone.

---

## ⚠️ SCOPE DECISION: K and DEF excluded from full modeling (made before Phase 4)

**Decision:** No walk-forward-trained XGBoost model, no RFECV-tuned feature set, no SHAP explainer will be built for Kicker or Team Defense. Both are excluded from Phase 4 onward.

**Reasoning:**
1. **Neither position is ever actually traded in this league.** The entire point of Stage 1's KVS is to price trade value — building full modeling infrastructure for positions that structurally never enter a trade spends real engineering effort on a capability the trade engine will never call.
2. **DEF's own evidence argues against it independently of the trading question.** The untuned naive-baseline check showed DEF's model *actively underperforming* simple persistence on Spearman rank correlation (0.188 vs. 0.307) — worse than doing nothing, not just "a small lift." Combined with DEF's thin 3-candidate feature set (a structural limit of modeling a team-unit position with individual-player-shaped data) and defense's well-known real-world year-to-year volatility (turnover luck, scheme/personnel changes), this is real evidence the position isn't well-modeled by what's currently available — not just an unfavorable use case.
3. **K's untuned result was also weak** (Spearman 0.366, barely above the 0.347 naive baseline; the only feature showing real signal in K's chart turned out to be a 5-player noise artifact once checked).

**What replaces full modeling for K/DEF:** a simple, explicitly-labeled non-ML heuristic (e.g. current-season raw VORP, or a short rolling average) wherever K/DEF value needs to be referenced elsewhere in the pipeline (roster overviews, keeper-cost context) — never presented with the same confidence or precision as the four real models.

**Documented in:** `notebooks/06_scope_decision_k_def.ipynb` — a short, standalone notebook stating this reasoning with the real numbers behind it, so a reviewer sees it as a deliberate, evidence-based engineering decision rather than an unexplained gap.

---

## Phase 4 — Stage 1 Model: XGBoost VORP Regressor — 🟡 NEARLY DONE
**Goal:** Trained, validated, per-position models producing the Keeper Value Score (KVS) — **rescoped to 4 positions: QB, RB, WR, TE.**

- [x] Four separate notebooks: `06a_model_qb.ipynb`, `06b_model_rb.ipynb`, `06c_model_wr.ipynb`, `06d_model_te.ipynb` — all four done, each trained on its Phase 3-selected (now locked, see `config/selected_features.yaml`) feature set
- [x] **Real hyperparameter tuning** (nested inside each walk-forward fold, 40-trial Optuna/TPE search per fold, `max_depth` 2-8 confirmed a real or at-worst-neutral win everywhere it was checked) — done for all four positions
- [x] Monotonic constraints where domain logic is unambiguous — checked independently per position, no assumption carried forward: QB rejected them, RB kept them, WR rejected them, TE kept them on a genuine near-tie (see the constraint-diagnostic finding below)
- [x] Per-fold MAE/RMSE/Spearman saved to `/data/processed/fold_metrics_<position>.csv` for all four positions
- [x] **Rerun the naive-baseline comparison with tuned models** — done for all four; QB/RB/WR each show a real, modest, same-direction lift on both metrics, while TE's tuned Spearman (0.689) does not clear its own naive baseline (0.691), the first position where that's true
- [x] Lock final model artifacts trained on all data through the most recent complete season — `qb_model.json`, `rb_model.json`, `wr_model.json`, `te_model.json` all exist
- [ ] `07_model_comparison_summary.ipynb` — one consolidated notebook comparing all 4 positions' final tuned performance side-by-side (naive vs. tuned, MAE, Spearman) — next up

**Exit criterion:** Four `KVS = predict(player, season)` functions (QB/RB/WR/TE only), each with a fold-by-fold validation report — including an honest, quantified comparison against the naive baseline, not just an assumed improvement. **Met for all four positions individually**; only the consolidated cross-position summary notebook remains before Phase 4 can be marked fully complete.

---

## ⚠️ KNOWN FEATURE LIMITATION: `vorp_delta_yoy`'s extreme tail — confirmed cross-position, all 4 modeled positions (QB + RB + WR + TE)

**Finding:** `vorp_delta_yoy` (year-over-year VORP swing) degrades sharply in its own extreme tail (`|delta| > 100`), independently confirmed for all four modeled positions via bucket-MAE, all now backed by real, committed, re-runnable notebook code (RB's own check did not exist until it was added retroactively — see the bug-fix note below):
- **QB**: MAE 51.88 (`|delta|`≤100) vs. 67.16 (`|delta|`>100) — 1.29x worse (`05b_qb_feature_experiment.ipynb`, in-sample). The extreme bucket's *in-sample* MAE (70.09) even exceeds the shipped model's own honest *held-out* MAE (67.72).
- **RB**: MAE 45.17 vs. 64.46 — **1.43x worse**, now computed directly in `06b_model_rb.ipynb` on pooled held-out (out-of-fold) predictions, replacing an earlier one-off, non-reproducible figure (40.43/56.89, 1.41x) that was never backed by real notebook code.
- **WR**: MAE 37.09 vs. 53.90 — 1.45x worse, measured directly on pooled held-out (out-of-fold) predictions across all 8 folds. WR's own quartile breakdown shows a clean monotonic rise (31.6 → 36.7 → 39.9 → 45.1 MAE across quartiles of `|vorp_delta_yoy|`).
- **TE**: MAE 26.83 (`|delta|`≤100) vs. 44.43 (`|delta|`>100) — **1.66x worse, the largest gap of the four**. TE's own quartile breakdown (20.76 → 21.67 → 30.79 → 36.30) is monotonic too.

**All four positions now checked with real, held-out, committed code.** Robust evidence this is a structural property of the feature itself (a genuinely noisy, mean-reverting quantity by construction), not a position-specific quirk.

**Correction, now fully settled: the `scarcity_z` "extremes fit better" claim was never real, and there is no position-family split.** This roadmap and both `06c_model_wr.ipynb` and `06d_model_te.ipynb` originally stated that QB and RB show `scarcity_z`'s own extremes (top/bottom deciles) fitting *better* than the middle 80%, with only WR and TE breaking that pattern — later revised to "3 of 3 checked positions (RB, WR, TE) show the top decile as hardest, QB unverified." **QB has now been checked too** (`06a_model_qb.ipynb`, out-of-fold, by decile, not averaged), closing the last open case:
- **QB** top decile (its actual star QBs): MAE 78.37 — 1.16x worse than overall (67.28). Bottom decile: MAE 36.97 — 0.55x of overall, the easiest bottom decile of any position checked. Combined average: 57.67, well *below* overall — the most misleading version of the averaged number yet, since QB's unusually easy bottom decile is large enough to fully offset a genuinely hard top decile.
- **RB** top decile: MAE 68.53 — 1.47x worse than overall (46.65). Bottom decile: MAE 26.70 — 0.57x of overall. Combined average: 47.62, close to neutral.
- **WR** top decile: MAE 55.64 — 1.50x worse than overall (37.12). Bottom decile: MAE 21.85 — 0.59x of overall. Combined average: 38.75, close to neutral.
- **TE** top decile: MAE 44.82 — 1.70x worse than overall (26.31). Bottom decile: MAE 14.97 — 0.57x of overall. Combined average: 29.90, moderately worse.

**4 of 4 positions, all checked with real, committed, decile-separated, out-of-fold code, show the identical pattern**: the model's own best-performing players are consistently its *hardest* segment to predict (1.16x-1.70x worse than overall), while its worst/replacement-level players are consistently its *easiest* (0.55x-0.59x of overall, a strikingly tight band across four unrelated positions). Averaging the two deciles together — the mistake behind every one of the original "extremes fit better" claims — lands anywhere from "well below overall" (QB) to "moderately worse" (TE) depending only on how easy that position's bottom decile happens to be, which is exactly why it looked like a real per-position split until each position was actually checked.

**The "receiving-volume vs. run/pocket-presence position-family" hypothesis is retired, not kept as a caveat.** It does not survive QB's own real check (QB confirms the pattern rather than breaking it) and there is no remaining exception to relabel — this is a likely-universal phenomenon: elite seasons are inherently harder to sustain, and harder for the model to predict the sustaining of, than replacement-level seasons are easy to predict the continuity of. Worth carrying into Phase 5's SHAP work as a real, model-wide property, not a position-specific one.

**Bug found and fixed across QB, RB, and WR: `low_confidence_extreme_delta` silently mis-flagged rookies.** The flag is computed as `|vorp_delta_yoy| > 100`. In pandas/NumPy, `NaN > 100` evaluates to `False` — **not** an error, and **not** `True` — so any row with no computable prior-season delta at all (a true rookie) read as "not extreme," indistinguishable from a genuinely small, reliable delta. This is a general Python/pandas gotcha worth naming plainly, since it's the kind of silent-`False`-on-`NaN` behavior that could recur anywhere else in this codebase a boolean comparison is built directly on a nullable column, not just here: **any `series > threshold` or `series.abs() > threshold` check needs an explicit decision about what `NaN` should mean before being trusted**, because pandas will never raise on it — it will just quietly pick `False`.
- **Discovered via TE's live 2026 section** (Tyler Warren, a true rookie, showed as unflagged despite having no delta at all), then confirmed identically in QB's and WR's already-shipped notebooks — QB's own headline "worst miss" narrative (Jayden Daniels) had incorrectly described his `vorp_delta_yoy` as "pointing toward continued strong value" when it was actually `NaN`.
- **Scope**: roughly 25-30% of every position's training rows have no computable delta (QB 25.3%, RB 29.7%, WR 29.4%, TE 28.2%) — a large, recurring share, not an edge case.
- **Fix**: a separate `no_delta_history` flag (`vorp_delta_yoy.isna()`), kept distinct from `low_confidence_extreme_delta` rather than merged into it as `True`, now implemented in all four `06x_model_*.ipynb` notebooks (RB's added alongside its first-ever delta-bucket/`scarcity_z` diagnostic code, since it never had any of this before).
- **Why a separate flag rather than defaulting rookies to the "safer," more cautious `True`**: checked before deciding, not assumed — an in-sample comparison against each shipped model found no-delta rows fit about as well as the general population (e.g. QB: 56.27 vs. 55.39 overall; RB: 38.36 vs. 43.21; WR: 31.00 vs. 34.15; TE: 20.73 vs. 25.08) and meaningfully *better* than the extreme-delta bucket at every position. This is **weaker evidence than the held-out-confirmed delta-bucket pattern above** — it's in-sample, so a good result here is a weaker signal than a bad one would have been (the model could simply be fitting these rows well without generalizing) — but it's real, consistent evidence across all four positions against defaulting to maximum caution, and it preserves a genuine distinction (unmeasured vs. measurably-large) that merging the two flags would have erased.

**RB was fully retrofitted, not just patched**: since RB never had a delta-bucket diagnostic, `scarcity_z` tail check, or any player-level prediction table before this pass, adding the flags "correctly" required building all of that (out-of-fold prediction collection, the delta-bucket and `scarcity_z` sections, a 2024 real-predictions table, and a live 2026 top-5 section) rather than bolting a column onto nothing. `06b_model_rb.ipynb`'s own worst miss under this new section is Christian McCaffrey (predicted -93.1, actual +240.2, a 333.3-point miss — the largest of any position notebook — driven by a 2024 season nearly wiped out by injury that fully reversed in a healthy 2025), flagged `low_confidence_extreme_delta` and directly relevant to reading his own flagged 2026 prediction (+74.7) with real skepticism.

---

## Phase 5 — Interpretability Layer: SHAP
**Goal:** Every KVS number is explainable — for QB/RB/WR/TE only, matching Phase 4's scope.

- [ ] SHAP values per player-season prediction, per position
- [ ] Summary plots (global) + force/waterfall plots (local, per-player — what gets shown in an actual trade dispute)
- [ ] `explain_player(player, season) -> top_5_drivers` reusable function

**Exit criterion:** Given any QB/RB/WR/TE player, a human-readable "why this KVS" breakdown.

---

## Phase 6 — Stage 2: Trade Fairness Engine
**Goal:** KVS → actionable trade verdict, combined with Phase 1B's keeper cost machinery.

- [ ] Draft capital curve (`draft_capital_curve.py`, dormant since Phase 1B) — now finally computable with real VORP data. Confirmed applicable to BOTH keeper cost pricing AND real traded draft picks (same underlying "what does a Round N pick typically return" question)
- [ ] Open design questions for this phase specifically: how much to discount future draft picks (more uncertain than this year's), and whether to blend multiple draft-class years or treat each independently
- [ ] Net KVS delta = predicted KVS − keeper_cost_VORP, using `scripts/project_roster_keeper_costs.py`'s validated projections
- [ ] K/DEF trades (if they ever occur) handled via the simple heuristic from the scope decision, not the full model pipeline
- [ ] Positional scarcity/need adjustment on top of raw delta
- [ ] Fairness score normalization
- [ ] Combine with SHAP output: fairness verdict + top drivers per side

**Exit criterion:** Given two proposed trade packages, a fairness score and explanation of what's driving the imbalance.

---

## Phase 7 — Backtesting Against Real League History
## Phase 8 — Packaging & Delivery

(Unchanged from original plan — see prior roadmap version for full detail.)

---

## Cross-Cutting Rules
- No feature or data split gets added without checking it against the walk-forward boundary first.
- Every metric gets reported per-fold before it gets averaged.
- Scoring config (`league_scoring_rules.yaml`) is the only place point values live.
- Keeper cost/draft round is a Stage 2-only input, never a Stage 1 feature.
- The keeper ledger is deterministic rules-based logic, not ML.
- **Every data-availability assumption gets documented explicitly, in-notebook, the moment it's made.**
- Every scoring formula gets verified against real Sleeper output for at least one known player.
- **Every feature-importance or model-performance claim gets checked against a naive baseline before being trusted** — added as a rule after this exact check surfaced DEF's real weakness in Phase 3.
- **Scope decisions (like excluding K/DEF) get documented with the evidence behind them**, not silently absorbed as an unexplained gap.
- Commit fold metrics and SHAP summaries to the repo.
