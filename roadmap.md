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

## Phase 4 — Stage 1 Model: XGBoost VORP Regressor — 🟡 NEXT
**Goal:** Trained, validated, per-position models producing the Keeper Value Score (KVS) — **rescoped to 4 positions: QB, RB, WR, TE.**

- [ ] Four separate notebooks: `06a_model_qb.ipynb`, `06b_model_rb.ipynb`, `06c_model_wr.ipynb`, `06d_model_te.ipynb` — each trained on its Phase 3-selected feature set
- [ ] **Real hyperparameter tuning** (nested inside each walk-forward fold) — the Phase 3 naive-baseline results used an untuned estimator; this is the step that tests whether real tuning meaningfully closes the gap, and by how much
- [ ] Monotonic constraints where domain logic is unambiguous
- [ ] Per-fold MAE/RMSE/Spearman saved to `/data/processed/fold_metrics_<position>.csv` (already exists in skeleton form from Phase 3; Phase 4 replaces these with tuned-model numbers)
- [ ] **Rerun the naive-baseline comparison with tuned models** — this is the real, final version of the reality check Phase 3 started; expect a modest but real improvement over the untuned numbers, and be honest in the writeup if it doesn't move much (persistence baselines are famously hard to beat in fantasy sports — that's a legitimate, expected finding, not a failure)
- [ ] Lock final model artifacts trained on all data through the most recent complete season
- [ ] `07_model_comparison_summary.ipynb` — one consolidated notebook comparing all 4 positions' final tuned performance side-by-side (naive vs. tuned, MAE, Spearman)

**Exit criterion:** Four `KVS = predict(player, season)` functions (QB/RB/WR/TE only), each with a fold-by-fold validation report — including an honest, quantified comparison against the naive baseline, not just an assumed improvement.

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
