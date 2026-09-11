"""
Phase 1B — Keeper Ledger.

Deterministic, rules-based. NO machine learning happens here, and this
module has zero dependency on the Stage 1 XGBoost pipeline (see roadmap
Phase 4). It answers one question per (owner, player, season): what round
did this team pay to keep or draft this player?

Grain of the output: one row per (season, owner_id, player_id).
`owner_id` (Sleeper's stable user_id) is used as the team identifier —
NOT `roster_id`, which Sleeper reassigns every season and is therefore
unsafe to join across years.
"""

import os
import yaml
import pandas as pd

from src.ingestion import sleeper_client as sc


# ---------------------------------------------------------------------------
# Step 1: Raw ingestion — walk the league's season chain and flatten every
# draft into a single long table.
# ---------------------------------------------------------------------------

def build_draft_history(league_id: str) -> pd.DataFrame:
    """
    Live pull. Requires network access to api.sleeper.app.

    Returns columns:
        season, owner_id, roster_id, player_id, round, pick_no, is_keeper
    """
    chain = sc.get_league_chain(league_id)
    rows = []

    for league in chain:
        season = league["season"]
        season_league_id = league["league_id"]

        rosters = sc.get_rosters(season_league_id)
        # roster_id -> owner_id map, THIS SEASON ONLY (roster_id is not
        # stable across seasons, so this map must be rebuilt every loop)
        roster_to_owner = {r["roster_id"]: r["owner_id"] for r in rosters}

        drafts = sc.get_drafts_for_league(season_league_id)
        if not drafts:
            continue  # season had no completed draft on record

        for draft in drafts:
            picks = sc.get_draft_picks(draft["draft_id"])
            for pick in picks:
                owner_id = roster_to_owner.get(pick["roster_id"])
                if owner_id is None:
                    continue  # orphaned roster slot (no manager assigned)
                rows.append({
                    "season": int(season),
                    "owner_id": owner_id,
                    "roster_id": pick["roster_id"],
                    "player_id": pick["player_id"],
                    "round": pick["round"],
                    "pick_no": pick["pick_no"],
                    "is_keeper": pick.get("is_keeper"),
                })

    df = pd.DataFrame(rows)
    return df.sort_values(["owner_id", "player_id", "season"]).reset_index(drop=True)


def apply_manual_overrides(draft_history: pd.DataFrame, override_path: str) -> pd.DataFrame:
    """
    Sleeper's `is_keeper` flag is frequently null even in real keeper
    leagues. This merges a commissioner-maintained CSV of known keeper
    picks for seasons/players where the native flag can't be trusted.

    Expected CSV columns: season, owner_id, player_id, is_keeper (bool)
    """
    if not os.path.exists(override_path):
        return draft_history

    overrides = pd.read_csv(override_path)
    df = draft_history.merge(
        overrides.rename(columns={"is_keeper": "is_keeper_override"}),
        on=["season", "owner_id", "player_id"],
        how="left",
    )
    df["is_keeper"] = df["is_keeper_override"].combine_first(df["is_keeper"])
    return df.drop(columns=["is_keeper_override"])


# ---------------------------------------------------------------------------
# Step 2: Escalation ledger — deterministic rules applied chronologically
# per (owner, player) chain.
# ---------------------------------------------------------------------------

def load_keeper_rules(config_path: str = "config/keeper_rules.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_keeper_ledger(draft_history: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """
    For each (owner_id, player_id) chain, sorted chronologically:
      - a pick with is_keeper falsy (False/None/NaN) is a fresh draft:
        resets the escalation chain, anchors `original_round`.
      - a pick with is_keeper truthy increments `consecutive_keeps` and
        keeps the prior `original_round` anchor.

    The keeper discount is FLAT, not compounding (commissioner-confirmed):
    a kept player always costs (original_round - rounds_lost_flat), no
    matter how many consecutive years in a row they've been kept — it does
    not escalate further each year. This flat rule took effect starting
    `rule_start_season`; seasons before that are not checked against it, so
    `expected_round_formula` is left null and `formula_mismatch` is False
    for those rows (fresh-draft rows are the one exception — a fresh draft
    trivially "expects" the round it was actually drafted at, so that part
    is season-independent).

    `expected_round_formula` / `formula_mismatch` are the data-quality
    signal for missed is_keeper flags or off-book league rule exceptions,
    scoped to seasons the flat rule actually governed.
    """
    rounds_lost = rules["escalation"]["rounds_lost_flat"]
    min_round = rules["escalation"]["min_round"]
    rule_start_season = rules["escalation"].get("rule_start_season")

    ledger_rows = []

    for (owner_id, player_id), grp in draft_history.groupby(["owner_id", "player_id"]):
        grp = grp.sort_values("season")
        consecutive_keeps = 0
        original_round = None

        for _, row in grp.iterrows():
            is_keeper = bool(row["is_keeper"]) if pd.notna(row["is_keeper"]) else False

            if not is_keeper:
                original_round = row["round"]
                consecutive_keeps = 0
            else:
                consecutive_keeps += 1

            # The flat formula's answer, independent of season — used both
            # as the validated `expected_round_formula` (when in scope) and
            # as the always-on `next_season_keeper_round` projection below.
            flat_round = (
                max(min_round, original_round - rounds_lost)
                if original_round is not None else None
            )

            rule_in_effect = (
                rule_start_season is None or row["season"] >= rule_start_season
            )

            if original_round is None:
                expected_round = None
            elif not is_keeper:
                # Fresh draft: the round paid IS the round drafted at — this
                # trivially "matches" regardless of season/rule scope.
                expected_round = original_round
            elif rule_in_effect:
                expected_round = flat_round
            else:
                # Kept before the flat rule took effect — not modeled, so
                # don't validate/flag against it.
                expected_round = None

            ledger_rows.append({
                "season": row["season"],
                "owner_id": owner_id,
                "player_id": player_id,
                "round_paid_actual": row["round"],
                "is_keeper": is_keeper,
                "consecutive_keeps": consecutive_keeps,
                "original_round": original_round,
                "expected_round_formula": expected_round,
                "formula_mismatch": (
                    expected_round is not None and expected_round != row["round"]
                ),
                "next_season_keeper_round": flat_round,
            })

    return pd.DataFrame(ledger_rows)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rules = load_keeper_rules()
    league_id = rules["league_id"]

    print(f"Pulling draft history for league {league_id} ...")
    raw = build_draft_history(league_id)
    raw = apply_manual_overrides(raw, rules["keeper_flag_source"]["manual_override_path"])
    raw.to_parquet("data/raw/draft_history.parquet", index=False)

    ledger = build_keeper_ledger(raw, rules)
    ledger.to_parquet("data/processed/keeper_ledger.parquet", index=False)

    n_mismatches = ledger["formula_mismatch"].sum()
    print(f"Ledger built: {len(ledger)} rows. {n_mismatches} formula mismatches "
          f"flagged for manual review (missed is_keeper flags or rule exceptions).")
