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
    # draft_history stores owner_id/player_id as strings (Sleeper IDs are
    # large integers that pd.read_csv otherwise infers as int64) — cast to
    # match or the merge below raises a dtype mismatch.
    overrides["owner_id"] = overrides["owner_id"].astype(draft_history["owner_id"].dtype)
    overrides["player_id"] = overrides["player_id"].astype(draft_history["player_id"].dtype)
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


def build_player_draft_anchors(draft_history: pd.DataFrame) -> pd.DataFrame:
    """
    League-wide "what round was this player actually drafted at" anchor —
    NOT scoped to any one owner. A fresh (is_keeper False) pick is a fact
    about the PLAYER, not about whoever happened to draft them: if Owner A
    drafts Rashee Rice at 9.07 in 2024, gets dropped, and Owner C later
    picks him up on waivers and keeps him in 2026, the round C loses is
    still anchored to that 2024 pick — C never drafted Rice themselves,
    but the player's draft history didn't reset just because he changed
    hands off-draft.

    Returns columns: player_id, season, anchor_round — one row per season
    a fresh draft pick of that player happened, anywhere in the league.
    Use `effective_original_round()` below to look up the anchor that was
    in force as of any given season.
    """
    fresh = draft_history[draft_history["is_keeper"].fillna(False) == False]  # noqa: E712
    return (
        fresh.groupby(["player_id", "season"], as_index=False)["round"]
        .first()
        .rename(columns={"round": "anchor_round"})
        .sort_values(["player_id", "season"])
    )


def build_pre_rule_round_history(draft_history: pd.DataFrame, rule_effective_season: int) -> pd.DataFrame:
    """
    Fallback anchor source, tier 2: for a player with NO genuine
    is_keeper=False row anywhere (so build_player_draft_anchors has
    nothing for them), fall back to their most recent recorded round from
    BEFORE the rule existed — regardless of that row's is_keeper flag.

    Real example this fixes: a player kept continuously since before 2026
    (e.g. paid round 15 in 2025) with no earlier "fresh draft" row on
    file at all. Before the rule existed, no round-loss cost applied, so
    whatever round is recorded pre-rule is a legitimate cost-free
    baseline — using it beats falling all the way to the generic
    never-drafted default, which would understate how established the
    player already was.
    """
    pre_rule = draft_history[draft_history["season"] < rule_effective_season]
    return (
        pre_rule.groupby(["player_id", "season"], as_index=False)["round"]
        .first()
        .rename(columns={"round": "pre_rule_round"})
        .sort_values(["player_id", "season"])
    )


def effective_original_round(anchors: pd.DataFrame, player_id: str, as_of_season: int):
    """Most recent fresh-draft round on record for this player at or
    before as_of_season, league-wide. None if the player has never had a
    genuine fresh-draft row (see resolve_original_round for the two
    fallback tiers)."""
    hits = anchors[(anchors["player_id"] == player_id) & (anchors["season"] <= as_of_season)]
    if hits.empty:
        return None
    return hits.sort_values("season").iloc[-1]["anchor_round"]


def effective_pre_rule_round(pre_rule_rounds: pd.DataFrame, player_id: str, rule_effective_season: int):
    """Most recent pre-rule-era round on record for this player, if any."""
    hits = pre_rule_rounds[
        (pre_rule_rounds["player_id"] == player_id) & (pre_rule_rounds["season"] < rule_effective_season)
    ]
    if hits.empty:
        return None
    return hits.sort_values("season").iloc[-1]["pre_rule_round"]


def total_rounds_for_season(season: int, rules: dict) -> int:
    """League ran 15 rounds in 2024, 18 rounds from 2025 onward
    (confirmed with commissioner) — see total_draft_rounds_by_season."""
    by_season = rules["escalation"]["total_draft_rounds_by_season"]
    return by_season.get(season, by_season["_default"])


def resolve_original_round(
    anchors: pd.DataFrame, pre_rule_rounds: pd.DataFrame, player_id: str, as_of_season: int, rules: dict
):
    """
    Three-tier fallback, never returns None:
      1. Most recent genuine fresh-draft (is_keeper=False) round, league-wide.
      2. Most recent pre-rule-era round on record (any is_keeper flag) —
         legitimate because no cost applied before the rule existed, so
         that round is a valid cost-free baseline even if it was itself a
         (then-uncosted) keeper pick.
      3. CONFIRMED with commissioner: a player with no record at all is
         treated as if drafted in the last round of the season you're
         resolving as of (15 in 2024, 18 from 2025 onward).
    """
    anchor = effective_original_round(anchors, player_id, as_of_season)
    if anchor is not None:
        return anchor

    rule_effective_season = rules["escalation"]["rule_effective_season"]
    pre_rule_anchor = effective_pre_rule_round(pre_rule_rounds, player_id, rule_effective_season)
    if pre_rule_anchor is not None:
        return pre_rule_anchor

    return total_rounds_for_season(as_of_season, rules)


def build_keeper_ledger(draft_history: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """
    For each (owner_id, player_id) chain, sorted chronologically:
      - a pick with is_keeper falsy (False/None/NaN) is a fresh draft.
      - a pick with is_keeper truthy COMPOUNDS off the player's LEAGUE-WIDE
        draft anchor (see build_player_draft_anchors): cost = `original_round
        - rounds_lost * keeps_since_rule_start`.

    original_round resolves league-wide (not owner-scoped — a waiver
    pickup still anchors to wherever the player was actually drafted),
    with a fallback to the last round for players never drafted at all
    (see resolve_original_round).

    Two different counters matter here, and conflating them was a real
    bug (caught via a real example: a player kept once BEFORE the 2026
    rule existed, then kept again in 2026, was incorrectly compounding as
    if 2026 were the player's SECOND rule-era keep instead of the first):
      - `consecutive_keeps`: the owner's full keep-streak across all of
        recorded history, regardless of which rule was in effect. Kept
        for reference/debugging only — NOT used in the formula.
      - `keeps_since_rule_start`: resets to 0 the moment a chain crosses
        into `rule_effective_season`, independent of whatever streak
        existed before. THIS is what the compounding formula uses — 2026
        is always step 1 for any player, even if they were already being
        kept under an older or informal rule beforehand.

    `expected_round_formula`/`formula_mismatch` are only computed for
    season >= `rule_effective_season`; earlier seasons are recorded but
    not validated, since a different or no formal rule applied then.

    NOTE ON SCOPE: this ledger does NOT project future keeper cost for
    currently-rostered players — it only covers rows that already went
    through an actual draft pick. It intentionally has no
    `next_season_keeper_round` column (an earlier version did, computed
    naively as `round_paid_actual - rounds_lost` with none of the anchor
    fallbacks above) — that column silently disagreed with the correct
    forward projection once waiver-acquired keepers, the rule-era
    boundary, and UDFA fallbacks were fixed. The single source of truth
    for "what would it cost to keep this player next year" is
    scripts/project_roster_keeper_costs.py, which reads this ledger's
    `keeps_since_rule_start`/`is_keeper` columns as input but runs the
    full three-tier anchor resolution (including the in-season-waiver
    branch) on top — covering every current roster, not just players
    who already have draft history.
    """
    rounds_lost = rules["escalation"]["rounds_lost"]
    min_round = rules["escalation"]["min_round"]
    rule_effective_season = rules["escalation"]["rule_effective_season"]

    anchors = build_player_draft_anchors(draft_history)
    pre_rule_rounds = build_pre_rule_round_history(draft_history, rule_effective_season)

    ledger_rows = []

    for (owner_id, player_id), grp in draft_history.groupby(["owner_id", "player_id"]):
        grp = grp.sort_values("season")
        consecutive_keeps = 0
        keeps_since_rule_start = 0
        entered_rule_era = False

        for _, row in grp.iterrows():
            is_keeper = bool(row["is_keeper"]) if pd.notna(row["is_keeper"]) else False
            consecutive_keeps = 0 if not is_keeper else consecutive_keeps + 1

            rule_applies = row["season"] >= rule_effective_season
            if rule_applies and not entered_rule_era:
                # First row at/after the rule's start for this chain: the
                # compounding counter restarts here NO MATTER what the
                # keep streak looked like before the rule existed.
                keeps_since_rule_start = 0
                entered_rule_era = True

            if rule_applies:
                keeps_since_rule_start = 0 if not is_keeper else keeps_since_rule_start + 1

            original_round = resolve_original_round(anchors, pre_rule_rounds, player_id, row["season"], rules)
            expected_round = (
                max(min_round, original_round - rounds_lost * keeps_since_rule_start)
                if rule_applies else None
            )

            ledger_rows.append({
                "season": row["season"],
                "owner_id": owner_id,
                "player_id": player_id,
                "round_paid_actual": row["round"],
                "is_keeper": is_keeper,
                "consecutive_keeps": consecutive_keeps,
                "keeps_since_rule_start": keeps_since_rule_start if rule_applies else None,
                "original_round": original_round,
                "expected_round_formula": expected_round,
                "formula_mismatch": (
                    expected_round is not None and expected_round != row["round"]
                ),
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
