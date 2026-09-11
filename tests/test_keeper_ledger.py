"""
Unit tests for the keeper ledger escalation logic. Runs entirely on
synthetic draft_history — no Sleeper API calls, no network dependency.
This is deliberate: the ledger's correctness is a pure function of the
escalation rules and should be verifiable without live data.
"""

import pandas as pd
import pytest

from src.keeper_ledger.build_ledger import build_keeper_ledger

RULES = {
    "escalation": {"rounds_lost": 2, "min_round": 1, "compounding": True, "rule_effective_season": 2026, "total_draft_rounds_by_season": {2024: 15, "_default": 18}},
}


def test_fresh_draft_resets_chain():
    """A player drafted fresh (is_keeper False) always anchors at their
    actual round with zero consecutive keeps."""
    df = pd.DataFrame([
        {"season": 2023, "owner_id": "u1", "player_id": "p1", "round": 5, "is_keeper": False},
    ])
    ledger = build_keeper_ledger(df, RULES)
    row = ledger.iloc[0]
    assert row["original_round"] == 5
    assert row["consecutive_keeps"] == 0
    assert row["next_season_keeper_round"] == 3  # 5 - 2


def test_escalation_compounds_across_keep_years():
    """CONFIRMED with commissioner: drafted Rd 8 -> kept -> kept compounds
    8 -> 6 -> 4. It does NOT plateau at 6."""
    df = pd.DataFrame([
        {"season": 2025, "owner_id": "u1", "player_id": "p1", "round": 8, "is_keeper": False},
        {"season": 2026, "owner_id": "u1", "player_id": "p1", "round": 6, "is_keeper": True},
        {"season": 2027, "owner_id": "u1", "player_id": "p1", "round": 4, "is_keeper": True},
    ])
    ledger = build_keeper_ledger(df, RULES).sort_values("season")
    rounds = ledger["round_paid_actual"].tolist()
    consecutive = ledger["consecutive_keeps"].tolist()
    assert rounds == [8, 6, 4]
    assert consecutive == [0, 1, 2]
    assert ledger.iloc[-1]["expected_round_formula"] == 4
    assert ledger.iloc[-1]["formula_mismatch"] == False  # noqa: E712
    assert ledger.iloc[-1]["next_season_keeper_round"] == 2  # 4 - 2


def test_flat_pattern_now_flags_as_mismatch():
    """If post-2026 data shows a player NOT escalating on a second keep
    year (plateauing instead of compounding), that should now be
    flagged — it doesn't match the confirmed compounding rule."""
    df = pd.DataFrame([
        {"season": 2025, "owner_id": "u1", "player_id": "p1", "round": 8, "is_keeper": False},
        {"season": 2026, "owner_id": "u1", "player_id": "p1", "round": 6, "is_keeper": True},
        {"season": 2027, "owner_id": "u1", "player_id": "p1", "round": 6, "is_keeper": True},  # should be 4
    ])
    ledger = build_keeper_ledger(df, RULES).sort_values("season")
    assert ledger.iloc[-1]["expected_round_formula"] == 4
    assert ledger.iloc[-1]["formula_mismatch"] == True  # noqa: E712


def test_pre_rule_seasons_are_never_flagged():
    """The compounding rule took effect in 2026. A 2024/2025 keep that
    doesn't match the formula is NOT a data-quality problem — a different
    or no formal rule applied then. These rows must never show
    formula_mismatch, regardless of what round was actually paid."""
    df = pd.DataFrame([
        {"season": 2023, "owner_id": "u1", "player_id": "p1", "round": 8, "is_keeper": False},
        {"season": 2024, "owner_id": "u1", "player_id": "p1", "round": 6, "is_keeper": True},
        {"season": 2025, "owner_id": "u1", "player_id": "p1", "round": 5, "is_keeper": True},  # wouldn't match either formula
    ])
    ledger = build_keeper_ledger(df, RULES).sort_values("season")
    assert ledger.iloc[1]["expected_round_formula"] is None
    assert ledger.iloc[2]["expected_round_formula"] is None
    assert ledger.iloc[1]["formula_mismatch"] == False  # noqa: E712
    assert ledger.iloc[2]["formula_mismatch"] == False  # noqa: E712


def test_rule_applies_starting_exactly_at_effective_season():
    """2026 onward IS checked against the compounding rule."""
    df = pd.DataFrame([
        {"season": 2025, "owner_id": "u1", "player_id": "p1", "round": 8, "is_keeper": False},
        {"season": 2026, "owner_id": "u1", "player_id": "p1", "round": 6, "is_keeper": True},  # matches
        {"season": 2027, "owner_id": "u1", "player_id": "p1", "round": 6, "is_keeper": True},  # would be a mismatch (should be 4)
    ])
    ledger = build_keeper_ledger(df, RULES).sort_values("season")
    assert ledger.iloc[1]["formula_mismatch"] == False  # noqa: E712
    assert ledger.iloc[2]["expected_round_formula"] == 4
    assert ledger.iloc[2]["formula_mismatch"] == True  # noqa: E712


def test_escalation_floors_at_min_round():
    """A player kept enough consecutive years that the formula would go
    below Round 1 must floor at Round 1, never go negative or zero."""
    df = pd.DataFrame([
        {"season": 2025, "owner_id": "u1", "player_id": "p1", "round": 3, "is_keeper": False},
        {"season": 2026, "owner_id": "u1", "player_id": "p1", "round": 1, "is_keeper": True},
        {"season": 2027, "owner_id": "u1", "player_id": "p1", "round": 1, "is_keeper": True},
    ])
    ledger = build_keeper_ledger(df, RULES).sort_values("season")
    assert ledger.iloc[-1]["next_season_keeper_round"] == 1
    assert ledger.iloc[-1]["expected_round_formula"] == 1


def test_null_is_keeper_treated_as_fresh_draft():
    """Sleeper's is_keeper is frequently null even for real keeper leagues.
    Null must be treated as False (fresh draft), not crash the pipeline."""
    df = pd.DataFrame([
        {"season": 2023, "owner_id": "u1", "player_id": "p1", "round": 7, "is_keeper": None},
    ])
    ledger = build_keeper_ledger(df, RULES)
    assert ledger.iloc[0]["is_keeper"] == False  # noqa: E712
    assert ledger.iloc[0]["consecutive_keeps"] == 0


def test_formula_mismatch_flag_catches_rule_violations():
    """If the actual round paid doesn't match what the compounding
    formula predicts (post-2026), flag it for manual review rather than
    silently trusting it — this is how missed is_keeper flags or off-book
    exceptions get caught."""
    df = pd.DataFrame([
        {"season": 2025, "owner_id": "u1", "player_id": "p1", "round": 8, "is_keeper": False},
        # Formula predicts round 6, but Sleeper recorded round 5 — mismatch.
        {"season": 2026, "owner_id": "u1", "player_id": "p1", "round": 5, "is_keeper": True},
    ])
    ledger = build_keeper_ledger(df, RULES).sort_values("season")
    assert ledger.iloc[-1]["expected_round_formula"] == 6
    assert ledger.iloc[-1]["formula_mismatch"] == True  # noqa: E712


def test_waiver_acquired_keeper_anchors_to_league_wide_draft_history():
    """The Rashee Rice case: Owner A drafts the player fresh in 2024
    (round 9). Owner A later drops him; Owner C picks him up on waivers
    mid-2025 (no draft record for C in 2025 at all — waivers aren't draft
    picks). In 2026, Owner C keeps him for the first time, costing round 7.
    C never drafted this player themselves, but the anchor must still
    resolve to the 2024 fresh pick, not None."""
    df = pd.DataFrame([
        {"season": 2024, "owner_id": "A", "player_id": "rice", "round": 9, "is_keeper": False},
        {"season": 2026, "owner_id": "C", "player_id": "rice", "round": 7, "is_keeper": True},
    ])
    ledger = build_keeper_ledger(df, RULES)
    c_row = ledger[(ledger["owner_id"] == "C") & (ledger["season"] == 2026)].iloc[0]
    assert c_row["original_round"] == 9
    assert c_row["consecutive_keeps"] == 1
    assert c_row["expected_round_formula"] == 7  # 9 - 2*1
    assert c_row["formula_mismatch"] == False  # noqa: E712


def test_independent_owners_do_not_cross_contaminate():
    """Two different owners holding the same player_id (e.g. after a trade,
    or a dynasty startup re-draft) must have fully independent chains."""
    df = pd.DataFrame([
        {"season": 2023, "owner_id": "u1", "player_id": "p1", "round": 4, "is_keeper": False},
        {"season": 2023, "owner_id": "u2", "player_id": "p1", "round": 9, "is_keeper": False},
    ])
    ledger = build_keeper_ledger(df, RULES)
    u1_row = ledger[ledger["owner_id"] == "u1"].iloc[0]
    u2_row = ledger[ledger["owner_id"] == "u2"].iloc[0]
    assert u1_row["next_season_keeper_round"] == 2
    assert u2_row["next_season_keeper_round"] == 7


def test_pre_rule_keep_streak_does_not_carry_into_rule_era_compounding():
    """THE RICE BUG: a player kept once BEFORE the 2026 rule existed
    (under an old/informal rule), then kept again in 2026, must treat 2026
    as compounding STEP 1 (original_round - 2), not step 2 (original_round
    - 4). The pre-2026 keep-year must not count toward the new rule's
    compounding counter, even though it's the same owner keeping the same
    player continuously."""
    df = pd.DataFrame([
        {"season": 2024, "owner_id": "u1", "player_id": "rice", "round": 9, "is_keeper": False},
        {"season": 2025, "owner_id": "u1", "player_id": "rice", "round": 8, "is_keeper": True},  # pre-rule keep, some old/informal cost
        {"season": 2026, "owner_id": "u1", "player_id": "rice", "round": 7, "is_keeper": True},  # first RULE-ERA keep
    ])
    ledger = build_keeper_ledger(df, RULES).sort_values("season")
    row_2026 = ledger[ledger["season"] == 2026].iloc[0]
    assert row_2026["keeps_since_rule_start"] == 1
    assert row_2026["expected_round_formula"] == 7  # 9 - 2*1, NOT 5
    assert row_2026["formula_mismatch"] == False  # noqa: E712
    assert row_2026["next_season_keeper_round"] == 5  # one more step from here: 7 - 2


def test_undrafted_player_falls_back_to_last_round():
    """CONFIRMED with commissioner: a player never drafted anywhere in
    league history (pure waiver-wire/UDFA add) is treated as if drafted
    in the last round of the CURRENT season's format (18 rounds from 2025
    onward) — first keep in 2026 costs round 16."""
    df = pd.DataFrame([
        {"season": 2026, "owner_id": "u1", "player_id": "never_drafted_guy", "round": 16, "is_keeper": True},
    ])
    ledger = build_keeper_ledger(df, RULES)
    row = ledger.iloc[0]
    assert row["original_round"] == 18
    assert row["expected_round_formula"] == 16  # 18 - 2*1
    assert row["formula_mismatch"] == False  # noqa: E712


def test_undrafted_fallback_uses_15_rounds_for_2024():
    """The league ran 15 rounds in 2024, not 18 — the UDFA fallback must
    resolve to the round count that was actually in effect that season,
    not blindly use the current 18."""
    df = pd.DataFrame([
        {"season": 2024, "owner_id": "u1", "player_id": "never_drafted_2024_guy", "round": 15, "is_keeper": False},
    ])
    ledger = build_keeper_ledger(df, RULES)
    row = ledger.iloc[0]
    assert row["original_round"] == 15  # not 18


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
