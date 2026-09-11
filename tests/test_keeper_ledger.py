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
    "escalation": {"rounds_lost_flat": 2, "min_round": 1, "rule_start_season": 2026},
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


def test_flat_discount_does_not_compound_across_keep_years():
    """Drafted Rd 8 -> kept -> kept should stay flat at 8 -> 6 -> 6, NOT
    escalate further to 4. The discount is a one-time flat offset from the
    original round, applied every year the player is kept."""
    df = pd.DataFrame([
        {"season": 2024, "owner_id": "u1", "player_id": "p1", "round": 8, "is_keeper": False},
        {"season": 2026, "owner_id": "u1", "player_id": "p1", "round": 6, "is_keeper": True},
        {"season": 2027, "owner_id": "u1", "player_id": "p1", "round": 6, "is_keeper": True},
    ])
    ledger = build_keeper_ledger(df, RULES).sort_values("season")
    rounds = ledger["round_paid_actual"].tolist()
    consecutive = ledger["consecutive_keeps"].tolist()
    assert rounds == [8, 6, 6]
    assert consecutive == [0, 1, 2]
    assert ledger.iloc[-1]["expected_round_formula"] == 6  # flat, not 4
    assert ledger.iloc[-1]["formula_mismatch"] == False  # noqa: E712
    assert ledger.iloc[-1]["next_season_keeper_round"] == 6  # 8 - 2, still flat


def test_flat_discount_floors_at_min_round():
    """A player whose original round minus the flat discount would go below
    Round 1 must floor at Round 1, never go negative or zero."""
    df = pd.DataFrame([
        {"season": 2024, "owner_id": "u1", "player_id": "p1", "round": 3, "is_keeper": False},
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
    """If the actual round paid doesn't match what the flat formula
    predicts (in a season the rule governs), flag it for manual review
    rather than silently trusting it — this is how missed is_keeper flags
    or off-book exceptions get caught."""
    df = pd.DataFrame([
        {"season": 2024, "owner_id": "u1", "player_id": "p1", "round": 8, "is_keeper": False},
        # Formula predicts round 6, but Sleeper recorded round 5 — mismatch.
        {"season": 2026, "owner_id": "u1", "player_id": "p1", "round": 5, "is_keeper": True},
    ])
    ledger = build_keeper_ledger(df, RULES).sort_values("season")
    assert ledger.iloc[-1]["expected_round_formula"] == 6
    assert ledger.iloc[-1]["formula_mismatch"] == True  # noqa: E712


def test_flat_rule_not_checked_before_rule_start_season():
    """The flat rule was commissioner-confirmed to start in 2026. A keeper
    pick from an earlier season that would 'mismatch' the flat formula must
    NOT be flagged — that era isn't modeled by this rule at all."""
    df = pd.DataFrame([
        {"season": 2022, "owner_id": "u1", "player_id": "p1", "round": 8, "is_keeper": False},
        # Would mismatch the flat formula (expects 6), but 2023 predates
        # rule_start_season (2026) — must not be validated or flagged.
        {"season": 2023, "owner_id": "u1", "player_id": "p1", "round": 5, "is_keeper": True},
    ])
    ledger = build_keeper_ledger(df, RULES).sort_values("season")
    assert pd.isna(ledger.iloc[-1]["expected_round_formula"])
    assert ledger.iloc[-1]["formula_mismatch"] == False  # noqa: E712


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
