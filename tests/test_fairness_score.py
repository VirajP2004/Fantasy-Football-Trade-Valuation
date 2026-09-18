"""
Unit tests for the trade engine's fairness score normalization. Runs
entirely on synthetic inputs -- same standard as
tests/test_positional_need.py and tests/test_net_value.py.
"""

import pytest

from src.trade_engine.fairness_score import (
    FAIRNESS_METHOD_MAGNITUDE_RELATIVE_BOTH_NEGATIVE,
    FAIRNESS_METHOD_PERCENTAGE_OF_TOTAL,
    compute_fairness_score,
    describe_fairness,
    fairness_score_for_trade,
)


# ---------------------------------------------------------------------------
# compute_fairness_score
# ---------------------------------------------------------------------------

def test_equal_values_split_fifty_fifty():
    result = compute_fairness_score(50.0, 50.0)
    assert result.team_a_share_pct == 50.0
    assert result.team_b_share_pct == 50.0
    assert result.shifted is False
    assert result.fairness_method == FAIRNESS_METHOD_PERCENTAGE_OF_TOTAL


def test_unequal_positive_values_produce_correct_percentages():
    """The example from this module's own docstring: 62/38."""
    result = compute_fairness_score(62.0, 38.0)
    assert result.team_a_share_pct == pytest.approx(62.0)
    assert result.team_b_share_pct == pytest.approx(38.0)
    assert result.shifted is False


def test_shares_always_sum_to_100():
    result = compute_fairness_score(17.0, 83.0)
    assert result.team_a_share_pct + result.team_b_share_pct == pytest.approx(100.0)


def test_one_side_negative_gets_shifted_not_broken():
    """value_a=-10, value_b=30 would be -50%/150% unshifted -- nonsensical.
    Shifting both by 10 gives 0 and 40, i.e. 0%/100%, preserving the
    entire gap going to team B while staying inside [0, 100]. Only one
    side is negative here, so the percentage-of-total path (not the
    both-negative fallback) is the one under test."""
    result = compute_fairness_score(-10.0, 30.0)
    assert result.shifted is True
    assert result.team_a_share_pct == pytest.approx(0.0)
    assert result.team_b_share_pct == pytest.approx(100.0)
    assert result.fairness_method == FAIRNESS_METHOD_PERCENTAGE_OF_TOTAL


def test_both_sides_negative_still_produces_valid_percentages():
    result = compute_fairness_score(-20.0, -5.0)
    assert 0.0 <= result.team_a_share_pct <= 100.0
    assert 0.0 <= result.team_b_share_pct <= 100.0
    assert result.team_b_share_pct > result.team_a_share_pct  # -5 is the "less bad" side
    assert result.fairness_method == FAIRNESS_METHOD_MAGNITUDE_RELATIVE_BOTH_NEGATIVE


def test_both_sides_negative_and_nearly_equal_does_not_produce_an_extreme_split():
    """Regression test for a real bug found by directly checking a
    suspected edge case: -100 vs -99 (a ~1% relative difference in loss
    magnitude, about as close to a fair-if-bad trade as two numbers can
    be) used to come back as a misleading 0%/100% under the old
    shift-then-percentage-of-total math, because shifting the floor to 0
    turned the tiny 1-point gap into the ENTIRE denominator. The
    both-negative fallback must keep this close to even instead."""
    result = compute_fairness_score(-100.0, -99.0)
    assert result.fairness_method == FAIRNESS_METHOD_MAGNITUDE_RELATIVE_BOTH_NEGATIVE
    assert result.team_a_share_pct == pytest.approx(49.7487, abs=1e-3)
    assert result.team_b_share_pct == pytest.approx(50.2513, abs=1e-3)
    # The whole point: nowhere near the old 0%/100%.
    assert abs(result.team_a_share_pct - 50.0) < 5.0
    assert abs(result.team_b_share_pct - 50.0) < 5.0


def test_both_sides_zero_is_an_even_split_not_a_crash():
    result = compute_fairness_score(0.0, 0.0)
    assert result.team_a_share_pct == 50.0
    assert result.team_b_share_pct == 50.0
    # 0.0 is not < 0, so this must take the ordinary percentage-of-total
    # path (and its total==0 branch), never the both-negative fallback.
    assert result.fairness_method == FAIRNESS_METHOD_PERCENTAGE_OF_TOTAL


def test_shifted_flag_false_when_no_shift_needed():
    result = compute_fairness_score(5.0, 5.0)
    assert result.shifted is False


def test_both_negative_fallback_never_sets_shifted():
    """`shifted` describes the percentage-of-total shift convention
    specifically -- the both-negative fallback doesn't shift at all, so
    it must always report shifted=False, not carry over a stale True."""
    result = compute_fairness_score(-100.0, -99.0)
    assert result.shifted is False


# ---------------------------------------------------------------------------
# fairness_score_for_trade
# ---------------------------------------------------------------------------

def test_fairness_score_for_trade_sums_each_side():
    result = fairness_score_for_trade(team_a_received=[10.0, 20.0], team_b_received=[15.0])
    assert result.team_a_value == 30.0
    assert result.team_b_value == 15.0
    assert result.team_a_share_pct == pytest.approx(200.0 / 3)


def test_fairness_score_for_trade_handles_empty_side():
    result = fairness_score_for_trade(team_a_received=[], team_b_received=[10.0])
    assert result.team_a_value == 0.0
    assert result.team_b_share_pct == 100.0


# ---------------------------------------------------------------------------
# describe_fairness
# ---------------------------------------------------------------------------

def test_describe_fairness_matches_docstring_example():
    result = compute_fairness_score(62.0, 38.0)
    assert describe_fairness(result) == "Team A receives 62% of the trade's total value."


def test_describe_fairness_uses_custom_team_name():
    result = compute_fairness_score(25.0, 75.0)
    assert describe_fairness(result, team_a_name="The Sharks") == "The Sharks receives 25% of the trade's total value."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
