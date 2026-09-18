"""
Unit tests for the trade engine's positional need adjustment. Runs
entirely on synthetic inputs -- same standard as tests/test_net_value.py
and tests/test_draft_capital_curve.py: the pure math is verifiable
without live data.
"""

import pytest

from src.trade_engine.positional_need import (
    CAVEAT_TEXT,
    EMPTY_POSITION_NEED_Z,
    MAX_ABS_NEED_Z,
    apply_positional_need_adjustment,
    positional_need_adjustment_vorp,
    team_positional_need_z,
)


# ---------------------------------------------------------------------------
# team_positional_need_z
# ---------------------------------------------------------------------------

def test_team_positional_need_z_averages_scarcity_z_at_position():
    roster = [
        {"position": "RB", "scarcity_z": 1.0},
        {"position": "RB", "scarcity_z": 3.0},
        {"position": "WR", "scarcity_z": -5.0},  # different position, ignored
    ]
    need_z, count = team_positional_need_z(roster, "RB")
    assert need_z == 2.0
    assert count == 2


def test_team_positional_need_z_empty_position_uses_sentinel():
    """Confirms zero players at a position is NOT silently read as
    average need (0.0 z) -- 'no data for this position on this roster'
    is a different, honest answer from 'we checked and it's average',
    and must not collapse into the same neutral-looking number."""
    roster = [{"position": "WR", "scarcity_z": 1.0}]
    need_z, count = team_positional_need_z(roster, "TE")
    assert need_z == EMPTY_POSITION_NEED_Z
    assert need_z == -2.0
    assert need_z != 0.0
    assert count == 0


def test_team_positional_need_z_empty_roster_uses_sentinel():
    need_z, count = team_positional_need_z([], "QB")
    assert need_z == EMPTY_POSITION_NEED_Z
    assert count == 0


def test_team_positional_need_z_clamps_extreme_positive():
    roster = [{"position": "TE", "scarcity_z": 50.0}]
    need_z, _ = team_positional_need_z(roster, "TE")
    assert need_z == MAX_ABS_NEED_Z


def test_team_positional_need_z_clamps_extreme_negative():
    roster = [{"position": "TE", "scarcity_z": -50.0}]
    need_z, _ = team_positional_need_z(roster, "TE")
    assert need_z == -MAX_ABS_NEED_Z


# ---------------------------------------------------------------------------
# positional_need_adjustment_vorp
# ---------------------------------------------------------------------------

def test_adjustment_vorp_zero_need_is_zero():
    assert positional_need_adjustment_vorp(0.0, position_std_vorp_that_season=40.0) == 0.0


def test_adjustment_vorp_positive_need_z_is_negative_adjustment():
    """A team already deep at the position (positive need_z) gets a
    discount (negative adjustment)."""
    adjustment = positional_need_adjustment_vorp(2.0, position_std_vorp_that_season=40.0, weight=0.25)
    assert adjustment == pytest.approx(-0.25 * 2.0 * 40.0)
    assert adjustment < 0


def test_adjustment_vorp_negative_need_z_is_positive_adjustment():
    """A thin team (negative need_z) gets a boost (positive adjustment)."""
    adjustment = positional_need_adjustment_vorp(-2.0, position_std_vorp_that_season=40.0, weight=0.25)
    assert adjustment == pytest.approx(0.25 * 2.0 * 40.0)
    assert adjustment > 0


def test_adjustment_vorp_scales_with_position_std():
    """The SAME need_z produces a bigger swing for a position with more
    real talent spread (higher position_std_vorp_that_season) -- this is
    the whole point of grounding the adjustment in scarcity_z's own std,
    rather than a flat constant applied identically to every position."""
    low_spread = positional_need_adjustment_vorp(-1.0, position_std_vorp_that_season=10.0)
    high_spread = positional_need_adjustment_vorp(-1.0, position_std_vorp_that_season=80.0)
    assert abs(high_spread) > abs(low_spread)


def test_adjustment_vorp_weight_zero_disables_adjustment():
    adjustment = positional_need_adjustment_vorp(3.0, position_std_vorp_that_season=40.0, weight=0.0)
    assert adjustment == 0.0


# ---------------------------------------------------------------------------
# apply_positional_need_adjustment
# ---------------------------------------------------------------------------

def test_apply_average_need_leaves_raw_delta_unchanged():
    roster = [{"position": "WR", "scarcity_z": 0.0}]
    result = apply_positional_need_adjustment(50.0, roster, "WR", position_std_vorp_that_season=30.0)
    assert result.team_need_z == 0.0
    assert result.adjustment_vorp == 0.0
    assert result.adjusted_net_kvs_delta == 50.0
    assert result.caveats == []


def test_apply_deep_roster_discounts_positive_raw_delta():
    roster = [
        {"position": "RB", "scarcity_z": 1.5},
        {"position": "RB", "scarcity_z": 2.5},
    ]
    result = apply_positional_need_adjustment(50.0, roster, "RB", position_std_vorp_that_season=30.0)
    assert result.team_need_z == 2.0
    assert result.adjustment_vorp < 0
    assert result.adjusted_net_kvs_delta < 50.0


def test_apply_thin_roster_boosts_positive_raw_delta():
    roster = [{"position": "TE", "scarcity_z": -3.0}]
    result = apply_positional_need_adjustment(20.0, roster, "TE", position_std_vorp_that_season=25.0)
    assert result.team_need_z < 0
    assert result.adjustment_vorp > 0
    assert result.adjusted_net_kvs_delta > 20.0


def test_apply_thin_need_can_turn_a_negative_delta_less_negative():
    """This is exactly why the adjustment is additive (VORP points), not
    a multiplier: a needed-but-mediocre trade should move TOWARD
    fairness, not get pushed further negative by a naive multiplicative
    boost on a negative number."""
    roster = [{"position": "QB", "scarcity_z": -3.0}]
    result = apply_positional_need_adjustment(-10.0, roster, "QB", position_std_vorp_that_season=20.0)
    assert result.adjusted_net_kvs_delta > -10.0


def test_apply_empty_position_uses_sentinel_and_flags_caveat():
    result = apply_positional_need_adjustment(10.0, roster=[], position="K", position_std_vorp_that_season=15.0)
    assert result.roster_player_count_at_position == 0
    assert result.team_need_z == EMPTY_POSITION_NEED_Z
    assert result.adjustment_vorp > 0
    assert CAVEAT_TEXT["empty_position"] in result.caveats


def test_apply_weight_zero_disables_adjustment_end_to_end():
    roster = [{"position": "RB", "scarcity_z": 3.0}]
    result = apply_positional_need_adjustment(40.0, roster, "RB", position_std_vorp_that_season=30.0, weight=0.0)
    assert result.adjusted_net_kvs_delta == 40.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
