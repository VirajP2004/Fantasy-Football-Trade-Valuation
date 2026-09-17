"""
Unit tests for the trade engine's net value calculation. Runs entirely on
synthetic inputs -- no model loading, no nflreadpy calls. Same standard as
tests/test_keeper_ledger.py and tests/test_draft_capital_curve.py: the
pure math (evaluate_net_value / net_value_from_projected_round) is
verifiable without live data; only the live feature-building/prediction
path (predict_kvs, evaluate_player_trade_value) touches real models and
data, and is exercised for real in notebooks/09_trade_engine_demo.ipynb
instead.
"""

import math

import pandas as pd
import pytest

from src.trade_engine.draft_capital_curve import build_draft_capital_curve
from src.trade_engine.net_value import (
    CAVEAT_TEXT,
    RELIABILITY_TIER_HEURISTIC,
    RELIABILITY_TIER_MODEL,
    compute_flags,
    evaluate_net_value,
    group_by_reliability_tier,
    net_value_from_projected_round,
)


# ---------------------------------------------------------------------------
# compute_flags
# ---------------------------------------------------------------------------

def test_compute_flags_normal_delta():
    low_conf, no_hist = compute_flags(42.0)
    assert low_conf is False
    assert no_hist is False


def test_compute_flags_extreme_positive_delta():
    low_conf, no_hist = compute_flags(150.0)
    assert low_conf is True
    assert no_hist is False


def test_compute_flags_extreme_negative_delta():
    low_conf, no_hist = compute_flags(-101.0)
    assert low_conf is True
    assert no_hist is False


def test_compute_flags_exactly_at_threshold_not_flagged():
    """Matches every 06x_model_*.ipynb notebook's own `> 100`, not `>= 100`."""
    low_conf, _ = compute_flags(100.0)
    assert low_conf is False


def test_compute_flags_missing_delta_is_no_delta_history_not_low_confidence():
    """The NaN-as-False bug fixed across QB/RB/WR/TE tonight: a missing
    delta must NOT silently read as "not extreme" (False, False) -- it
    must be its own distinct, separate flag."""
    low_conf, no_hist = compute_flags(float("nan"))
    assert low_conf is False
    assert no_hist is True


# ---------------------------------------------------------------------------
# evaluate_net_value
# ---------------------------------------------------------------------------

def test_net_kvs_delta_is_predicted_minus_keeper_cost():
    result = evaluate_net_value("Test Player", 2026, "RB", predicted_kvs=80.0, keeper_cost_vorp_value=30.0)
    assert result.net_kvs_delta == 50.0
    assert result.predicted_kvs == 80.0
    assert result.keeper_cost_vorp == 30.0


def test_net_kvs_delta_can_be_negative():
    result = evaluate_net_value("Test Player", 2026, "RB", predicted_kvs=10.0, keeper_cost_vorp_value=40.0)
    assert result.net_kvs_delta == -30.0


def test_no_caveats_when_no_flags_set():
    result = evaluate_net_value("Test Player", 2026, "RB", predicted_kvs=10.0, keeper_cost_vorp_value=5.0)
    assert result.caveats == []
    assert result.low_confidence_extreme_delta is False
    assert result.no_delta_history is False
    assert result.used_heuristic is False


def test_low_confidence_extreme_delta_caveat_never_silently_dropped():
    result = evaluate_net_value(
        "Test Player", 2026, "QB", predicted_kvs=10.0, keeper_cost_vorp_value=5.0,
        low_confidence_extreme_delta=True,
    )
    assert result.low_confidence_extreme_delta is True
    assert CAVEAT_TEXT["low_confidence_extreme_delta"] in result.caveats


def test_no_delta_history_caveat_never_silently_dropped():
    result = evaluate_net_value(
        "Test Player", 2026, "TE", predicted_kvs=10.0, keeper_cost_vorp_value=5.0,
        no_delta_history=True,
    )
    assert result.no_delta_history is True
    assert CAVEAT_TEXT["no_delta_history"] in result.caveats


def test_used_heuristic_caveat_never_silently_dropped():
    result = evaluate_net_value(
        "Test Kicker", 2026, "K", predicted_kvs=10.0, keeper_cost_vorp_value=5.0,
        used_heuristic=True,
    )
    assert result.used_heuristic is True
    assert CAVEAT_TEXT["used_heuristic"] in result.caveats


def test_all_three_caveats_can_coexist():
    """Nothing about these three flags is mutually exclusive -- a K/DEF
    heuristic prediction with a missing delta history must carry BOTH
    caveats, not just one."""
    result = evaluate_net_value(
        "Test Kicker", 2026, "K", predicted_kvs=10.0, keeper_cost_vorp_value=5.0,
        no_delta_history=True, used_heuristic=True,
    )
    assert len(result.caveats) == 2
    assert result.low_confidence_extreme_delta is False


# ---------------------------------------------------------------------------
# net_value_from_projected_round (integration with the real draft curve)
# ---------------------------------------------------------------------------

def _simple_curve():
    picks = pd.DataFrame(
        [(1, 100.0), (1, 100.0), (5, 10.0), (5, 10.0), (10, -20.0), (10, -20.0)],
        columns=["round", "vorp"],
    )
    return build_draft_capital_curve(picks, min_picks_per_round=2)


def test_net_value_from_projected_round_converts_round_to_vorp():
    curve = _simple_curve()
    result = net_value_from_projected_round("Test Player", 2026, "WR", predicted_kvs=50.0, projected_keeper_round=5, draft_curve=curve)
    assert result.keeper_cost_vorp == 10.0
    assert result.net_kvs_delta == 40.0


def test_net_value_from_projected_round_falls_back_for_unseen_round():
    curve = _simple_curve()
    result = net_value_from_projected_round("Test Player", 2026, "WR", predicted_kvs=50.0, projected_keeper_round=99, draft_curve=curve)
    assert result.keeper_cost_vorp == -20.0  # worst observed band, per keeper_cost_vorp's fallback


# ---------------------------------------------------------------------------
# reliability_tier -- must be a first-class, queryable/filterable field, not
# just a string buried in the caveats list.
# ---------------------------------------------------------------------------

def test_reliability_tier_is_model_when_not_heuristic():
    result = evaluate_net_value("Test Player", 2026, "RB", predicted_kvs=10.0, keeper_cost_vorp_value=5.0)
    assert result.reliability_tier == RELIABILITY_TIER_MODEL


def test_reliability_tier_is_heuristic_when_used_heuristic_true():
    result = evaluate_net_value(
        "Test Kicker", 2026, "K", predicted_kvs=10.0, keeper_cost_vorp_value=5.0, used_heuristic=True
    )
    assert result.reliability_tier == RELIABILITY_TIER_HEURISTIC


def test_reliability_tier_is_directly_filterable_without_parsing_caveats():
    """The whole point of this field: a caller must be able to filter/query
    it directly (`r.reliability_tier == ...`), not grep the free-text
    `caveats` list for a substring."""
    results = [
        evaluate_net_value("Model Player", 2026, "WR", predicted_kvs=50.0, keeper_cost_vorp_value=10.0),
        evaluate_net_value("Heuristic Kicker", 2026, "K", predicted_kvs=20.0, keeper_cost_vorp_value=5.0, used_heuristic=True),
    ]
    model_only = [r for r in results if r.reliability_tier == RELIABILITY_TIER_MODEL]
    heuristic_only = [r for r in results if r.reliability_tier == RELIABILITY_TIER_HEURISTIC]
    assert [r.player_name for r in model_only] == ["Model Player"]
    assert [r.player_name for r in heuristic_only] == ["Heuristic Kicker"]


# ---------------------------------------------------------------------------
# group_by_reliability_tier -- tiers must never blend into one ranked list.
# ---------------------------------------------------------------------------

def test_group_by_reliability_tier_separates_model_and_heuristic():
    results = [
        evaluate_net_value("Model A", 2026, "RB", predicted_kvs=100.0, keeper_cost_vorp_value=0.0),
        evaluate_net_value("Heuristic A", 2026, "DEF", predicted_kvs=200.0, keeper_cost_vorp_value=0.0, used_heuristic=True),
        evaluate_net_value("Model B", 2026, "WR", predicted_kvs=50.0, keeper_cost_vorp_value=0.0),
    ]
    tiers = group_by_reliability_tier(results)
    assert set(tiers.keys()) == {RELIABILITY_TIER_MODEL, RELIABILITY_TIER_HEURISTIC}
    assert {r.player_name for r in tiers[RELIABILITY_TIER_MODEL]} == {"Model A", "Model B"}
    assert {r.player_name for r in tiers[RELIABILITY_TIER_HEURISTIC]} == {"Heuristic A"}


def test_group_by_reliability_tier_never_lets_heuristic_outrank_model_in_same_list():
    """The exact scenario that motivated this: a heuristic prediction with
    a huge raw net_kvs_delta (200) must NEVER appear ranked above, or even
    in the same list as, a model prediction with a smaller one (100) --
    they must live in two separate lists, full stop."""
    results = [
        evaluate_net_value("Model A", 2026, "RB", predicted_kvs=100.0, keeper_cost_vorp_value=0.0),
        evaluate_net_value("Heuristic A", 2026, "DEF", predicted_kvs=200.0, keeper_cost_vorp_value=0.0, used_heuristic=True),
    ]
    tiers = group_by_reliability_tier(results)
    model_names = [r.player_name for r in tiers[RELIABILITY_TIER_MODEL]]
    heuristic_names = [r.player_name for r in tiers[RELIABILITY_TIER_HEURISTIC]]
    assert "Heuristic A" not in model_names
    assert "Model A" not in heuristic_names


def test_group_by_reliability_tier_sorts_within_each_tier_descending():
    results = [
        evaluate_net_value("Model Low", 2026, "RB", predicted_kvs=10.0, keeper_cost_vorp_value=0.0),
        evaluate_net_value("Model High", 2026, "RB", predicted_kvs=90.0, keeper_cost_vorp_value=0.0),
        evaluate_net_value("Heuristic Low", 2026, "K", predicted_kvs=5.0, keeper_cost_vorp_value=0.0, used_heuristic=True),
        evaluate_net_value("Heuristic High", 2026, "K", predicted_kvs=40.0, keeper_cost_vorp_value=0.0, used_heuristic=True),
    ]
    tiers = group_by_reliability_tier(results)
    assert [r.player_name for r in tiers[RELIABILITY_TIER_MODEL]] == ["Model High", "Model Low"]
    assert [r.player_name for r in tiers[RELIABILITY_TIER_HEURISTIC]] == ["Heuristic High", "Heuristic Low"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
