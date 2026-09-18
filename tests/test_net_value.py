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

import pandas as pd
import pytest

from src.trade_engine.draft_capital_curve import build_draft_capital_curve
from src.trade_engine.net_value import (
    CAVEAT_TEXT,
    UnsupportedPositionError,
    compute_flags,
    evaluate_net_value,
    evaluate_player_trade_value,
    net_value_from_projected_round,
    position_std_vorp_that_season,
    predict_kvs,
)


# ---------------------------------------------------------------------------
# position_std_vorp_that_season
# ---------------------------------------------------------------------------

def _vorp_labels_fixture():
    return pd.DataFrame([
        {"position": "RB", "season": 2025, "vorp": 10.0},
        {"position": "RB", "season": 2025, "vorp": 30.0},
        {"position": "RB", "season": 2025, "vorp": 50.0},
        {"position": "RB", "season": 2024, "vorp": 1000.0},  # different season, must be ignored
        {"position": "WR", "season": 2025, "vorp": 999.0},  # different position, must be ignored
    ])


def test_position_std_vorp_that_season_matches_manual_std():
    result = position_std_vorp_that_season(_vorp_labels_fixture(), "RB", 2025)
    assert result == pytest.approx(20.0)  # std of [10, 30, 50]


def test_position_std_vorp_that_season_raises_for_no_matching_rows():
    with pytest.raises(ValueError):
        position_std_vorp_that_season(_vorp_labels_fixture(), "TE", 2025)


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


def test_both_caveats_can_coexist():
    result = evaluate_net_value(
        "Test Player", 2026, "QB", predicted_kvs=10.0, keeper_cost_vorp_value=5.0,
        low_confidence_extreme_delta=True, no_delta_history=True,
    )
    assert len(result.caveats) == 2


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
# K/DEF are out of scope: predict_kvs and evaluate_player_trade_value must
# raise UnsupportedPositionError, never return a number of any kind (no
# heuristic fallback exists in this module any more).
# ---------------------------------------------------------------------------

def test_predict_kvs_raises_for_k():
    with pytest.raises(UnsupportedPositionError):
        predict_kvs("Any Kicker", 2025, "K", repo_root=".")


def test_predict_kvs_raises_for_def():
    with pytest.raises(UnsupportedPositionError):
        predict_kvs("Any Defense", 2025, "DEF", repo_root=".")


def test_predict_kvs_k_error_references_scope_decision():
    """The error must actually explain itself, not just fail silently or
    with a generic message -- someone hitting this should be pointed at
    the reasoning, not left to guess why."""
    with pytest.raises(UnsupportedPositionError, match="06_scope_decision_k_def"):
        predict_kvs("Any Kicker", 2025, "K", repo_root=".")


def test_evaluate_player_trade_value_raises_for_k():
    with pytest.raises(UnsupportedPositionError):
        evaluate_player_trade_value("Any Kicker", 2025, "K", projected_keeper_round=10, draft_curve=_simple_curve(), repo_root=".")


def test_evaluate_player_trade_value_raises_for_def():
    with pytest.raises(UnsupportedPositionError):
        evaluate_player_trade_value("Any Defense", 2025, "DEF", projected_keeper_round=10, draft_curve=_simple_curve(), repo_root=".")


def test_unsupported_position_error_is_not_raised_for_modeled_positions():
    """Guard against a too-broad check accidentally catching real
    positions -- this should fail for a totally different reason (no real
    vorp_labels.parquet at this repo_root), never UnsupportedPositionError."""
    with pytest.raises(Exception) as exc_info:
        predict_kvs("Nonexistent Player", 2025, "QB", repo_root="/nonexistent/repo/root")
    assert not isinstance(exc_info.value, UnsupportedPositionError)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
