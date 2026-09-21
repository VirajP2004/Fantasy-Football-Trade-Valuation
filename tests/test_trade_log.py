"""
Unit tests for `trade_log.py`'s pure row-building logic
(`build_trade_log_row`) only -- no SQLite, no disk I/O. Same standard as
tests/test_evaluate_trade.py: real dataclasses from `evaluate_trade.py`/
`fairness_score.py` constructed directly as fixtures, nothing mocked,
since they're cheap plain dataclasses with no live dependency of their
own.
"""

import json
import uuid
from datetime import datetime

from src.trade_engine.evaluate_trade import AttributedCaveat, PlayerTradeContribution, TradeEvaluationResult
from src.trade_engine.fairness_score import FAIRNESS_METHOD_PERCENTAGE_OF_TOTAL, FairnessScoreResult
from src.trade_engine.trade_log import build_trade_log_row


def _make_contribution(player_name: str, adjusted: float, caveats=None) -> PlayerTradeContribution:
    return PlayerTradeContribution(
        player_name=player_name, season=2025, position="RB",
        predicted_kvs=80.0, keeper_cost_vorp=30.0, raw_net_kvs_delta=50.0,
        team_need_z=0.0, adjustment_vorp=0.0, adjusted_net_kvs_delta=adjusted,
        caveats=list(caveats or []),
    )


def _make_result(team_a_players, team_b_players, caveats=None) -> TradeEvaluationResult:
    team_a_total = sum(p.adjusted_net_kvs_delta for p in team_a_players)
    team_b_total = sum(p.adjusted_net_kvs_delta for p in team_b_players)
    fairness = FairnessScoreResult(
        team_a_value=team_a_total, team_b_value=team_b_total,
        team_a_share_pct=62.0, team_b_share_pct=38.0,
        shifted=False, fairness_method=FAIRNESS_METHOD_PERCENTAGE_OF_TOTAL,
    )
    return TradeEvaluationResult(
        team_a_players=team_a_players, team_b_players=team_b_players,
        team_a_total=team_a_total, team_b_total=team_b_total,
        fairness=fairness, team_a_explained_drivers=[], team_b_explained_drivers=[],
        caveats=list(caveats or []),
    )


def test_build_trade_log_row_produces_expected_flat_row():
    result = _make_result(
        [_make_contribution("RB One", 50.0)],
        [_make_contribution("WR One", 10.0)],
    )
    row = build_trade_log_row(
        "Team A", "Team B", result, "2026-v1",
        trade_id="fixed-id", timestamp="2026-09-21T00:00:00+00:00",
    )

    assert row["trade_id"] == "fixed-id"
    assert row["timestamp"] == "2026-09-21T00:00:00+00:00"
    assert row["team_a"] == "Team A"
    assert row["team_b"] == "Team B"
    assert row["fairness_score"] == 62.0
    assert row["fairness_method"] == FAIRNESS_METHOD_PERCENTAGE_OF_TOTAL
    assert row["net_kvs_delta_a"] == 50.0
    assert row["net_kvs_delta_b"] == 10.0
    assert row["model_version"] == "2026-v1"


def test_team_players_serialize_as_json_list_of_names():
    result = _make_result(
        [_make_contribution("RB One", 50.0), _make_contribution("RB Two", 5.0)],
        [_make_contribution("WR One", 10.0)],
    )
    row = build_trade_log_row("Team A", "Team B", result, "2026-v1")

    assert json.loads(row["team_a_players"]) == ["RB One", "RB Two"]
    assert json.loads(row["team_b_players"]) == ["WR One"]


def test_caveats_serialize_with_side_and_player_attribution():
    caveats = [AttributedCaveat(side="team_a", player_name="RB One", caveat="low_confidence_extreme_delta: ...")]
    result = _make_result([_make_contribution("RB One", 50.0)], [], caveats=caveats)
    row = build_trade_log_row("Team A", "Team B", result, "2026-v1")

    decoded = json.loads(row["caveats"])
    assert decoded == [{"side": "team_a", "player_name": "RB One", "caveat": "low_confidence_extreme_delta: ..."}]


def test_no_caveats_serializes_as_empty_list_not_null():
    result = _make_result([_make_contribution("RB One", 50.0)], [])
    row = build_trade_log_row("Team A", "Team B", result, "2026-v1")

    assert json.loads(row["caveats"]) == []


def test_trade_id_and_timestamp_are_generated_when_omitted():
    result = _make_result([_make_contribution("RB One", 50.0)], [])
    row = build_trade_log_row("Team A", "Team B", result, "2026-v1")

    # Does not raise -- a real uuid4 and a real ISO-8601 timestamp.
    uuid.UUID(row["trade_id"])
    datetime.fromisoformat(row["timestamp"])


def test_two_calls_without_explicit_ids_generate_distinct_trade_ids():
    result = _make_result([_make_contribution("RB One", 50.0)], [])
    row1 = build_trade_log_row("Team A", "Team B", result, "2026-v1")
    row2 = build_trade_log_row("Team A", "Team B", result, "2026-v1")

    assert row1["trade_id"] != row2["trade_id"]
