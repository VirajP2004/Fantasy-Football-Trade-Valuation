"""
Unit tests for the trade engine's final integration point, `evaluate_trade`.

`evaluate_player_fn`/`explain_player_fn` are injected with synthetic
fakes throughout -- no real model loading, no SHAP, no nflreadpy calls --
so what's actually under test is the ORCHESTRATION: per-player
aggregation, positional-need wiring, fairness scoring, top-mover
selection, explanation attachment, and caveat rollup. Same standard as
the rest of tests/: pure logic verified on synthetic inputs;
`notebooks/10_evaluate_trade_demo.ipynb` exercises the real, live path
end to end.
"""

import pandas as pd
import pytest

from src.explain.explain_player import FeatureDriver, PlayerExplanation
from src.trade_engine.evaluate_trade import TradeAsset, evaluate_trade
from src.trade_engine.net_value import CAVEAT_TEXT, NetValueResult
from src.trade_engine.positional_need import CAVEAT_TEXT as NEED_CAVEAT_TEXT


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------

def _vorp_labels_fixture():
    """RB std = 20.0 (values 10, 30, 50); WR std = 20.0 (values 20, 40, 60)."""
    return pd.DataFrame([
        {"position": "RB", "season": 2025, "vorp": 10.0},
        {"position": "RB", "season": 2025, "vorp": 30.0},
        {"position": "RB", "season": 2025, "vorp": 50.0},
        {"position": "WR", "season": 2025, "vorp": 20.0},
        {"position": "WR", "season": 2025, "vorp": 40.0},
        {"position": "WR", "season": 2025, "vorp": 60.0},
    ])


def _make_fake_evaluate_player_fn(net_values_by_player, caveats_by_player=None):
    caveats_by_player = caveats_by_player or {}

    def fake(player_name, season, position, projected_keeper_round, draft_curve, repo_root, vorp_labels=None):
        predicted_kvs, keeper_cost = net_values_by_player[player_name]
        return NetValueResult(
            player_name=player_name, season=season, position=position,
            predicted_kvs=predicted_kvs, keeper_cost_vorp=keeper_cost,
            net_kvs_delta=predicted_kvs - keeper_cost,
            low_confidence_extreme_delta=False, no_delta_history=False,
            caveats=list(caveats_by_player.get(player_name, [])),
        )
    return fake


def _make_fake_explain_player_fn(should_fail=frozenset()):
    def fake(player_name, season, position, repo_root, top_n=5, vorp_labels=None):
        if player_name in should_fail:
            raise ValueError(f"no feature row for {player_name}")
        return PlayerExplanation(
            player_name=player_name, season=season, position=position,
            predicted_vorp=42.0, base_value=10.0,
            top_drivers=[FeatureDriver(feature="scarcity_z", feature_value=1.0, shap_value=5.0, direction="increases")],
        )
    return fake


_DUMMY_CURVE = object()
_DUMMY_REPO_ROOT = "."


# ---------------------------------------------------------------------------
# Basic wiring: per-player values, totals, fairness
# ---------------------------------------------------------------------------

def test_evaluate_trade_computes_totals_and_fairness():
    team_a_players = [TradeAsset("RB One", 2025, "RB", projected_keeper_round=5)]
    team_b_players = [TradeAsset("WR One", 2025, "WR", projected_keeper_round=5)]
    # Average roster (need_z == 0.0) so the adjustment is a no-op and the
    # totals are exactly the raw net_kvs_delta values.
    team_a_roster = [{"position": "RB", "scarcity_z": 0.0}]
    team_b_roster = [{"position": "WR", "scarcity_z": 0.0}]

    fake_eval = _make_fake_evaluate_player_fn({"RB One": (80.0, 30.0), "WR One": (40.0, 30.0)})
    fake_explain = _make_fake_explain_player_fn()

    result = evaluate_trade(
        team_a_players, team_b_players, team_a_roster, team_b_roster,
        draft_curve=_DUMMY_CURVE, repo_root=_DUMMY_REPO_ROOT, vorp_labels=_vorp_labels_fixture(),
        evaluate_player_fn=fake_eval, explain_player_fn=fake_explain,
    )

    assert result.team_a_total == pytest.approx(50.0)  # 80 - 30
    assert result.team_b_total == pytest.approx(10.0)  # 40 - 30
    assert result.fairness.team_a_share_pct == pytest.approx(100.0 * 50 / 60)
    assert result.fairness.team_b_share_pct == pytest.approx(100.0 * 10 / 60)


def test_positional_need_adjustment_is_applied_against_receiving_teams_roster():
    """team_a_players moves TO team A, so it must be adjusted against
    team_a_roster (deep at RB here), not team_b_roster."""
    team_a_players = [TradeAsset("RB One", 2025, "RB", projected_keeper_round=5)]
    team_b_players = []
    team_a_roster = [
        {"position": "RB", "scarcity_z": 2.0},
        {"position": "RB", "scarcity_z": 2.0},
    ]  # deep at RB -> discount
    team_b_roster = []

    fake_eval = _make_fake_evaluate_player_fn({"RB One": (80.0, 30.0)})
    result = evaluate_trade(
        team_a_players, team_b_players, team_a_roster, team_b_roster,
        draft_curve=_DUMMY_CURVE, repo_root=_DUMMY_REPO_ROOT, vorp_labels=_vorp_labels_fixture(),
        evaluate_player_fn=fake_eval, explain_player_fn=_make_fake_explain_player_fn(),
    )
    contribution = result.team_a_players[0]
    assert contribution.raw_net_kvs_delta == pytest.approx(50.0)
    assert contribution.team_need_z == pytest.approx(2.0)
    assert contribution.adjustment_vorp < 0  # discounted for being deep at RB
    assert contribution.adjusted_net_kvs_delta < contribution.raw_net_kvs_delta
    assert result.team_a_total == pytest.approx(contribution.adjusted_net_kvs_delta)


# ---------------------------------------------------------------------------
# Top-mover selection and explanation attachment
# ---------------------------------------------------------------------------

def test_top_movers_are_selected_by_absolute_adjusted_value_not_raw_positivity():
    """A big loss should count as a "top mover" just as much as a big
    gain -- selection is by |adjusted_net_kvs_delta|, not by which
    player has the largest positive number."""
    team_a_players = [
        TradeAsset("Small Gain", 2025, "RB", projected_keeper_round=5),
        TradeAsset("Big Loss", 2025, "RB", projected_keeper_round=5),
    ]
    team_a_roster = [{"position": "RB", "scarcity_z": 0.0}]  # no-op adjustment
    fake_eval = _make_fake_evaluate_player_fn({
        "Small Gain": (35.0, 30.0),   # +5
        "Big Loss": (10.0, 90.0),     # -80
    })

    result = evaluate_trade(
        team_a_players, [], team_a_roster, [],
        draft_curve=_DUMMY_CURVE, repo_root=_DUMMY_REPO_ROOT, vorp_labels=_vorp_labels_fixture(),
        evaluate_player_fn=fake_eval, explain_player_fn=_make_fake_explain_player_fn(), top_n_explained=1,
    )
    assert len(result.team_a_explained_drivers) == 1
    assert result.team_a_explained_drivers[0].player_name == "Big Loss"


def test_top_n_explained_is_capped_by_available_players():
    team_a_players = [TradeAsset("Only Player", 2025, "RB", projected_keeper_round=5)]
    fake_eval = _make_fake_evaluate_player_fn({"Only Player": (50.0, 30.0)})

    result = evaluate_trade(
        team_a_players, [], [{"position": "RB", "scarcity_z": 0.0}], [],
        draft_curve=_DUMMY_CURVE, repo_root=_DUMMY_REPO_ROOT, vorp_labels=_vorp_labels_fixture(),
        evaluate_player_fn=fake_eval, explain_player_fn=_make_fake_explain_player_fn(), top_n_explained=2,
    )
    assert len(result.team_a_explained_drivers) == 1


def test_explained_driver_carries_real_top_drivers_from_explain_player():
    team_a_players = [TradeAsset("RB One", 2025, "RB", projected_keeper_round=5)]
    fake_eval = _make_fake_evaluate_player_fn({"RB One": (80.0, 30.0)})

    result = evaluate_trade(
        team_a_players, [], [{"position": "RB", "scarcity_z": 0.0}], [],
        draft_curve=_DUMMY_CURVE, repo_root=_DUMMY_REPO_ROOT, vorp_labels=_vorp_labels_fixture(),
        evaluate_player_fn=fake_eval, explain_player_fn=_make_fake_explain_player_fn(),
    )
    driver = result.team_a_explained_drivers[0]
    assert driver.explanation_error is None
    assert len(driver.top_drivers) == 1
    assert driver.top_drivers[0].feature == "scarcity_z"


def test_explanation_failure_is_captured_not_raised():
    """A SHAP lookup failure for a top-mover must not blow up the whole
    trade evaluation -- the fairness score is already fully computed by
    the time explanations run."""
    team_a_players = [TradeAsset("Mystery Player", 2025, "RB", projected_keeper_round=5)]
    fake_eval = _make_fake_evaluate_player_fn({"Mystery Player": (80.0, 30.0)})
    fake_explain = _make_fake_explain_player_fn(should_fail={"Mystery Player"})

    result = evaluate_trade(
        team_a_players, [], [{"position": "RB", "scarcity_z": 0.0}], [],
        draft_curve=_DUMMY_CURVE, repo_root=_DUMMY_REPO_ROOT, vorp_labels=_vorp_labels_fixture(),
        evaluate_player_fn=fake_eval, explain_player_fn=fake_explain,
    )
    # The core verdict is unaffected:
    assert result.team_a_total == pytest.approx(50.0)
    assert result.fairness.team_a_share_pct == 100.0
    # The explanation failure is captured explicitly, not silently dropped:
    driver = result.team_a_explained_drivers[0]
    assert driver.top_drivers == []
    assert "Mystery Player" in driver.explanation_error


# ---------------------------------------------------------------------------
# Caveat rollup
# ---------------------------------------------------------------------------

def test_net_value_caveats_are_surfaced_and_attributed():
    team_a_players = [TradeAsset("Flagged Player", 2025, "RB", projected_keeper_round=5)]
    fake_eval = _make_fake_evaluate_player_fn(
        {"Flagged Player": (80.0, 30.0)},
        caveats_by_player={"Flagged Player": [CAVEAT_TEXT["low_confidence_extreme_delta"]]},
    )

    result = evaluate_trade(
        team_a_players, [], [{"position": "RB", "scarcity_z": 0.0}], [],
        draft_curve=_DUMMY_CURVE, repo_root=_DUMMY_REPO_ROOT, vorp_labels=_vorp_labels_fixture(),
        evaluate_player_fn=fake_eval, explain_player_fn=_make_fake_explain_player_fn(),
    )
    assert CAVEAT_TEXT["low_confidence_extreme_delta"] in result.team_a_players[0].caveats
    matching = [c for c in result.caveats if c.player_name == "Flagged Player"]
    assert len(matching) == 1
    assert matching[0].side == "team_a"
    assert matching[0].caveat == CAVEAT_TEXT["low_confidence_extreme_delta"]


def test_empty_position_caveat_from_positional_need_is_surfaced():
    """The receiving team has NO players at TE at all -- the
    empty_position caveat from positional_need.py must make it all the
    way to the final result's caveats list, not just live on the
    per-player contribution."""
    team_a_players = [TradeAsset("TE One", 2025, "TE", projected_keeper_round=8)]
    fake_eval = _make_fake_evaluate_player_fn({"TE One": (40.0, 20.0)})

    result = evaluate_trade(
        team_a_players, [], team_a_roster=[], team_b_roster=[],
        draft_curve=_DUMMY_CURVE, repo_root=_DUMMY_REPO_ROOT,
        vorp_labels=pd.concat([_vorp_labels_fixture(), pd.DataFrame([
            {"position": "TE", "season": 2025, "vorp": 5.0},
            {"position": "TE", "season": 2025, "vorp": 15.0},
        ])], ignore_index=True),
        evaluate_player_fn=fake_eval, explain_player_fn=_make_fake_explain_player_fn(),
    )
    caveat_texts = [c.caveat for c in result.caveats if c.player_name == "TE One"]
    assert NEED_CAVEAT_TEXT["empty_position"] in caveat_texts


def test_no_caveats_produces_empty_list_not_none():
    team_a_players = [TradeAsset("Clean Player", 2025, "RB", projected_keeper_round=5)]
    fake_eval = _make_fake_evaluate_player_fn({"Clean Player": (80.0, 30.0)})

    result = evaluate_trade(
        team_a_players, [], [{"position": "RB", "scarcity_z": 0.0}], [],
        draft_curve=_DUMMY_CURVE, repo_root=_DUMMY_REPO_ROOT, vorp_labels=_vorp_labels_fixture(),
        evaluate_player_fn=fake_eval, explain_player_fn=_make_fake_explain_player_fn(),
    )
    assert result.caveats == []
    assert result.team_a_players[0].caveats == []


# ---------------------------------------------------------------------------
# Empty sides
# ---------------------------------------------------------------------------

def test_one_sided_trade_gives_the_empty_side_zero_total_and_no_explanations():
    team_a_players = [TradeAsset("RB One", 2025, "RB", projected_keeper_round=5)]
    fake_eval = _make_fake_evaluate_player_fn({"RB One": (80.0, 30.0)})

    result = evaluate_trade(
        team_a_players, [], [{"position": "RB", "scarcity_z": 0.0}], [],
        draft_curve=_DUMMY_CURVE, repo_root=_DUMMY_REPO_ROOT, vorp_labels=_vorp_labels_fixture(),
        evaluate_player_fn=fake_eval, explain_player_fn=_make_fake_explain_player_fn(),
    )
    assert result.team_b_total == 0.0
    assert result.team_b_players == []
    assert result.team_b_explained_drivers == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
