"""
Unit tests for the trade engine's draft capital curve. Runs entirely on
synthetic data -- no Sleeper/nflreadpy calls, no parquet files on disk.
Same standard as tests/test_keeper_ledger.py: the curve's math and the
crosswalk's matching logic are pure functions of their inputs and should
be verifiable without live data.
"""

import pandas as pd
import pytest

from src.trade_engine.draft_capital_curve import (
    attach_realized_vorp,
    build_draft_capital_curve,
    build_name_position_crosswalk,
    keeper_cost_vorp,
    normalize_player_name,
    resolve_fresh_picks,
    resolve_vorp_key,
)


# ---------------------------------------------------------------------------
# normalize_player_name
# ---------------------------------------------------------------------------

def test_normalize_strips_suffix_and_punctuation():
    assert normalize_player_name("Marvin Harrison Jr.") == "marvin harrison"
    assert normalize_player_name("Marvin Harrison") == "marvin harrison"


def test_normalize_handles_apostrophe_and_hyphen():
    assert normalize_player_name("Ja'Marr Chase") == normalize_player_name("Jamarr Chase")
    assert normalize_player_name("Amon-Ra St. Brown") == normalize_player_name("Amon Ra St Brown")


def test_normalize_none_and_empty():
    assert normalize_player_name(None) is None
    assert normalize_player_name("") is None


# ---------------------------------------------------------------------------
# build_name_position_crosswalk
# ---------------------------------------------------------------------------

def test_crosswalk_maps_unique_name_position_pairs():
    nfl_players = pd.DataFrame([
        {"display_name": "Bo Nix", "position": "QB", "gsis_id": "00-1111111"},
        {"display_name": "Malik Nabers", "position": "WR", "gsis_id": "00-2222222"},
    ])
    crosswalk = build_name_position_crosswalk(nfl_players)
    assert crosswalk[("bo nix", "QB")] == "00-1111111"
    assert crosswalk[("malik nabers", "WR")] == "00-2222222"


def test_crosswalk_drops_genuine_name_collisions():
    """Two distinct real players sharing a (normalized name, position) --
    must be dropped entirely, not guessed at."""
    nfl_players = pd.DataFrame([
        {"display_name": "Kyle Williams", "position": "WR", "gsis_id": "00-1111111"},
        {"display_name": "Kyle Williams", "position": "WR", "gsis_id": "00-9999999"},
    ])
    crosswalk = build_name_position_crosswalk(nfl_players)
    assert ("kyle williams", "WR") not in crosswalk


def test_crosswalk_ignores_rows_missing_name_or_gsis():
    nfl_players = pd.DataFrame([
        {"display_name": "Bo Nix", "position": "QB", "gsis_id": None},
        {"display_name": None, "position": "WR", "gsis_id": "00-2222222"},
    ])
    crosswalk = build_name_position_crosswalk(nfl_players)
    assert crosswalk == {}


# ---------------------------------------------------------------------------
# resolve_vorp_key
# ---------------------------------------------------------------------------

def test_resolve_def_passes_through_team_code():
    key, method = resolve_vorp_key({"position": "DEF", "player_id": "PIT"}, {})
    assert key == "PIT"
    assert method == "def_passthrough"


def test_resolve_direct_gsis_id_when_present():
    key, method = resolve_vorp_key(
        {"position": "RB", "full_name": "Christian McCaffrey", "gsis_id": "00-0033280"}, {}
    )
    assert key == "00-0033280"
    assert method == "direct"


def test_resolve_falls_back_to_name_position_crosswalk():
    crosswalk = {("bo nix", "QB"): "00-1111111"}
    key, method = resolve_vorp_key(
        {"position": "QB", "full_name": "Bo Nix", "gsis_id": None}, crosswalk
    )
    assert key == "00-1111111"
    assert method == "name_fallback"


def test_resolve_unresolved_when_nothing_matches():
    key, method = resolve_vorp_key(
        {"position": "WR", "full_name": "Nobody Real", "gsis_id": None}, {}
    )
    assert key is None
    assert method == "unresolved"


# ---------------------------------------------------------------------------
# resolve_fresh_picks / attach_realized_vorp
# ---------------------------------------------------------------------------

def test_resolve_fresh_picks_excludes_keepers_and_other_seasons():
    draft_history = pd.DataFrame([
        {"season": 2024, "player_id": "p1", "round": 3, "is_keeper": False},
        {"season": 2024, "player_id": "p2", "round": 4, "is_keeper": True},
        {"season": 2023, "player_id": "p3", "round": 5, "is_keeper": False},
        {"season": 2025, "player_id": "p4", "round": 2, "is_keeper": None},
    ])
    sleeper_players = {
        "p1": {"position": "RB", "full_name": "Player One", "gsis_id": "00-0000001"},
        "p4": {"position": "WR", "full_name": "Player Four", "gsis_id": "00-0000004"},
    }
    nfl_players = pd.DataFrame([{"display_name": "x", "position": "QB", "gsis_id": "00-9999999"}])

    resolved = resolve_fresh_picks(draft_history, sleeper_players, nfl_players, seasons=[2024, 2025])
    seasons_seen = set(resolved["season"])
    assert seasons_seen == {2024, 2025}
    assert "p2" not in resolved["player_id"].values  # keeper excluded
    assert "p3" not in resolved["player_id"].values  # wrong season excluded
    assert set(resolved["player_id"]) == {"p1", "p4"}


def test_resolve_fresh_picks_does_not_exclude_k_or_def():
    """Locks in a deliberate design decision (see this module's own
    docstring): this curve prices what a draft SLOT is worth, and real
    late rounds are disproportionately K/DEF -- excluding them here would
    be a completely different, wrong decision from net_value.py's
    separate refusal to EVALUATE a K/DEF player. Nothing in
    resolve_fresh_picks/attach_realized_vorp should ever filter by
    position; this test exists so that filter can't be added by accident
    later without a test failing to flag it."""
    draft_history = pd.DataFrame([
        {"season": 2025, "player_id": "kicker1", "round": 18, "is_keeper": False},
        {"season": 2025, "player_id": "PIT", "round": 17, "is_keeper": False},
    ])
    sleeper_players = {
        "kicker1": {"position": "K", "full_name": "Test Kicker", "gsis_id": "00-0000009"},
        "PIT": {"position": "DEF", "player_id": "PIT"},
    }
    nfl_players = pd.DataFrame([{"display_name": "x", "position": "QB", "gsis_id": "00-9999999"}])

    resolved = resolve_fresh_picks(draft_history, sleeper_players, nfl_players, seasons=[2025])
    assert set(resolved["player_id"]) == {"kicker1", "PIT"}

    vorp_labels = pd.DataFrame([
        {"season": 2025, "player_id": "00-0000009", "vorp": 12.5},
        {"season": 2025, "player_id": "PIT", "vorp": -8.0},
    ])
    with_vorp = attach_realized_vorp(resolved, vorp_labels)
    assert with_vorp["vorp"].notna().all()
    assert set(with_vorp["vorp"]) == {12.5, -8.0}


def test_attach_realized_vorp_uses_vorp_not_vorp_next():
    resolved_picks = pd.DataFrame([
        {"season": 2024, "round": 1, "vorp_key": "00-0000001"},
    ])
    vorp_labels = pd.DataFrame([
        {"season": 2024, "player_id": "00-0000001", "vorp": 55.5, "vorp_next": 999.0},
    ])
    result = attach_realized_vorp(resolved_picks, vorp_labels)
    assert result.loc[0, "vorp"] == 55.5


# ---------------------------------------------------------------------------
# build_draft_capital_curve
# ---------------------------------------------------------------------------

def _picks(round_vorp_pairs):
    return pd.DataFrame(round_vorp_pairs, columns=["round", "vorp"])


def test_no_banding_when_every_round_clears_threshold():
    picks = _picks([(1, 10.0), (1, 20.0), (2, 5.0), (2, 15.0)])
    curve = build_draft_capital_curve(picks, min_picks_per_round=2)
    assert not curve.loc[1, "banded"]
    assert not curve.loc[2, "banded"]
    assert curve.loc[1, "avg_vorp"] == 15.0
    assert curve.loc[2, "avg_vorp"] == 10.0
    assert curve.loc[1, "n_picks"] == 2


def test_thin_trailing_rounds_band_forward():
    """Rounds 1-2 each individually clear the threshold; rounds 3-4 don't
    on their own but do once pooled together."""
    picks = _picks([
        (1, 10.0), (1, 20.0), (1, 30.0),
        (2, 10.0), (2, 20.0), (2, 30.0),
        (3, 100.0),
        (4, 200.0),
    ])
    curve = build_draft_capital_curve(picks, min_picks_per_round=2)
    assert not curve.loc[1, "banded"]
    assert not curve.loc[2, "banded"]
    assert curve.loc[3, "banded"]
    assert curve.loc[4, "banded"]
    assert curve.loc[3, "band_rounds"] == (3, 4)
    assert curve.loc[4, "band_rounds"] == (3, 4)
    assert curve.loc[3, "avg_vorp"] == 150.0
    assert curve.loc[3, "n_picks"] == 2


def test_trailing_round_with_no_higher_round_merges_backward():
    """The very last round can't reach the threshold even alone, and
    there's nothing higher to pull into it -- must merge into the
    previous already-completed band instead of being left short."""
    picks = _picks([
        (1, 10.0), (1, 20.0),
        (2, 999.0),  # only 1 pick, never reaches threshold=2 on its own
    ])
    curve = build_draft_capital_curve(picks, min_picks_per_round=2)
    assert curve.loc[2, "banded"]
    assert curve.loc[2, "band_rounds"] == (1, 2)
    assert curve.loc[1, "band_rounds"] == (1, 2)
    assert curve.loc[2, "n_picks"] == 3
    assert curve.loc[2, "avg_vorp"] == pytest.approx((10.0 + 20.0 + 999.0) / 3)


def test_entire_input_smaller_than_threshold_collapses_to_one_band():
    picks = _picks([(1, 10.0), (2, 20.0)])
    curve = build_draft_capital_curve(picks, min_picks_per_round=100)
    assert curve.loc[1, "band_rounds"] == (1, 2)
    assert curve.loc[2, "band_rounds"] == (1, 2)
    assert curve.loc[1, "n_picks"] == 2
    assert curve.loc[1, "banded"]


def test_raises_on_null_vorp_rather_than_silently_dropping():
    picks = pd.DataFrame([{"round": 1, "vorp": None}])
    with pytest.raises(ValueError):
        build_draft_capital_curve(picks)


def test_real_data_shape_bands_only_the_two_thin_late_rounds():
    """Reproduces the actual 2024-2025 counts found during investigation
    (round: n_picks), confirming min_picks_per_round=10 bands exactly
    rounds 16-17 and leaves every other round, including round 18 at
    exactly the threshold, alone."""
    counts = {
        1: 14, 2: 15, 3: 19, 4: 15, 5: 15, 6: 18, 7: 19, 8: 14, 9: 16, 10: 18,
        11: 18, 12: 17, 13: 18, 14: 20, 15: 18, 16: 9, 17: 9, 18: 10,
    }
    rows = [(round_number, 1.0) for round_number, n in counts.items() for _ in range(n)]
    picks = _picks(rows)
    curve = build_draft_capital_curve(picks, min_picks_per_round=10)

    for round_number in range(1, 16):
        assert not curve.loc[round_number, "banded"], f"round {round_number} should not be banded"
    assert not curve.loc[18, "banded"], "round 18 (n=10) sits exactly at the threshold"
    assert curve.loc[16, "banded"] and curve.loc[17, "banded"]
    assert curve.loc[16, "band_rounds"] == (16, 17)
    assert curve.loc[16, "n_picks"] == 18


# ---------------------------------------------------------------------------
# keeper_cost_vorp
# ---------------------------------------------------------------------------

def test_keeper_cost_vorp_exact_round():
    curve = build_draft_capital_curve(_picks([(1, 10.0), (1, 30.0)]), min_picks_per_round=2)
    assert keeper_cost_vorp(1, curve) == 20.0


def test_keeper_cost_vorp_falls_back_to_worst_round():
    curve = build_draft_capital_curve(
        _picks([(1, 100.0), (1, 100.0), (2, 5.0), (2, 5.0)]), min_picks_per_round=2
    )
    # round 99 was never observed at all -- falls back to the worst (minimum) band.
    assert keeper_cost_vorp(99, curve) == 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
