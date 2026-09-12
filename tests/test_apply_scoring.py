"""
Regression test for the fumbles_lost scoring bug.

apply_scoring.py used to look for a column literally named "fumbles_lost"
to apply the fumbles.fumbles_lost rule from league_scoring_rules.yaml.
That column doesn't exist in the raw dataset at all (the real column is
the pre-aggregated "fumbles_lost_total" — the sum of sack_fumbles_lost,
rushing_fumbles_lost, and receiving_fumbles_lost) so the check silently
evaluated False and the fumbles penalty never applied to ANY player.

This test hand-picks a real player-season with a known nonzero
fumbles_lost_total (Saquon Barkley, 2023 — fumbles_lost_total=2) and
asserts the fumbles penalty is applied as a flat, position-independent
-2.0/fumble, so this exact silent no-op can never come back undetected.
"""

import pandas as pd
import pytest

from src.scoring.apply_scoring import compute_fantasy_points, load_scoring_rules

# Saquon Barkley's real 2023 season stats (data/raw/raw_seasonal_stats.csv).
BARKLEY_2023 = {
    "rushing_yards": 962,
    "rushing_tds": 6,
    "rushing_2pt_conversions": 1,
    "receptions": 41,
    "receiving_yards": 280,
    "receiving_tds": 4,
    "receiving_2pt_conversions": 0,
    "passing_yards": 0,
    "passing_tds": 0,
    "passing_interceptions": 0,
    "passing_2pt_conversions": 0,
    "fumbles_lost_total": 2,
}


def test_fumbles_lost_total_applies_flat_penalty_regression():
    """custom_fantasy_points must drop by exactly 2 * fumbles_lost_total
    relative to a dataset where the fumbles column is missing entirely —
    the exact scenario the old "fumbles_lost" column-name bug produced."""
    rules = load_scoring_rules()
    fumbles_lost_total = BARKLEY_2023["fumbles_lost_total"]
    fumbles_penalty_weight = rules["fumbles"]["fumbles_lost"]  # -2.0 per league rules

    df_with_fumbles = pd.DataFrame([BARKLEY_2023])
    df_without_fumbles = pd.DataFrame(
        [{k: v for k, v in BARKLEY_2023.items() if k != "fumbles_lost_total"}]
    )

    fp_with = compute_fantasy_points(df_with_fumbles, rules).iloc[0]
    fp_without = compute_fantasy_points(df_without_fumbles, rules).iloc[0]

    # The bug's signature: dropping the fumbles column silently zeroed out
    # the penalty instead of erroring — so the fixed version must show a
    # real, exact difference between "column present" and "column absent".
    assert fp_without - fp_with == pytest.approx(-fumbles_lost_total * fumbles_penalty_weight)
    assert fp_with == pytest.approx(fp_without + fumbles_lost_total * fumbles_penalty_weight)

    # Matches the real total from the Sleeper app for Barkley's 2023 season.
    assert fp_with == pytest.approx(202.7)


def test_fumbles_penalty_is_flat_regardless_of_position():
    """The league rule is the same -2.0/fumble for every position — no
    QB/RB/WR/TE branching. A QB-shaped row and an RB-shaped row losing the
    same number of fumbles must lose the same number of points for it."""
    rules = load_scoring_rules()
    fumbles_penalty_weight = rules["fumbles"]["fumbles_lost"]

    qb_row = pd.DataFrame([{"passing_yards": 0, "fumbles_lost_total": 3}])
    rb_row = pd.DataFrame([{"rushing_yards": 0, "fumbles_lost_total": 3}])

    qb_fp = compute_fantasy_points(qb_row, rules).iloc[0]
    rb_fp = compute_fantasy_points(rb_row, rules).iloc[0]

    assert qb_fp == pytest.approx(3 * fumbles_penalty_weight)
    assert rb_fp == pytest.approx(3 * fumbles_penalty_weight)
    assert qb_fp == rb_fp


# Ka'imi Fairbairn's real 2025 season stats (data/raw/raw_seasonal_stats.csv).
# Verified against this league's actual live Sleeper scoring (summed
# players_points across all 18 fantasy weeks he was rostered): the real
# season total is 190.0. Computing from raw stats without counting his 1
# blocked FG as a miss gives 191.0 — confirming blocked kicks must fold
# into fg_missed/pat_missed or the kicking category silently overcounts.
FAIRBAIRN_2025 = {
    "fg_made_0_19": 0,
    "fg_made_20_29": 8,
    "fg_made_30_39": 11,
    "fg_made_40_49": 16,
    "fg_made_50_59": 9,
    "fg_made_60_": 0,
    "fg_missed": 3,
    "fg_blocked": 1,
    "pat_made": 28,
    "pat_missed": 0,
    "pat_blocked": 0,
}


def test_kicker_scoring_regression_fairbairn_2025():
    """Regression guard for the kicker scoring gap: apply_scoring.py used
    to have no "kicking" section at all, so every kicker scored 0.0
    regardless of performance. Pins Ka'imi Fairbairn's real, Sleeper-
    verified 2025 season total."""
    rules = load_scoring_rules()
    df = pd.DataFrame([FAIRBAIRN_2025])
    fp = compute_fantasy_points(df, rules).iloc[0]
    assert fp == pytest.approx(190.0)


def test_blocked_kicks_count_as_misses():
    """A blocked FG/PAT must be penalized the same as a missed one — this
    is what the naive per-column mapping (fg_missed only, ignoring
    fg_blocked) gets wrong, and it's a real, silent 1-point-per-block
    undercount if skipped."""
    rules = load_scoring_rules()

    without_blocks = {**FAIRBAIRN_2025, "fg_blocked": 0}
    with_blocks = FAIRBAIRN_2025

    fp_without = compute_fantasy_points(pd.DataFrame([without_blocks]), rules).iloc[0]
    fp_with = compute_fantasy_points(pd.DataFrame([with_blocks]), rules).iloc[0]

    fg_missed_weight = rules["kicking"]["fg_missed"]
    assert fp_with - fp_without == pytest.approx(fg_missed_weight)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
