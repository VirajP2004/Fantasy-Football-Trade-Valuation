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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
