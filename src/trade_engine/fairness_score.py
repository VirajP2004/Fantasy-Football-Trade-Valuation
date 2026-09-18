"""
Trade Engine -- Fairness Score.

Converts each side's adjusted net trade value (`net_value.py`'s
`net_kvs_delta`, or `positional_need.py`'s `adjusted_net_kvs_delta` once
that adjustment layer is included) into a single, readable fairness
metric: percentage of total value moved. "Team A receives 62% of the
trade's total value" is the plain-language form of a 62/38 split.

WHY PERCENTAGE-OF-TOTAL, NOT A RAW DIFFERENCE OR RATIO:
- A raw value difference ("+42 VORP in Team A's favor") is not
  self-scaling -- +42 means something completely different in a trade
  worth 50 total VORP than one worth 500. Percentage-of-total is scale
  invariant, so it is comparable across trades of any size.
- A ratio ("1.6x") breaks down or reports misleadingly when either
  side's value is at or near zero or negative, and reads less directly
  than a percentage to someone with no modeling background.

THE SHIFT CONVENTION (a real modeling choice, not a neutral technical
detail): percentage-of-total is only meaningful when there is an actual
positive pool of value to divide. `value / (value_a + value_b)` breaks
immediately when either side is negative (a team can genuinely receive a
package that nets negative for them in isolation) -- divide-by-zero, or
a share outside a readable [0, 100] range. When exactly one side is
negative, this module shifts BOTH sides by the same constant -- the
amount needed to bring the more negative side up to exactly 0 -- before
taking the percentage. Shifting both sides equally preserves the gap
between them while guaranteeing a defined, always-valid [0, 100] result.
`shifted=True` on the result says plainly whether this happened.

WHY THAT SAME SHIFT IS *NOT* USED WHEN BOTH SIDES ARE NEGATIVE (found and
fixed after directly checking a suspected edge case, not assumed safe):
`compute_fairness_score(-100.0, -99.0)` -- two net losses within 1% of
each other, about as close to a fair (if bad) trade for both sides as
two numbers can be -- shifts to (0, 1) and reports **0% / 100%**. The
shift is mathematically consistent (it always has been), but it is
actively misleading here: when both sides are net losers, there is no
real "gained value" pool being split at all, and shifting the floor up
to zero turns the tiny absolute gap between two large losses into the
ENTIRE denominator, so any small difference maps to a near-total swing.
This is not a rare corner case -- it is the generic behavior any time
both sides are negative and close in magnitude, which is exactly the
"two comparably bad trades for both sides" case a fairness score most
needs to get right.

**Fix:** when both `team_a_value` and `team_b_value` are negative, this
module does NOT use percentage-of-total (shifted or otherwise). It falls
back to a magnitude-relative split instead:

    diff = team_a_value - team_b_value
    magnitude_sum = abs(team_a_value) + abs(team_b_value)
    team_a_share_pct = 50 + 50 * (diff / magnitude_sum)   # clamped to [0, 100]

This centers on 50/50 (two equal losses = an even split, not an
undefined or extreme one) and moves toward 0/100 only as one side's loss
genuinely dominates the other's, in proportion to how much of the total
PAIN (not value) each side is carrying. Re-checked against the
discovered case: `(-100, -99)` now gives team_a_share_pct ~= 49.75%,
team_b_share_pct ~= 50.25% -- close to even, matching the two values'
actual near-equivalence, not the previous 0%/100%. `fairness_method` on
the result records which formula was actually used
(`"percentage_of_total"` or `"magnitude_relative_both_negative"`), so a
caller can never mistake one for the other.
"""

from dataclasses import dataclass

FAIRNESS_METHOD_PERCENTAGE_OF_TOTAL = "percentage_of_total"
FAIRNESS_METHOD_MAGNITUDE_RELATIVE_BOTH_NEGATIVE = "magnitude_relative_both_negative"


@dataclass
class FairnessScoreResult:
    team_a_value: float
    team_b_value: float
    team_a_share_pct: float
    team_b_share_pct: float
    shifted: bool
    fairness_method: str


def _both_negative_magnitude_relative_score(team_a_value: float, team_b_value: float) -> tuple[float, float]:
    diff = team_a_value - team_b_value
    magnitude_sum = abs(team_a_value) + abs(team_b_value)
    # magnitude_sum == 0 only if both are exactly 0.0, which is not
    # "negative" and never reaches this branch -- see compute_fairness_score.
    team_a_share_pct = 50.0 + 50.0 * (diff / magnitude_sum)
    team_a_share_pct = max(0.0, min(100.0, team_a_share_pct))
    return team_a_share_pct, 100.0 - team_a_share_pct


def compute_fairness_score(team_a_value: float, team_b_value: float) -> FairnessScoreResult:
    """`team_a_value`/`team_b_value` are the total adjusted value each
    side RECEIVES in the trade (sum across every player/pick moving to
    that side), not what they give up."""
    if team_a_value < 0 and team_b_value < 0:
        team_a_share_pct, team_b_share_pct = _both_negative_magnitude_relative_score(team_a_value, team_b_value)
        return FairnessScoreResult(
            team_a_value=team_a_value,
            team_b_value=team_b_value,
            team_a_share_pct=team_a_share_pct,
            team_b_share_pct=team_b_share_pct,
            shifted=False,
            fairness_method=FAIRNESS_METHOD_MAGNITUDE_RELATIVE_BOTH_NEGATIVE,
        )

    shift = max(0.0, -min(team_a_value, team_b_value))
    shifted_a = team_a_value + shift
    shifted_b = team_b_value + shift
    total = shifted_a + shifted_b

    if total == 0:
        # Both sides worth exactly the same after shifting (including the
        # genuine 0/0 case, e.g. two empty trade sides) -- an even split
        # is the only fairness-preserving answer, not an undefined result.
        team_a_share_pct, team_b_share_pct = 50.0, 50.0
    else:
        team_a_share_pct = 100.0 * shifted_a / total
        team_b_share_pct = 100.0 * shifted_b / total

    return FairnessScoreResult(
        team_a_value=team_a_value,
        team_b_value=team_b_value,
        team_a_share_pct=team_a_share_pct,
        team_b_share_pct=team_b_share_pct,
        shifted=shift != 0.0,
        fairness_method=FAIRNESS_METHOD_PERCENTAGE_OF_TOTAL,
    )


def fairness_score_for_trade(team_a_received: list[float], team_b_received: list[float]) -> FairnessScoreResult:
    """Convenience wrapper for a trade with multiple assets per side --
    sums each side's received values (e.g. a list of `net_kvs_delta` or
    `adjusted_net_kvs_delta` figures) and scores the total."""
    return compute_fairness_score(sum(team_a_received), sum(team_b_received))


def describe_fairness(result: FairnessScoreResult, team_a_name: str = "Team A") -> str:
    """"Team A receives 62% of the trade's total value." -- always framed
    from `team_a_name`'s side, whichever way the trade actually leans."""
    return f"{team_a_name} receives {result.team_a_share_pct:.0f}% of the trade's total value."
