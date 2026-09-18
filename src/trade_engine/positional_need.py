"""
Trade Engine -- Positional Need Adjustment.

Adjusts a trade's raw `net_kvs_delta` (`net_value.py`) by how much the
receiving team actually needs that position, rather than treating every
incoming player's value as position-agnostic. A team already stacked at
a position should value another player there less than the raw
predicted-KVS-minus-keeper-cost number says; a team thin at a position
should value one more than that.

GROUNDING (not an arbitrary multiplier): the adjustment is built
directly on `scarcity_z`, `net_value.py`'s own positional-scarcity
feature (see `POSITION_FEATURES`), not a new invented scale.
`scarcity_z` standardizes a player's VORP within their own
position-season -- `(vorp - position_mean_vorp) / position_std_vorp_that_season`
-- specifically so "1.0" means the same thing (one standard deviation
above a typical player at that position, that season) whether the
position is QB or TE. That is exactly the property a cross-position need
adjustment requires: "how deep is this team at position X" has to be
expressed in units comparable to "how much value is this incoming
player" for the two to combine into one adjustment.

THE FORMULA:
    team_need_z = mean(scarcity_z of the roster's players at `position`)
    adjustment_vorp = -weight * team_need_z * position_std_vorp_that_season
    adjusted_net_kvs_delta = raw_net_kvs_delta + adjustment_vorp

`position_std_vorp_that_season` -- the SAME quantity `scarcity_z` itself
is built from -- is reused here as the conversion factor from "SD units
of roster need" back to "VORP points of adjustment". This is why the
adjustment is additive (in VORP points, the same unit as
`net_kvs_delta`) rather than a percentage multiplier: `net_kvs_delta`
can be negative (a real net loss in isolation), and a multiplicative
discount/boost flips the wrong direction on a negative number (boosting
a needed-but-bad trade would make it *more* negative, not less). An
additive VORP adjustment does not have that failure mode, and it means a
fixed SD of roster need swings the result by a different,
position-appropriate amount depending on how spread out that position's
real talent actually is that season -- not a flat constant applied
identically everywhere.

    - team_need_z > 0 (roster's players there are above-average):
      adjustment_vorp < 0, incoming value at that position is discounted.
    - team_need_z < 0 (roster's players there are below average, or the
      position is empty -- see EMPTY_POSITION_NEED_Z): adjustment_vorp > 0,
      incoming value is boosted.
    - team_need_z == 0, or weight == 0: no adjustment.

`weight` (default `NEED_ADJUSTMENT_WEIGHT`) is the one genuinely free
constant here: it controls how much a full standard deviation of roster
need swings the adjustment, in units of a full standard deviation of
that position's own VORP spread -- not what scale need is measured on
(that comes from `scarcity_z`/`position_std_vorp_that_season`, not this
module). `team_need_z` is clamped to +/-`MAX_ABS_NEED_Z` before use so an
extreme roster (many elite players stacked at one position) can't swing
the adjustment unboundedly.

EMPTY_POSITION_NEED_Z: a position with zero rostered players is not
"average" need, it is the thinnest possible case -- treated as a fixed,
strongly-negative need_z rather than raising or silently defaulting to
0.0/no-adjustment, since "no players at all" is real signal a
roster-construction adjustment should not discard. Flagged via
`caveats`, the same never-silently-dropped pattern `net_value.py` uses
for its own confidence flags. Checked directly against this league's
real current rosters (`data/processed/roster_keeper_table_2027.csv`,
166/183 players joined to `vorp_labels.parquet`'s 2025 season): every
real, populated team/position's `team_need_z` fell in [-0.29, +4.26]
overall ([+0.22, +4.26] across QB/RB/WR/TE specifically) -- real rosters
skew well above replacement-level average by construction (they're a
curated subset of the full NFL population `scarcity_z`'s mean/std is
computed over), so -2.0 sits comfortably outside any range a real,
present-but-bad position has actually produced, and is never mistaken
for one.
"""

from dataclasses import dataclass, field

NEED_ADJUSTMENT_WEIGHT = 0.25
MAX_ABS_NEED_Z = 3.0
EMPTY_POSITION_NEED_Z = -2.0

CAVEAT_TEXT = {
    "empty_position": (
        f"empty_position: the roster has no players at this position at all -- treated as a "
        f"fixed, maximally-thin need_z ({EMPTY_POSITION_NEED_Z}) rather than an assumed-average "
        "0.0, since 'no players' is real signal a roster-construction adjustment should not discard."
    ),
}


@dataclass
class PositionalNeedResult:
    position: str
    team_need_z: float
    roster_player_count_at_position: int
    position_std_vorp_that_season: float
    weight: float
    raw_net_kvs_delta: float
    adjustment_vorp: float
    adjusted_net_kvs_delta: float
    caveats: list = field(default_factory=list)


def team_positional_need_z(roster: list[dict], position: str) -> tuple[float, int]:
    """The roster's own average `scarcity_z` among its players at
    `position` -- the team's current strength/depth there, expressed in
    the same standardized units as any individual player's `scarcity_z`.

    Returns (need_z, roster_player_count_at_position). An empty position
    returns `EMPTY_POSITION_NEED_Z` rather than an undefined average.
    `need_z` is clamped to +/-`MAX_ABS_NEED_Z`.
    """
    at_position = [p for p in roster if p["position"] == position]
    if not at_position:
        return EMPTY_POSITION_NEED_Z, 0

    raw_need_z = sum(p["scarcity_z"] for p in at_position) / len(at_position)
    clamped_need_z = max(-MAX_ABS_NEED_Z, min(MAX_ABS_NEED_Z, raw_need_z))
    return clamped_need_z, len(at_position)


def positional_need_adjustment_vorp(
    team_need_z: float,
    position_std_vorp_that_season: float,
    weight: float = NEED_ADJUSTMENT_WEIGHT,
) -> float:
    """The VORP-point adjustment itself -- see this module's docstring
    for why `position_std_vorp_that_season` is the conversion factor."""
    return -weight * team_need_z * position_std_vorp_that_season


def apply_positional_need_adjustment(
    raw_net_kvs_delta: float,
    roster: list[dict],
    position: str,
    position_std_vorp_that_season: float,
    *,
    weight: float = NEED_ADJUSTMENT_WEIGHT,
) -> PositionalNeedResult:
    """Combines `net_value.py`'s raw `net_kvs_delta` with the receiving
    team's own roster construction at `position` into an adjusted value.
    Pure and fully unit-testable -- takes the roster and
    `position_std_vorp_that_season` as plain inputs rather than reaching
    into `vorp_labels` itself, same separation `net_value.py` keeps
    between `evaluate_net_value` (pure) and `predict_kvs` (live data)."""
    team_need_z, count = team_positional_need_z(roster, position)
    adjustment = positional_need_adjustment_vorp(team_need_z, position_std_vorp_that_season, weight)

    caveats = []
    if count == 0:
        caveats.append(CAVEAT_TEXT["empty_position"])

    return PositionalNeedResult(
        position=position,
        team_need_z=team_need_z,
        roster_player_count_at_position=count,
        position_std_vorp_that_season=position_std_vorp_that_season,
        weight=weight,
        raw_net_kvs_delta=raw_net_kvs_delta,
        adjustment_vorp=adjustment,
        adjusted_net_kvs_delta=raw_net_kvs_delta + adjustment,
        caveats=caveats,
    )
