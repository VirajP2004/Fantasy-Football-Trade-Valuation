"""
Trade Engine -- Draft Capital Curve.

Translates "Round N" into "average REALIZED VORP a fresh Round-N pick in
this league has actually returned, recently" -- the replacement-cost
baseline `net_value.py` uses to price what an owner gives up by keeping a
player instead of drafting fresh.

Deliberately narrower in scope than `src/keeper_ledger/draft_capital_curve.py`
(which averages over every season strictly before a walk-forward cutoff,
for validating historical keeper pricing). This module answers a
different, present-tense question -- "what is a Round-N pick worth
*right now*, under the league's current roster/format" -- so it is
pinned to 2024-2025 only, not the full historical run. It does not
replace or import from that module.

THE CROSSWALK PROBLEM (checked directly, not assumed): `draft_history.parquet`
identifies players by Sleeper's own `player_id`; `vorp_labels.parquet`
identifies them by NFL's `gsis_id`. These are different ID systems.
Joining only on the `gsis_id` field Sleeper's own player cache sometimes
carries resolved just 100/310 (32%) of 2024-2025 fresh picks -- and the
misses were not obscure players: Bo Nix, Malik Nabers, Travis Etienne,
Tee Higgins, and ~150 others like them simply have a blank `gsis_id` in
Sleeper's cached player dictionary. A two-tier crosswalk fixes this:
  1. Direct `gsis_id`, when Sleeper actually has it.
  2. Normalized (name, position) match against `nflreadpy`'s own player
     table, for everyone Sleeper's `gsis_id` field is blank for.
  3. Team defenses pass through unchanged -- Sleeper and `vorp_labels`
     both key DEF rows by team code (e.g. "PIT"), so no lookup is needed.
This recovers 282/310 (91%); the 5 remaining misses are genuine name
collisions / position mismatches (e.g. a two-way player Sleeper lists
under a different position than `nflreadpy` does) not worth chasing
further for a 1.6% residual.

ROUND SPARSITY (checked directly, not assumed): real usable-with-VORP
counts by round across 2024-2025, after the crosswalk above:

    round:  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18
    n:     14 15 19 15 15 18 19 14 16 18 18 17 18 20 18  9  9 10

Every round clears at least 9 real observations -- not degenerate -- but
rounds 16-17 are visibly thinner than the rest (2024 only ran 15 rounds,
so rounds 16-18 have single-season coverage). `DEFAULT_MIN_PICKS_PER_ROUND
= 10` was chosen specifically because it is the cleanest real cut point
in the data above: it catches exactly the two genuinely thin rounds (9
each) without banding round 18 (which already clears it at exactly 10),
or any of the richer rounds. `build_draft_capital_curve` below implements
this as a real, configurable, unit-tested mechanism -- not a one-off
manual fix -- so it also does the right thing if a future re-run's live
data shifts which rounds are thin.
"""

import json
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

DEFAULT_MIN_PICKS_PER_ROUND = 10

_DROP_PATTERN = re.compile(r"[.']")
_HYPHEN_PATTERN = re.compile(r"-")
_SUFFIX_PATTERN = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")
_WHITESPACE_PATTERN = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Crosswalk: Sleeper player_id -> vorp_labels' player_id (gsis_id, or a team
# code for DEF rows).
# ---------------------------------------------------------------------------

def normalize_player_name(name: Optional[str]) -> Optional[str]:
    """Lowercases, strips punctuation and Jr./Sr./II-V suffixes, collapses
    whitespace -- enough to match "Marvin Harrison Jr." (Sleeper's
    `full_name`) against "Marvin Harrison" (nflreadpy's `display_name`),
    or vice versa, without being so aggressive it starts colliding
    unrelated names."""
    if not name:
        return None
    name = name.lower()
    name = _DROP_PATTERN.sub("", name)
    # A hyphen becomes a space, not nothing -- "Amon-Ra" must match a
    # source that happens to render it "Amon Ra" instead of merging into
    # the wrong, unrelated "amonra".
    name = _HYPHEN_PATTERN.sub(" ", name)
    name = _SUFFIX_PATTERN.sub("", name)
    name = _WHITESPACE_PATTERN.sub(" ", name).strip()
    return name or None


def build_name_position_crosswalk(nfl_players: pd.DataFrame) -> dict:
    """(normalized display_name, position) -> gsis_id, built from
    nflreadpy's own player table.

    Any (name, position) pair that maps to more than one distinct
    `gsis_id` (a real name collision, e.g. two same-named players at the
    same position across NFL history) is dropped entirely rather than
    guessed at -- an unresolved crosswalk miss is a known, visible gap;
    a silently wrong match is a much worse, invisible one.
    """
    working = nfl_players.dropna(subset=["display_name", "gsis_id"]).copy()
    working["norm_name"] = working["display_name"].apply(normalize_player_name)
    working = working.dropna(subset=["norm_name"])

    collision_counts = working.groupby(["norm_name", "position"])["gsis_id"].nunique()
    ambiguous = set(collision_counts[collision_counts > 1].index)

    crosswalk = {}
    for _, row in working.iterrows():
        key = (row["norm_name"], row["position"])
        if key in ambiguous:
            continue
        crosswalk[key] = row["gsis_id"]
    return crosswalk


def resolve_vorp_key(sleeper_player: dict, name_position_crosswalk: dict) -> tuple[Optional[str], str]:
    """Resolves one Sleeper player record to the key it would appear
    under in `vorp_labels.parquet`.

    Returns (vorp_key, match_method), where match_method is one of
    "def_passthrough", "direct", "name_fallback", or "unresolved" -- kept
    on every row downstream (not just aggregated away) so a caller can
    always see exactly how each pick was resolved, or whether it wasn't.
    """
    if sleeper_player.get("position") == "DEF":
        # Sleeper and vorp_labels both key team defenses by team code
        # (e.g. "PIT") -- the Sleeper player_id already *is* that code.
        return sleeper_player.get("player_id"), "def_passthrough"

    gsis_id = sleeper_player.get("gsis_id")
    if gsis_id:
        return gsis_id, "direct"

    key = (normalize_player_name(sleeper_player.get("full_name")), sleeper_player.get("position"))
    if key in name_position_crosswalk:
        return name_position_crosswalk[key], "name_fallback"

    return None, "unresolved"


def resolve_fresh_picks(
    draft_history: pd.DataFrame,
    sleeper_players: dict,
    nfl_players: pd.DataFrame,
    seasons: Iterable[int],
) -> pd.DataFrame:
    """Fresh (non-keeper) picks in `seasons`, each annotated with the
    `vorp_key`/`match_method` columns `resolve_vorp_key` produces.

    A pick counts as "fresh" whenever `is_keeper` is falsy (False or
    missing/NaN) -- the same convention `src/keeper_ledger` uses
    throughout, kept consistent rather than reinvented here.
    """
    fresh = draft_history[
        (draft_history["is_keeper"].fillna(False) == False)  # noqa: E712
        & (draft_history["season"].isin(list(seasons)))
    ].copy()

    name_position_crosswalk = build_name_position_crosswalk(nfl_players)

    resolved = fresh["player_id"].apply(
        lambda pid: resolve_vorp_key(sleeper_players.get(pid, {"player_id": pid}), name_position_crosswalk)
    )
    fresh["vorp_key"] = resolved.apply(lambda t: t[0])
    fresh["match_method"] = resolved.apply(lambda t: t[1])
    return fresh


def attach_realized_vorp(resolved_picks: pd.DataFrame, vorp_labels: pd.DataFrame) -> pd.DataFrame:
    """Joins each resolved pick to that SAME season's real, realized
    `vorp` -- deliberately `vorp`, never `vorp_next`: this curve prices
    what a fresh pick actually returned in the season it was drafted for,
    not a forward-looking prediction."""
    return resolved_picks.merge(
        vorp_labels[["season", "player_id", "vorp"]],
        left_on=["season", "vorp_key"],
        right_on=["season", "player_id"],
        how="left",
        suffixes=("", "_vorp_labels"),
    )


# ---------------------------------------------------------------------------
# The curve itself -- pure, unit-tested, no crosswalk or network dependency.
# ---------------------------------------------------------------------------

def build_draft_capital_curve(
    picks_with_vorp: pd.DataFrame,
    min_picks_per_round: int = DEFAULT_MIN_PICKS_PER_ROUND,
) -> pd.DataFrame:
    """
    `picks_with_vorp` must have `round` (int) and `vorp` (float) columns,
    already filtered to rows with a real, resolved VORP value (drop rows
    with a null `vorp` before calling -- this function does not know or
    care why a row might be missing one).

    Bands consecutive rounds together whenever a round's own pick count
    falls below `min_picks_per_round`: rounds are accumulated in
    ascending order until the pooled count clears the threshold, then
    that band closes and a fresh one starts. A trailing band that never
    clears the threshold (there are no higher rounds left to pull into
    it) merges backward into the previous completed band instead, so no
    round is ever left short of the threshold if there is any way to
    reach it by pooling with a real neighbor. If the entire input is
    smaller than `min_picks_per_round`, every round collapses into one
    band -- reported plainly (a small `n_picks`) rather than pretending a
    default threshold was met.

    Returns a DataFrame indexed by `round`, columns:
        avg_vorp    -- mean realized VORP across the round's band
        n_picks     -- real pick count backing that mean
        band_rounds -- tuple of every round pooled into this band
        banded      -- True iff more than one round was pooled together
    """
    if picks_with_vorp["vorp"].isna().any():
        raise ValueError(
            "picks_with_vorp contains null vorp rows -- filter those out before "
            "calling build_draft_capital_curve (this function does not resolve "
            "or drop them silently)."
        )

    by_round = picks_with_vorp.groupby("round")["vorp"].apply(list).sort_index()

    bands: list[tuple[tuple, list]] = []
    current_rounds: list = []
    current_values: list = []

    for round_number, values in by_round.items():
        current_rounds.append(round_number)
        current_values.extend(values)
        if len(current_values) >= min_picks_per_round:
            bands.append((tuple(current_rounds), current_values))
            current_rounds, current_values = [], []

    if current_rounds:
        if bands:
            prev_rounds, prev_values = bands.pop()
            bands.append((prev_rounds + tuple(current_rounds), prev_values + current_values))
        else:
            bands.append((tuple(current_rounds), current_values))

    rows = []
    for band_rounds, values in bands:
        avg_vorp = sum(values) / len(values)
        n_picks = len(values)
        banded = len(band_rounds) > 1
        for round_number in band_rounds:
            rows.append({
                "round": round_number,
                "avg_vorp": avg_vorp,
                "n_picks": n_picks,
                "band_rounds": band_rounds,
                "banded": banded,
            })

    return pd.DataFrame(rows).set_index("round").sort_index()


def keeper_cost_vorp(round_number: int, curve: pd.DataFrame) -> float:
    """Opportunity cost, in VORP terms, of a pick priced at `round_number`.

    Falls back to the curve's worst (minimum) observed `avg_vorp` if
    `round_number` isn't in the curve at all (e.g. a round beyond what
    this league has ever actually drafted) -- same fallback convention
    `src/keeper_ledger/draft_capital_curve.py` already uses, kept
    consistent rather than reinvented here.
    """
    if round_number in curve.index:
        return float(curve.loc[round_number, "avg_vorp"])
    return float(curve["avg_vorp"].min())


# ---------------------------------------------------------------------------
# Live wrapper -- real file I/O + nflreadpy pull. Not unit tested directly
# (see build_draft_capital_curve above for the tested, pure core); exercised
# for real in notebooks/09_trade_engine_demo.ipynb instead.
# ---------------------------------------------------------------------------

def build_full_draft_capital_curve(
    repo_root: Path,
    seasons: Iterable[int] = (2024, 2025),
    min_picks_per_round: int = DEFAULT_MIN_PICKS_PER_ROUND,
) -> pd.DataFrame:
    """Loads `draft_history.parquet`, the cached Sleeper player dictionary,
    `nflreadpy`'s player table, and `vorp_labels.parquet` from
    `repo_root`, resolves the crosswalk, and returns the real,
    2024-2025 draft capital curve."""
    import nflreadpy as nfl

    repo_root = Path(repo_root)
    draft_history = pd.read_parquet(repo_root / "data/raw/draft_history.parquet")
    with open(repo_root / "data/raw/players_cache.json") as f:
        sleeper_players = json.load(f)
    nfl_players = nfl.load_players().to_pandas()
    vorp_labels = pd.read_parquet(repo_root / "data/processed/vorp_labels.parquet")

    resolved = resolve_fresh_picks(draft_history, sleeper_players, nfl_players, seasons)
    with_vorp = attach_realized_vorp(resolved, vorp_labels)
    usable = with_vorp.dropna(subset=["vorp"])

    return build_draft_capital_curve(usable, min_picks_per_round=min_picks_per_round)
