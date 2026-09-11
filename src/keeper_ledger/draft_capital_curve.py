"""
Phase 1B — Draft Capital Curve.

Translates "Round N" into "average VORP a fresh Round-N pick has actually
returned historically." This is the replacement-cost baseline Stage 2 uses
to price what an owner gives up by keeping instead of drafting fresh.

DEPENDENCY NOTE: this module needs `vorp_labels` (season, player_id,
realized_vorp), which is a Phase 2 deliverable and does not exist yet in
this repo. The functions below are ready to run the moment
`data/processed/vorp_labels.parquet` exists — nothing here blocks on the
Stage 1 model being trained, only on the VORP label table being built.
"""

import pandas as pd


def build_draft_capital_curve(
    draft_history: pd.DataFrame,
    vorp_labels: pd.DataFrame,
    as_of_season: int,
) -> pd.Series:
    """
    Average realized VORP by draft round, computed ONLY from fresh
    (non-keeper) picks in seasons strictly before `as_of_season` — this
    is the walk-forward boundary: never let a season "see" its own or a
    future season's realized outcomes when pricing its own trades.

    Returns: pd.Series indexed by round -> mean realized_vorp
    """
    fresh_picks = draft_history[
        (draft_history["is_keeper"].fillna(False) == False)  # noqa: E712
        & (draft_history["season"] < as_of_season)
    ]

    merged = fresh_picks.merge(
        vorp_labels, on=["season", "player_id"], how="inner"
    )

    if merged.empty:
        raise ValueError(
            f"No fresh-pick/VORP overlap found for seasons < {as_of_season}. "
            "Check that vorp_labels covers the same seasons as draft_history."
        )

    return merged.groupby("round")["realized_vorp"].mean().sort_index()


def keeper_cost_vorp(round_number: int, curve: pd.Series) -> float:
    """
    Opportunity cost, in VORP terms, of a keeper priced at `round_number`.
    Falls back to the worst (minimum) observed round value if the exact
    round is missing from the curve (e.g. very late rounds with sparse
    historical data).
    """
    if round_number in curve.index:
        return float(curve.loc[round_number])
    return float(curve.min())
