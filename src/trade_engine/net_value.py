"""
Trade Engine -- Net Value.

net_KVS_delta = predicted_KVS - keeper_cost_VORP

`predicted_KVS` is the player's predicted next-season VORP, from the
appropriate position's tuned XGBoost model (QB/RB/WR/TE) -- the only four
positions this module supports. **K and DEF are out of scope, full stop**:
`predict_kvs`/`evaluate_player_trade_value` raise `UnsupportedPositionError`
for either position rather than returning a number of any kind. This
follows `roadmap.md`'s Phase 4 scope decision directly (see
`notebooks/06_scope_decision_k_def.ipynb` for the full reasoning: DEF's
untuned model actively underperformed simple persistence, K/DEF have a
structurally thin feature set, and both show real-world year-to-year
volatility no available features capture well) -- no ML model, and no
non-ML heuristic either, is produced for K/DEF anywhere in this module.
An earlier version of this module *did* have a K/DEF heuristic path
(current-season realized VORP, explicitly labeled as non-model output);
it has been removed entirely, not just hidden, because a heuristic
number sitting next to real model predictions in the same return type is
too easy to misuse downstream regardless of how it's labeled -- an
explicit error at the boundary is safer than a value that still has to
be filtered back out later.

`keeper_cost_VORP` is `scripts/project_roster_keeper_costs.py`'s
already-validated projected keeper round, converted to a VORP figure via
`draft_capital_curve.keeper_cost_vorp`. (Note: the draft capital curve
itself still legitimately includes K/DEF fresh picks in its per-round
averages -- pricing what a round is worth is a different question from
whether a specific player can be evaluated, since any position can
occupy any round. See `draft_capital_curve.py`'s own module docstring.)

CONFIDENCE FLAGS ARE NEVER DROPPED SILENTLY: every `NetValueResult` this
module produces carries `low_confidence_extreme_delta` and
`no_delta_history` explicitly, plus a human-readable `caveats` list built
from them -- the same two flags QB/RB/WR/TE's model notebooks (and the
Phase 5 SHAP notebooks) have used all along:
  - `low_confidence_extreme_delta`: `|vorp_delta_yoy| > 100` -- this
    exact region is the one all four `08x_shap_*.ipynb` notebooks and
    every `06x_model_*.ipynb` notebook found degrades held-out MAE.
  - `no_delta_history`: `vorp_delta_yoy` is null (typically a true
    rookie) -- kept as a SEPARATE flag from the above, never folded into
    it (see roadmap.md's `NaN > threshold` bug writeup) -- a missing
    delta is a different kind of uncertainty than a measured extreme one.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from src.trade_engine.draft_capital_curve import keeper_cost_vorp

LOW_CONFIDENCE_DELTA_THRESHOLD = 100

POSITION_FEATURES = {
    "QB": ["scarcity_z", "draft_pick_inverse", "vorp_delta_yoy", "passing_epa", "age", "ol_pass_protection_proxy"],
    "RB": ["scarcity_z", "vorp_delta_yoy", "age", "draft_pick_inverse"],
    "WR": ["scarcity_z", "wopr", "vorp_delta_yoy", "receiving_epa", "air_yards_share", "age", "draft_pick_inverse", "target_share"],
    "TE": ["scarcity_z", "air_yards_share", "receiving_epa", "wopr", "target_share",
           "draft_tier_Round 2-3", "draft_pick_inverse", "vorp_delta_yoy", "injury_designations_count"],
}
MODELED_POSITIONS = set(POSITION_FEATURES)
UNSUPPORTED_POSITIONS = {"K", "DEF"}

CAVEAT_TEXT = {
    "low_confidence_extreme_delta": (
        "low_confidence_extreme_delta: |vorp_delta_yoy| > 100 -- this region has confirmed, "
        "held-out degraded accuracy at every modeled position (see 08a-08d_shap_*.ipynb); "
        "treat the exact predicted_KVS number with real skepticism, not just the direction."
    ),
    "no_delta_history": (
        "no_delta_history: vorp_delta_yoy is missing (no usable prior season, e.g. a true "
        "rookie) -- a DIFFERENT and separate reason for caution than low_confidence_extreme_delta, "
        "not a stronger or weaker version of it."
    ),
}


class UnsupportedPositionError(ValueError):
    """Raised for any position this module will not produce a predicted_KVS
    for -- currently K and DEF, per roadmap.md's Phase 4 scope decision.
    Never caught internally and silently papered over with a fallback
    number; this is meant to stop a caller at the boundary."""


@dataclass
class NetValueResult:
    player_name: str
    season: int
    position: str
    predicted_kvs: float
    keeper_cost_vorp: float
    net_kvs_delta: float
    low_confidence_extreme_delta: bool
    no_delta_history: bool
    caveats: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pure core -- fully unit-testable, no model loading or nflreadpy dependency.
# ---------------------------------------------------------------------------

def evaluate_net_value(
    player_name: str,
    season: int,
    position: str,
    predicted_kvs: float,
    keeper_cost_vorp_value: float,
    *,
    low_confidence_extreme_delta: bool = False,
    no_delta_history: bool = False,
) -> NetValueResult:
    """Combines an already-computed prediction and keeper cost into a
    `NetValueResult`, attaching every applicable caveat explicitly. This
    is the pure, testable heart of the module -- `predict_kvs` and
    `evaluate_player_trade_value` below do the live data/model work and
    then call this."""
    caveats = []
    if low_confidence_extreme_delta:
        caveats.append(CAVEAT_TEXT["low_confidence_extreme_delta"])
    if no_delta_history:
        caveats.append(CAVEAT_TEXT["no_delta_history"])

    return NetValueResult(
        player_name=player_name,
        season=season,
        position=position,
        predicted_kvs=predicted_kvs,
        keeper_cost_vorp=keeper_cost_vorp_value,
        net_kvs_delta=predicted_kvs - keeper_cost_vorp_value,
        low_confidence_extreme_delta=low_confidence_extreme_delta,
        no_delta_history=no_delta_history,
        caveats=caveats,
    )


def net_value_from_projected_round(
    player_name: str,
    season: int,
    position: str,
    predicted_kvs: float,
    projected_keeper_round: int,
    draft_curve: pd.DataFrame,
    **flag_kwargs,
) -> NetValueResult:
    """Same as `evaluate_net_value`, but takes a projected keeper ROUND
    (as `scripts/project_roster_keeper_costs.py` produces) and converts
    it to VORP via the draft capital curve, rather than requiring the
    caller to do that conversion themselves."""
    keeper_cost = keeper_cost_vorp(projected_keeper_round, draft_curve)
    return evaluate_net_value(player_name, season, position, predicted_kvs, keeper_cost, **flag_kwargs)


# ---------------------------------------------------------------------------
# Live feature-building -- one function per modeled position, each mirroring
# that position's own 06x_model_*.ipynb pipeline exactly (same columns, same
# construction), since that pipeline IS the definition of the shipped
# model's inputs. Not unit tested directly (real nflreadpy pulls); the
# tested surface is evaluate_net_value/net_value_from_projected_round above.
# ---------------------------------------------------------------------------

def position_std_vorp_that_season(vorp_labels: pd.DataFrame, position: str, season: int) -> float:
    """The same `position_std_vorp_that_season` quantity `_add_scarcity_z_and_delta`
    computes internally below (and discards once `scarcity_z` is derived from
    it) -- exposed here as its own small, pure, testable function so a
    caller outside this module (`positional_need.py`'s VORP-unit
    conversion factor, via `evaluate_trade.py`) gets the IDENTICAL number
    `scarcity_z` was itself standardized against, rather than re-deriving
    it a second, possibly-diverging way."""
    position_rows = vorp_labels[(vorp_labels["position"] == position) & (vorp_labels["season"] == season)]
    if position_rows.empty:
        raise ValueError(f"No {position!r} rows for season {season} in vorp_labels")
    return float(position_rows["vorp"].std())


def _add_scarcity_z_and_delta(df: pd.DataFrame, vorp_labels: pd.DataFrame, position: str) -> pd.DataFrame:
    season_position_stats = (
        vorp_labels.groupby(["season", "position"])["vorp"]
        .agg(position_mean_vorp="mean", position_std_vorp_that_season="std")
        .reset_index()
    )
    pos_stats = season_position_stats[season_position_stats["position"] == position]
    df = df.merge(pos_stats[["season", "position_mean_vorp", "position_std_vorp_that_season"]], on="season", how="left")
    df["scarcity_z"] = (df["vorp"] - df["position_mean_vorp"]) / df["position_std_vorp_that_season"]
    df = df.drop(columns=["position_mean_vorp", "position_std_vorp_that_season"])

    prior = df[["player_id", "season", "vorp"]].copy()
    prior["season"] = prior["season"] + 1
    prior = prior.rename(columns={"vorp": "vorp_last_season"})
    df = df.merge(prior, on=["player_id", "season"], how="left")
    df["vorp_delta_yoy"] = df["vorp"] - df["vorp_last_season"]
    return df.drop(columns=["vorp_last_season"])


def _add_age_and_draft_pick_inverse(df: pd.DataFrame, nfl_players: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(nfl_players[["gsis_id", "birth_date", "draft_pick"]], left_on="player_id", right_on="gsis_id", how="left")
    df["birth_date"] = pd.to_datetime(df["birth_date"])
    season_start = pd.to_datetime(df["season"].astype(str) + "-09-01")
    df["age"] = (season_start - df["birth_date"]).dt.days / 365.25
    df["draft_pick_inverse"] = 1 / df["draft_pick"]
    return df.drop(columns=["gsis_id", "birth_date", "draft_pick"])


def build_qb_panel(vorp_labels: pd.DataFrame, nfl_players: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
    qb = vorp_labels[vorp_labels["position"] == "QB"][
        ["season", "player_id", "player_display_name", "recent_team", "vorp", "passing_epa"]
    ].copy()
    qb = _add_scarcity_z_and_delta(qb, vorp_labels, "QB")
    qb = _add_age_and_draft_pick_inverse(qb, nfl_players)

    team_stats = team_stats[(team_stats["season"] >= 2008)].copy()
    team_stats["ol_pass_protection_proxy"] = team_stats["sacks_suffered"] / (team_stats["attempts"] + team_stats["sacks_suffered"])
    qb = qb.merge(
        team_stats[["season", "team", "ol_pass_protection_proxy"]],
        left_on=["season", "recent_team"], right_on=["season", "team"], how="left",
    )
    return qb.drop(columns=["team", "recent_team"])


def build_rb_panel(vorp_labels: pd.DataFrame, nfl_players: pd.DataFrame) -> pd.DataFrame:
    rb = vorp_labels[vorp_labels["position"] == "RB"][["season", "player_id", "player_display_name", "vorp"]].copy()
    rb = _add_scarcity_z_and_delta(rb, vorp_labels, "RB")
    return _add_age_and_draft_pick_inverse(rb, nfl_players)


def build_wr_panel(vorp_labels: pd.DataFrame, nfl_players: pd.DataFrame) -> pd.DataFrame:
    wr = vorp_labels[vorp_labels["position"] == "WR"][
        ["season", "player_id", "player_display_name", "vorp", "wopr", "receiving_epa", "air_yards_share", "target_share"]
    ].copy()
    wr = _add_scarcity_z_and_delta(wr, vorp_labels, "WR")
    return _add_age_and_draft_pick_inverse(wr, nfl_players)


def build_te_panel(vorp_labels: pd.DataFrame, nfl_players: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    te = vorp_labels[vorp_labels["position"] == "TE"][
        ["season", "player_id", "player_display_name", "vorp", "wopr", "receiving_epa", "air_yards_share", "target_share"]
    ].copy()
    te = _add_scarcity_z_and_delta(te, vorp_labels, "TE")

    te = te.merge(nfl_players[["gsis_id", "draft_round", "draft_pick"]], left_on="player_id", right_on="gsis_id", how="left")
    te = te.drop(columns=["gsis_id"])
    te["draft_pick_inverse"] = 1 / te["draft_pick"]
    te["draft_tier_Round 2-3"] = te["draft_round"].isin([2, 3]).astype(float)
    te = te.drop(columns=["draft_round", "draft_pick"])

    flagged = injuries[injuries["report_status"].isin(["Questionable", "Doubtful", "Out"])]
    injury_counts = flagged.groupby(["gsis_id", "season"]).size().reset_index(name="injury_designations_count")
    te = te.merge(injury_counts, left_on=["player_id", "season"], right_on=["gsis_id", "season"], how="left")
    te = te.drop(columns=["gsis_id"])
    has_2009plus = te["season"] >= 2009
    te.loc[has_2009plus, "injury_designations_count"] = te.loc[has_2009plus, "injury_designations_count"].fillna(0)
    return te


def compute_flags(delta_value) -> tuple[bool, bool]:
    """Shared, position-agnostic definition of the two confidence flags --
    identical to every 06x_model_*.ipynb notebook's own flag functions."""
    if pd.isna(delta_value):
        return False, True
    return bool(abs(delta_value) > LOW_CONFIDENCE_DELTA_THRESHOLD), False


def predict_kvs(
    player_name: str,
    season: int,
    position: str,
    repo_root: Path,
    vorp_labels: Optional[pd.DataFrame] = None,
) -> tuple[float, bool, bool]:
    """Returns (predicted_kvs, low_confidence_extreme_delta, no_delta_history)
    for a real player+season, using the appropriate tuned model.

    Raises `UnsupportedPositionError` for K or DEF -- no model, and no
    heuristic fallback, is produced for either position (see this
    module's docstring)."""
    import xgboost as xgb
    import nflreadpy as nfl

    if position in UNSUPPORTED_POSITIONS:
        raise UnsupportedPositionError(
            f"Position '{position}' is out of scope for trade evaluation. Per roadmap.md's "
            "Phase 4 scope decision, no XGBoost model or SHAP explainer exists or is planned "
            "for K/DEF -- see notebooks/06_scope_decision_k_def.ipynb for the full reasoning. "
            "This function will not return a predicted_KVS of any kind for this position."
        )
    if position not in MODELED_POSITIONS:
        raise ValueError(f"Unknown position: {position!r}")

    repo_root = Path(repo_root)
    if vorp_labels is None:
        vorp_labels = pd.read_parquet(repo_root / "data/processed/vorp_labels.parquet")

    nfl_players = nfl.load_players().to_pandas()
    if position == "QB":
        team_stats = nfl.load_team_stats(seasons=True, summary_level="reg").to_pandas()
        panel = build_qb_panel(vorp_labels, nfl_players, team_stats)
    elif position == "RB":
        panel = build_rb_panel(vorp_labels, nfl_players)
    elif position == "WR":
        panel = build_wr_panel(vorp_labels, nfl_players)
    else:
        injuries = nfl.load_injuries(seasons=True).to_pandas()
        panel = build_te_panel(vorp_labels, nfl_players, injuries)

    row = panel[(panel["player_display_name"] == player_name) & (panel["season"] == season)]
    if row.empty:
        raise ValueError(f"No feature row for {player_name} ({position}, {season})")
    row = row.iloc[0]

    features = POSITION_FEATURES[position]
    model = xgb.XGBRegressor()
    model.load_model(str(repo_root / "data" / "models" / f"{position.lower()}_model.json"))
    predicted = float(model.predict(pd.DataFrame([row[features]], columns=features))[0])

    low_conf, no_hist = compute_flags(row["vorp_delta_yoy"])
    return predicted, low_conf, no_hist


def evaluate_player_trade_value(
    player_name: str,
    season: int,
    position: str,
    projected_keeper_round: int,
    draft_curve: pd.DataFrame,
    repo_root: Path,
    vorp_labels: Optional[pd.DataFrame] = None,
) -> NetValueResult:
    """End-to-end convenience: predicts KVS, prices the projected keeper
    round via the draft capital curve, and returns the full
    NetValueResult with every applicable caveat attached.

    Raises `UnsupportedPositionError` for K or DEF -- propagated directly
    from `predict_kvs` (see this module's docstring)."""
    predicted_kvs, low_conf, no_hist = predict_kvs(
        player_name, season, position, repo_root, vorp_labels=vorp_labels
    )
    return net_value_from_projected_round(
        player_name, season, position, predicted_kvs, projected_keeper_round, draft_curve,
        low_confidence_extreme_delta=low_conf, no_delta_history=no_hist,
    )
