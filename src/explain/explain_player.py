"""
Explain Player -- Phase 5's reusable SHAP explanation function, actually
made reusable.

Phase 5's own exit criterion (`roadmap.md`) called for
`explain_player(player, season) -> top_5_drivers` as a reusable function.
What was actually built was FOUR byte-identical copies of that function,
one pasted into each of `notebooks/08a-d_shap_*.ipynb`, differing only in
which position's model/panel they closed over -- never promoted into an
importable module, so nothing outside a notebook could call it.
`src/explain/` existed in the repo the whole time as an empty, unused
placeholder. This module is that promotion, done once `evaluate_trade.py`
(Phase 6) needed a real, callable `explain_player` and the gap became a
blocker rather than a cosmetic one.

This does NOT re-derive the QB/RB/WR/TE feature panels a second way --
it reuses `net_value.py`'s own `build_qb_panel`/`build_rb_panel`/
`build_wr_panel`/`build_te_panel` and `POSITION_FEATURES`, the exact
construction `predict_kvs` already depends on for the SAME player-season
row. Re-deriving the panel independently here (as each notebook did) is
exactly the kind of duplicated logic that drifts silently over time.

Like `net_value.py`'s `predict_kvs`, this is a live function -- it loads
a real XGBoost model, builds a real `shap.TreeExplainer`, and pulls
`nflreadpy` data -- not unit tested directly. Exercised for real in
`notebooks/10_evaluate_trade_demo.ipynb`.

K and DEF raise the same `UnsupportedPositionError` `net_value.py` raises
for them, for the same reason: no model, and therefore no SHAP explainer,
exists or is planned for either position.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.trade_engine.net_value import (
    MODELED_POSITIONS,
    POSITION_FEATURES,
    UNSUPPORTED_POSITIONS,
    UnsupportedPositionError,
    build_qb_panel,
    build_rb_panel,
    build_te_panel,
    build_wr_panel,
)

DEFAULT_TOP_N = 5


@dataclass
class FeatureDriver:
    feature: str
    feature_value: float
    shap_value: float
    direction: str  # "increases" or "decreases"


@dataclass
class PlayerExplanation:
    player_name: str
    season: int
    position: str
    predicted_vorp: float
    base_value: float
    top_drivers: list = field(default_factory=list)  # list[FeatureDriver]


def _build_panel(position: str, vorp_labels: pd.DataFrame, nfl_players: pd.DataFrame, team_stats, injuries) -> pd.DataFrame:
    if position == "QB":
        return build_qb_panel(vorp_labels, nfl_players, team_stats)
    if position == "RB":
        return build_rb_panel(vorp_labels, nfl_players)
    if position == "WR":
        return build_wr_panel(vorp_labels, nfl_players)
    return build_te_panel(vorp_labels, nfl_players, injuries)


def explain_player(
    player_name: str,
    season: int,
    position: str,
    repo_root: Path,
    *,
    top_n: int = DEFAULT_TOP_N,
    vorp_labels: Optional[pd.DataFrame] = None,
) -> PlayerExplanation:
    """Returns the top `top_n` features driving that position's tuned
    model's prediction for one player-season, ranked by absolute SHAP
    contribution, with signed value and direction -- the same output the
    four notebook copies of this function produced, now callable from
    anywhere (`evaluate_trade.py`, in particular).

    Raises `UnsupportedPositionError` for K/DEF, same as `predict_kvs`.
    """
    import shap
    import xgboost as xgb
    import nflreadpy as nfl

    if position in UNSUPPORTED_POSITIONS:
        raise UnsupportedPositionError(
            f"Position '{position}' is out of scope for trade evaluation. Per roadmap.md's "
            "Phase 4 scope decision, no XGBoost model or SHAP explainer exists or is planned "
            "for K/DEF -- see notebooks/06_scope_decision_k_def.ipynb for the full reasoning. "
            "This function will not return an explanation of any kind for this position."
        )
    if position not in MODELED_POSITIONS:
        raise ValueError(f"Unknown position: {position!r}")

    repo_root = Path(repo_root)
    if vorp_labels is None:
        vorp_labels = pd.read_parquet(repo_root / "data/processed/vorp_labels.parquet")

    nfl_players = nfl.load_players().to_pandas()
    team_stats = nfl.load_team_stats(seasons=True, summary_level="reg").to_pandas() if position == "QB" else None
    injuries = nfl.load_injuries(seasons=True).to_pandas() if position == "TE" else None

    panel = _build_panel(position, vorp_labels, nfl_players, team_stats, injuries)
    row_match = panel[(panel["player_display_name"] == player_name) & (panel["season"] == season)]
    if row_match.empty:
        raise ValueError(f"No feature row for {player_name} ({position}, {season})")

    features = POSITION_FEATURES[position]
    row = row_match[features].iloc[0]
    row_df = pd.DataFrame([row], columns=features)

    model = xgb.XGBRegressor()
    model.load_model(str(repo_root / "data" / "models" / f"{position.lower()}_model.json"))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(row_df)
    prediction = float(model.predict(row_df)[0])

    contrib = pd.DataFrame({
        "feature": features,
        "feature_value": row.to_numpy(),
        "shap_value": shap_values.values[0],
    })
    contrib["direction"] = np.where(contrib["shap_value"] >= 0, "increases", "decreases")
    contrib["abs_shap"] = contrib["shap_value"].abs()
    top = contrib.sort_values("abs_shap", ascending=False).head(top_n)

    top_drivers = [
        FeatureDriver(
            feature=r.feature,
            feature_value=float(r.feature_value),
            shap_value=float(r.shap_value),
            direction=r.direction,
        )
        for r in top.itertuples()
    ]

    return PlayerExplanation(
        player_name=player_name,
        season=season,
        position=position,
        predicted_vorp=prediction,
        base_value=float(explainer.expected_value),
        top_drivers=top_drivers,
    )
