"""
Trade Engine -- Evaluate Trade. The final Phase 6 integration point.

Ties every prior piece together into one call: `net_value.py`'s per-player
predicted-KVS-minus-keeper-cost, `positional_need.py`'s roster-construction
adjustment, `fairness_score.py`'s percentage-of-total verdict, and
`src/explain/explain_player.py`'s real SHAP top-5-drivers for whichever
player(s) actually drove the result -- not a hand-wavy "this trade is
62/38" with no way to see why.

PIPELINE, PER PLAYER:
    1. `evaluate_player_trade_value` -- predicted_kvs, keeper_cost_vorp,
       raw net_kvs_delta, plus that player's own confidence caveats
       (`low_confidence_extreme_delta` / `no_delta_history`).
    2. `apply_positional_need_adjustment` -- adjusts that raw delta using
       the RECEIVING team's own roster (a player moving to team A is
       adjusted against `team_a_roster`, not `team_b_roster`), plus its
       own `empty_position` caveat when relevant.

Both steps' caveats are attached to that player's `PlayerTradeContribution`
AND rolled up into the final result's `caveats` list, tagged with which
side and which player they belong to -- never dropped, and never
anonymized into a pile that can't be traced back to the player that
produced them (the same standard `net_value.py` and `positional_need.py`
already hold themselves to individually).

THEN, PER SIDE:
    3. Sum that side's adjusted net values into one total.
    4. `compute_fairness_score(team_a_total, team_b_total)` for the
       headline verdict.
    5. The `top_n_explained` (default 2) players on EACH side with the
       largest |adjusted_net_kvs_delta| -- i.e. whoever is actually
       driving that side's total, whether the number is unusually good
       or unusually bad for them -- get a real `explain_player` call, so
       the verdict comes with "why", not just the number.

WHY EXPLANATION FAILURES DON'T ABORT THE WHOLE TRADE EVALUATION: the
fairness score and every player's adjusted value are the load-bearing
result; `explain_player`'s SHAP breakdown is an enrichment on top of an
already-complete answer, not a dependency of it. If it raises for a
top-mover (an edge case `predict_kvs` didn't also hit, since both look up
the same panel row), that failure is caught and recorded verbatim on that
player's `ExplainedDriver.explanation_error` -- surfaced explicitly, never
silently swallowed -- rather than raising out of `evaluate_trade` and
discarding a verdict that was otherwise fully computed.

`evaluate_player_fn`/`explain_player_fn` are injectable (default to the
real `evaluate_player_trade_value`/`explain_player`) specifically so the
orchestration logic above -- aggregation, top-mover selection, caveat
rollup -- is unit-testable with synthetic results, without needing real
models/SHAP/nflreadpy for every test. Same reasoning `net_value.py` uses
to keep `evaluate_net_value` pure while `predict_kvs` stays live-data-only;
here the "pure" part is the orchestration, not the arithmetic, so
dependency injection is the mechanism instead of a pure/live function
split.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from src.explain.explain_player import PlayerExplanation, explain_player
from src.trade_engine.fairness_score import FairnessScoreResult, compute_fairness_score
from src.trade_engine.net_value import evaluate_player_trade_value, position_std_vorp_that_season
from src.trade_engine.positional_need import NEED_ADJUSTMENT_WEIGHT, apply_positional_need_adjustment

DEFAULT_TOP_N_EXPLAINED = 2


@dataclass
class TradeAsset:
    """One player moving in the trade, and the keeper round it costs to
    keep it (needed to price `keeper_cost_vorp` via the draft curve)."""
    player_name: str
    season: int
    position: str
    projected_keeper_round: int


@dataclass
class AttributedCaveat:
    """A caveat traceable back to exactly which side and player produced
    it -- see this module's docstring for why that traceability matters."""
    side: str  # "team_a" or "team_b"
    player_name: str
    caveat: str


@dataclass
class PlayerTradeContribution:
    player_name: str
    season: int
    position: str
    predicted_kvs: float
    keeper_cost_vorp: float
    raw_net_kvs_delta: float
    team_need_z: float
    adjustment_vorp: float
    adjusted_net_kvs_delta: float
    caveats: list = field(default_factory=list)


@dataclass
class ExplainedDriver:
    side: str  # "team_a" or "team_b"
    player_name: str
    season: int
    position: str
    adjusted_net_kvs_delta: float
    top_drivers: list = field(default_factory=list)  # list[FeatureDriver]
    explanation_error: Optional[str] = None


@dataclass
class TradeEvaluationResult:
    team_a_players: list  # list[PlayerTradeContribution]
    team_b_players: list  # list[PlayerTradeContribution]
    team_a_total: float
    team_b_total: float
    fairness: FairnessScoreResult
    team_a_explained_drivers: list  # list[ExplainedDriver]
    team_b_explained_drivers: list  # list[ExplainedDriver]
    caveats: list = field(default_factory=list)  # list[AttributedCaveat]


def _evaluate_side(
    assets: list,
    roster: list,
    side: str,
    draft_curve: pd.DataFrame,
    repo_root: Path,
    vorp_labels: pd.DataFrame,
    need_weight: float,
    evaluate_player_fn: Callable,
) -> tuple:
    contributions = []
    attributed_caveats = []

    for asset in assets:
        net_result = evaluate_player_fn(
            asset.player_name, asset.season, asset.position,
            asset.projected_keeper_round, draft_curve, repo_root,
            vorp_labels=vorp_labels,
        )
        std_vorp = position_std_vorp_that_season(vorp_labels, asset.position, asset.season)
        need_result = apply_positional_need_adjustment(
            net_result.net_kvs_delta, roster, asset.position, std_vorp, weight=need_weight,
        )

        caveats = list(net_result.caveats) + list(need_result.caveats)
        contributions.append(PlayerTradeContribution(
            player_name=asset.player_name,
            season=asset.season,
            position=asset.position,
            predicted_kvs=net_result.predicted_kvs,
            keeper_cost_vorp=net_result.keeper_cost_vorp,
            raw_net_kvs_delta=net_result.net_kvs_delta,
            team_need_z=need_result.team_need_z,
            adjustment_vorp=need_result.adjustment_vorp,
            adjusted_net_kvs_delta=need_result.adjusted_net_kvs_delta,
            caveats=caveats,
        ))
        for caveat in caveats:
            attributed_caveats.append(AttributedCaveat(side=side, player_name=asset.player_name, caveat=caveat))

    return contributions, attributed_caveats


def _select_top_movers(contributions: list, n: int) -> list:
    """Whoever is actually driving this side's total -- largest absolute
    contribution, whether that contribution is unusually good or
    unusually bad, not just the largest positive one."""
    return sorted(contributions, key=lambda c: abs(c.adjusted_net_kvs_delta), reverse=True)[:n]


def _explain_one(
    contribution: PlayerTradeContribution,
    side: str,
    repo_root: Path,
    vorp_labels: pd.DataFrame,
    explain_player_fn: Callable,
) -> ExplainedDriver:
    try:
        explanation: PlayerExplanation = explain_player_fn(
            contribution.player_name, contribution.season, contribution.position, repo_root,
            vorp_labels=vorp_labels,
        )
        return ExplainedDriver(
            side=side,
            player_name=contribution.player_name,
            season=contribution.season,
            position=contribution.position,
            adjusted_net_kvs_delta=contribution.adjusted_net_kvs_delta,
            top_drivers=explanation.top_drivers,
        )
    except Exception as exc:  # noqa: BLE001 -- deliberate: see module docstring
        return ExplainedDriver(
            side=side,
            player_name=contribution.player_name,
            season=contribution.season,
            position=contribution.position,
            adjusted_net_kvs_delta=contribution.adjusted_net_kvs_delta,
            top_drivers=[],
            explanation_error=str(exc),
        )


def evaluate_trade(
    team_a_players: list,
    team_b_players: list,
    team_a_roster: list,
    team_b_roster: list,
    draft_curve: pd.DataFrame,
    repo_root: Path,
    *,
    vorp_labels: Optional[pd.DataFrame] = None,
    top_n_explained: int = DEFAULT_TOP_N_EXPLAINED,
    need_weight: float = NEED_ADJUSTMENT_WEIGHT,
    evaluate_player_fn: Callable = evaluate_player_trade_value,
    explain_player_fn: Callable = explain_player,
) -> TradeEvaluationResult:
    """`team_a_players`/`team_b_players` are the `TradeAsset`s moving TO
    that side (i.e. `team_a_players` is what team A receives, formerly
    team B's) -- matching `fairness_score.py`'s "value received" framing
    and `positional_need.py`'s need adjustment (applied against the
    RECEIVING team's own existing roster, `team_a_roster`/`team_b_roster`,
    each a list of `{"position": ..., "scarcity_z": ...}` dicts).

    Raises whatever `evaluate_player_trade_value` raises (including
    `UnsupportedPositionError` for K/DEF) -- a player this engine cannot
    evaluate at all is not something a fairness verdict can quietly work
    around; that failure belongs at this boundary, same as it already
    does inside `net_value.py` itself.
    """
    repo_root = Path(repo_root)
    if vorp_labels is None:
        vorp_labels = pd.read_parquet(repo_root / "data/processed/vorp_labels.parquet")

    team_a_contributions, team_a_caveats = _evaluate_side(
        team_a_players, team_a_roster, "team_a", draft_curve, repo_root, vorp_labels, need_weight, evaluate_player_fn,
    )
    team_b_contributions, team_b_caveats = _evaluate_side(
        team_b_players, team_b_roster, "team_b", draft_curve, repo_root, vorp_labels, need_weight, evaluate_player_fn,
    )

    team_a_total = sum(c.adjusted_net_kvs_delta for c in team_a_contributions)
    team_b_total = sum(c.adjusted_net_kvs_delta for c in team_b_contributions)
    fairness = compute_fairness_score(team_a_total, team_b_total)

    team_a_top = _select_top_movers(team_a_contributions, min(top_n_explained, len(team_a_contributions)))
    team_b_top = _select_top_movers(team_b_contributions, min(top_n_explained, len(team_b_contributions)))

    team_a_explained = [_explain_one(c, "team_a", repo_root, vorp_labels, explain_player_fn) for c in team_a_top]
    team_b_explained = [_explain_one(c, "team_b", repo_root, vorp_labels, explain_player_fn) for c in team_b_top]

    return TradeEvaluationResult(
        team_a_players=team_a_contributions,
        team_b_players=team_b_contributions,
        team_a_total=team_a_total,
        team_b_total=team_b_total,
        fairness=fairness,
        team_a_explained_drivers=team_a_explained,
        team_b_explained_drivers=team_b_explained,
        caveats=team_a_caveats + team_b_caveats,
    )
