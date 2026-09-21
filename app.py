"""
Streamlit Trade Evaluation App -- Phase 8, Part 1.

The interactive front end for `src/trade_engine/evaluate_trade.py`:
pick a team and players on each side of a proposed trade, see the real
fairness verdict, per-side net value, every caveat a player's number
carries, and the SHAP drivers behind each side's biggest movers --
then the trade is logged to a local SQLite history
(`src/trade_engine/trade_log.py`).

Reads only the repo's existing static processed data
(`data/processed/roster_keeper_table_2027.csv`, `vorp_labels.parquet`)
plus the same real XGBoost models/SHAP/nflreadpy calls
`evaluate_trade.py` already makes -- no live Sleeper API calls, and no
new modeling/feature-engineering code. SQLite is the deliberate v1
persistence choice (see `trade_log.py`'s module docstring).

Run with: streamlit run app.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.trade_engine.config import MODEL_VERSION
from src.trade_engine.draft_capital_curve import build_full_draft_capital_curve, normalize_player_name
from src.trade_engine.evaluate_trade import TradeAsset, evaluate_trade
from src.trade_engine.fairness_score import describe_fairness
from src.trade_engine.net_value import UnsupportedPositionError
from src.trade_engine.trade_log import build_trade_log_row, get_trade_history, log_trade

REPO_ROOT = Path(__file__).resolve().parent
ROSTER_TABLE_PATH = REPO_ROOT / "data/processed/roster_keeper_table_2027.csv"
VORP_LABELS_PATH = REPO_ROOT / "data/processed/vorp_labels.parquet"

# Matches notebooks/10_evaluate_trade_demo.ipynb's own convention: each
# player's real, already-realized SEASON feeds the next-season KVS
# prediction, and `roster_keeper_table_2027.csv` prices what it would
# cost to keep that player past SEASON.
SEASON = 2025


# ---------------------------------------------------------------------------
# Startup -- loaded once, not on every interaction.
# ---------------------------------------------------------------------------

def _build_roster_with_scarcity_z(owner: str, roster_table: pd.DataFrame, v2025: pd.DataFrame,
                                   def_rows: pd.DataFrame, def_mean: float, def_std: float,
                                   team_name_to_code: dict) -> list:
    """The receiving-team roster shape `positional_need.py` expects:
    `[{"position": ..., "scarcity_z": ...}, ...]`. `roster_keeper_table_2027.csv`
    only has player names, so each player is joined against `vorp_labels`'
    SEASON rows by normalized name (DEF rows by team code instead, since
    `vorp_labels` keys defenses by team code, not full team name).
    Unmatched players (e.g. rookies with no `vorp_labels` row yet) are
    skipped rather than guessed at -- same convention
    notebooks/10_evaluate_trade_demo.ipynb already uses."""
    roster = []
    for _, row in roster_table[roster_table["owner"] == owner].iterrows():
        position, player = row["position"], row["player"]
        if position == "DEF":
            code = team_name_to_code.get(player)
            match = def_rows[def_rows["player_id"] == code] if code else pd.DataFrame()
            if match.empty:
                continue
            z = (match["vorp"].iloc[0] - def_mean) / def_std
            roster.append({"position": "DEF", "scarcity_z": float(z)})
            continue
        norm_name = normalize_player_name(player)
        match = v2025[(v2025["position"] == position) & (v2025["norm_name"] == norm_name)]
        if match.empty:
            continue
        roster.append({"position": position, "scarcity_z": float(match["scarcity_z"].iloc[0])})
    return roster


@st.cache_resource(show_spinner="Loading league data and draft capital curve...")
def load_league_data():
    """Everything expensive enough to matter -- the draft capital curve
    (real nflreadpy pulls) and every team's need-adjustment roster (also
    nflreadpy, for the DEF team-code lookup) -- built once per server
    process, not once per button click. The XGBoost model JSON files
    themselves are small and already loaded fresh, per call, inside
    `evaluate_player_trade_value`/`explain_player` -- there is no
    separate "load the 4 model artifacts" step to hoist out here without
    changing those functions' own signatures, which is out of scope for
    this app."""
    import nflreadpy as nfl

    draft_curve = build_full_draft_capital_curve(REPO_ROOT)
    vorp_labels = pd.read_parquet(VORP_LABELS_PATH)
    roster_table = pd.read_csv(ROSTER_TABLE_PATH)

    v2025 = vorp_labels[vorp_labels["season"] == SEASON][["position", "player_display_name", "vorp"]].copy()
    v2025["norm_name"] = v2025["player_display_name"].apply(normalize_player_name)
    position_stats = v2025.groupby("position")["vorp"].agg(mean="mean", std="std")
    v2025 = v2025.merge(position_stats, on="position", how="left")
    v2025["scarcity_z"] = (v2025["vorp"] - v2025["mean"]) / v2025["std"]

    teams = nfl.load_teams().to_pandas()
    team_name_to_code = dict(zip(teams["team_name"], teams["team_abbr"]))
    def_rows = vorp_labels[(vorp_labels["season"] == SEASON) & (vorp_labels["position"] == "DEF")][["player_id", "vorp"]]
    def_mean = v2025.loc[v2025["position"] == "DEF", "mean"].iloc[0]
    def_std = v2025.loc[v2025["position"] == "DEF", "std"].iloc[0]

    team_names = sorted(roster_table["owner"].unique().tolist())
    rosters_by_team = {
        owner: _build_roster_with_scarcity_z(owner, roster_table, v2025, def_rows, def_mean, def_std, team_name_to_code)
        for owner in team_names
    }

    return {
        "draft_curve": draft_curve,
        "vorp_labels": vorp_labels,
        "roster_table": roster_table,
        "team_names": team_names,
        "rosters_by_team": rosters_by_team,
    }


def _team_players(roster_table: pd.DataFrame, owner: str) -> pd.DataFrame:
    return roster_table[roster_table["owner"] == owner][["player", "position", "round_lost_if_kept_2027"]]


def _make_trade_assets(selected_players: list, team_players: pd.DataFrame) -> list:
    assets = []
    for label in selected_players:
        player_name = label.rsplit(" (", 1)[0]
        row = team_players[team_players["player"] == player_name].iloc[0]
        assets.append(TradeAsset(
            player_name=player_name,
            season=SEASON,
            position=row["position"],
            projected_keeper_round=int(row["round_lost_if_kept_2027"]),
        ))
    return assets


# ---------------------------------------------------------------------------
# Results rendering
# ---------------------------------------------------------------------------

def _render_caveats(result, team_a_name: str, team_b_name: str):
    if not result.caveats:
        return
    st.subheader("Caveats")
    side_names = {"team_a": team_a_name, "team_b": team_b_name}
    for c in result.caveats:
        st.warning(f"**{side_names[c.side]} -- {c.player_name}**: {c.caveat}")


def _render_side(result, side_name: str, contributions, explained_drivers, total: float):
    st.markdown(f"#### {side_name} receives")
    st.metric(f"{side_name} net KVS delta", f"{total:+.2f}")

    if contributions:
        table = pd.DataFrame([{
            "Player": p.player_name,
            "Pos": p.position,
            "Predicted KVS": round(p.predicted_kvs, 2),
            "Keeper cost": round(p.keeper_cost_vorp, 2),
            "Need adj.": round(p.adjustment_vorp, 2),
            "Adjusted delta": round(p.adjusted_net_kvs_delta, 2),
        } for p in contributions])
        st.dataframe(table, hide_index=True, width="stretch")

    for driver in explained_drivers:
        st.markdown(f"**Why {driver.player_name}'s number looks the way it does:**")
        if driver.explanation_error:
            st.caption(f"SHAP explanation unavailable: {driver.explanation_error}")
            continue
        top = driver.top_drivers[:2]
        if not top:
            continue
        sentence = "; ".join(
            f"**{d.feature}** ({d.feature_value:.2f}) {d.direction} it by {abs(d.shap_value):.2f}"
            for d in top
        )
        st.write(sentence)
        chart_df = pd.DataFrame({
            "feature": [d.feature for d in top],
            "shap_value": [d.shap_value for d in top],
        }).set_index("feature")
        st.bar_chart(chart_df, width="stretch")


def render_results(result, team_a_name: str, team_b_name: str):
    st.subheader("Fairness verdict")
    st.success(describe_fairness(result.fairness, team_a_name=team_a_name))
    if result.fairness.shifted:
        st.caption("Values were shifted to a common non-negative floor before splitting (one side was negative).")
    if result.fairness.fairness_method == "magnitude_relative_both_negative":
        st.caption("Both sides net negative in isolation -- split by relative magnitude of loss, not percentage-of-total.")

    col_a, col_b = st.columns(2)
    with col_a:
        _render_side(result, team_a_name, result.team_a_players, result.team_a_explained_drivers, result.team_a_total)
    with col_b:
        _render_side(result, team_b_name, result.team_b_players, result.team_b_explained_drivers, result.team_b_total)

    _render_caveats(result, team_a_name, team_b_name)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def evaluate_trade_page(data: dict):
    team_names = data["team_names"]
    roster_table = data["roster_table"]
    rosters_by_team = data["rosters_by_team"]

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Team A")
        team_a = st.selectbox("Team A", team_names, key="team_a_select")
        options_a = [f"{r.player} ({r.position})" for r in _team_players(roster_table, team_a).itertuples()]
        players_a = st.multiselect("Team A sends", options_a, key="players_a_select")

    with col_b:
        st.markdown("### Team B")
        remaining_teams = [t for t in team_names if t != team_a]
        team_b = st.selectbox("Team B", remaining_teams, key="team_b_select")
        options_b = [f"{r.player} ({r.position})" for r in _team_players(roster_table, team_b).itertuples()]
        players_b = st.multiselect("Team B sends", options_b, key="players_b_select")

    can_evaluate = len(players_a) > 0 and len(players_b) > 0
    if st.button("Evaluate Trade", disabled=not can_evaluate, type="primary"):
        team_a_players_moving = _make_trade_assets(players_b, _team_players(roster_table, team_b))  # B's players move TO A
        team_b_players_moving = _make_trade_assets(players_a, _team_players(roster_table, team_a))  # A's players move TO B

        try:
            with st.spinner("Evaluating trade..."):
                result = evaluate_trade(
                    team_a_players_moving, team_b_players_moving,
                    rosters_by_team[team_a], rosters_by_team[team_b],
                    data["draft_curve"], REPO_ROOT, vorp_labels=data["vorp_labels"],
                )
        except UnsupportedPositionError:
            st.error(
                "This trade includes a K or DEF player. Kickers and defenses aren't modeled "
                "(no predicted value, no SHAP explanation) -- remove them from the trade to evaluate it."
            )
            return

        render_results(result, team_a, team_b)

        row = build_trade_log_row(team_a, team_b, result, MODEL_VERSION)
        log_trade(row)
        st.caption(f"Logged to trade history (model version {MODEL_VERSION}).")

    _render_how_to_read()


def _render_how_to_read():
    with st.expander("How to read this", expanded=True):
        st.markdown(
            "**Predicted KVS** — the model's raw projection of how much value this player "
            "will produce next season, before anything else is factored in.\n"
            "- Higher # = the model expects more production from this player.\n\n"
            "**Keeper cost** — what it costs you in draft capital to keep this player next year.\n"
            "- Higher # = it costs you more to keep them (a better draft pick). This isn't a "
            "knock on the player — a great player naturally has a high keeper cost.\n\n"
            "**Need adj.** — an adjustment based on how deep your team already is at that position.\n"
            "- Higher # = your team is thin there, so this player helps you more.\n"
            "- Lower # = your team is already deep there, so one more player at that spot helps less.\n\n"
            "**Adjusted delta** — the bottom-line number: is this player worth more than what it "
            "costs to keep them, for your specific team. This is the number that matters most.\n"
            "- Higher # (closer to zero or positive) = better value for that side of the trade.\n"
            "- Lower # (more negative) = worse value for that side of the trade.\n\n"
            "**Fairness %** — how evenly the trade's total value is split between both sides.\n"
            "- Closer to 50% = an even trade.\n"
            "- Further from 50% = more lopsided — whoever's above 50% got the better end of it.\n\n"
            "One important note: these numbers are based on each player's full 2025 season, not "
            "anything that's happened in 2026 so far (including this week). This tool tells you "
            "if a trade is fair based on who a player was heading into this year — not how "
            "they're playing right now."
        )


def history_page():
    history = get_trade_history()
    if history.empty:
        st.info("No trades evaluated yet.")
        return

    display = history.copy()
    display["team_a_players"] = display["team_a_players"].apply(lambda s: ", ".join(json.loads(s)))
    display["team_b_players"] = display["team_b_players"].apply(lambda s: ", ".join(json.loads(s)))
    display["fairness_score"] = display["fairness_score"].round(1)
    display["net_kvs_delta_a"] = display["net_kvs_delta_a"].round(2)
    display["net_kvs_delta_b"] = display["net_kvs_delta_b"].round(2)
    st.dataframe(
        display[[
            "timestamp", "team_a", "team_a_players", "team_b", "team_b_players",
            "fairness_score", "fairness_method", "net_kvs_delta_a", "net_kvs_delta_b", "model_version",
        ]],
        hide_index=True, width="stretch",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Keeper Trade Evaluator", layout="wide")
st.title("Keeper Trade Evaluator")

data = load_league_data()

tab_evaluate, tab_history = st.tabs(["Evaluate Trade", "History"])
with tab_evaluate:
    evaluate_trade_page(data)
with tab_history:
    history_page()
