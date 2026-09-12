"""
Roster keeper cost table — every player on every current roster, the
round they were drafted at in 2026, and what it would hypothetically cost
to keep them into 2027.

Run from the repo root (after build_ledger.py has already been run at
least once):
    python -m scripts.project_roster_keeper_costs
"""

import pandas as pd

from src.ingestion import sleeper_client as sc
from src.keeper_ledger.build_ledger import (
    build_player_draft_anchors,
    build_pre_rule_round_history,
    resolve_original_round,
    total_rounds_for_season,
    load_keeper_rules,
)
from scripts.sanity_check_2026_keepers import load_players, load_owner_names

PROJECTION_SEASON = 2027
CURRENT_SEASON = 2026


def main():
    rules = load_keeper_rules()
    league_id = rules["league_id"]
    rounds_lost = rules["escalation"]["rounds_lost"]
    min_round = rules["escalation"]["min_round"]

    draft_history = pd.read_parquet("data/raw/draft_history.parquet")
    ledger_2026 = pd.read_parquet("data/processed/keeper_ledger.parquet")
    ledger_2026 = ledger_2026[ledger_2026["season"] == 2026]
    draft_2026 = draft_history[draft_history["season"] == 2026]

    anchors = build_player_draft_anchors(draft_history)
    pre_rule_rounds = build_pre_rule_round_history(draft_history, rules["escalation"]["rule_effective_season"])
    players = load_players()
    owner_names = load_owner_names(league_id)

    print("Fetching current rosters...")
    rosters = sc.get_rosters(league_id)

    rows = []
    for roster in rosters:
        owner_id = roster.get("owner_id")
        if owner_id is None:
            continue
        owner_name = owner_names.get(owner_id, owner_id)

        for player_id in (roster.get("players") or []):
            # Was this player actually part of the 2026 draft for this
            # owner (fresh or keeper)? If not, they were added via
            # in-season waiver after the draft closed.
            drafted_2026 = draft_2026[
                (draft_2026["owner_id"] == owner_id) & (draft_2026["player_id"] == player_id)
            ]
            drafted_2026_round = int(drafted_2026.iloc[0]["round"]) if not drafted_2026.empty else None

            # How many rule-era years running has this owner already kept
            # this player, as of 2026? 0 if not currently a keeper —
            # meaning 2027 would be their first rule-era keep.
            ledger_row = ledger_2026[
                (ledger_2026["owner_id"] == owner_id) & (ledger_2026["player_id"] == player_id)
            ]
            keeps_since_rule_start_2026 = (
                int(ledger_row.iloc[0]["keeps_since_rule_start"])
                if not ledger_row.empty and ledger_row.iloc[0]["is_keeper"]
                else 0
            )

            has_fresh_anchor = not anchors[anchors["player_id"] == player_id].empty

            if drafted_2026_round is None:
                # CONFIRMED with commissioner: a genuine in-season waiver
                # add THIS YEAR (no 2026 draft row at all — never occupied
                # a live draft slot) is treated as round 18 for 2026, full
                # stop, regardless of any unrelated draft history this
                # player_id might have under a different owner in a past
                # season. There's nothing to explain/validate here (unlike
                # Rice/Irving/Watson, who all DO have a real 2026 draft
                # row whose round the league-wide anchor accounts for) —
                # this is purely a hypothetical projection from scratch.
                original_round = total_rounds_for_season(CURRENT_SEASON, rules)
                source_note = f"in-season waiver add — treated as round {original_round} for {CURRENT_SEASON}"
            else:
                original_round = resolve_original_round(anchors, pre_rule_rounds, player_id, PROJECTION_SEASON, rules)
                source_note = "" if has_fresh_anchor else "no fresh-draft record — used pre-rule/last-round fallback"

            projected_round = max(
                min_round, original_round - rounds_lost * (keeps_since_rule_start_2026 + 1)
            )

            p = players.get(str(player_id), {})
            player_name = (
                p.get("full_name")
                or f"{p.get('first_name','')} {p.get('last_name','')}".strip()
                or f"id {player_id}"
            )

            rows.append({
                "owner": owner_name,
                "player": player_name,
                "position": p.get("position", "?"),
                "drafted_2026_round": drafted_2026_round if drafted_2026_round is not None else "waiver add (no 2026 draft pick)",
                f"round_lost_if_kept_{PROJECTION_SEASON}": projected_round,
                "note": source_note,
            })

    report = pd.DataFrame(rows).sort_values(["owner", "player"])

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", None)
    print(f"\n{len(report)} rostered players across {report['owner'].nunique()} teams:\n")
    print(report.to_string(index=False))

    report.to_csv("data/processed/roster_keeper_table_2027.csv", index=False)
    print("\nSaved to data/processed/roster_keeper_table_2027.csv")


if __name__ == "__main__":
    main()
