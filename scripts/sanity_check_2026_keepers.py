"""
Sanity check — NOT part of the production pipeline.

Shows every 2026 keeper with the player's actual name, which manager kept
them, and what round they cost — so you can eyeball it against what you
actually remember happening in your league before trusting the ledger for
anything downstream.

Run from the repo root:
    python scripts/sanity_check_2026_keepers.py

Requires data/processed/keeper_ledger.parquet to already exist (i.e. you've
already run src/keeper_ledger/build_ledger.py at least once).
"""

import json
import os
import time

import pandas as pd

from src.ingestion import sleeper_client as sc
from src.keeper_ledger.build_ledger import load_keeper_rules

PLAYERS_CACHE_PATH = "data/raw/players_cache.json"
PLAYERS_CACHE_MAX_AGE_SECONDS = 24 * 60 * 60  # Sleeper: call /players/nfl at most once a day


def load_players(force_refresh: bool = False) -> dict:
    """Cached wrapper around sc.get_all_players() — this endpoint is
    several MB and Sleeper asks callers not to hit it repeatedly."""
    cache_is_fresh = (
        os.path.exists(PLAYERS_CACHE_PATH)
        and (time.time() - os.path.getmtime(PLAYERS_CACHE_PATH)) < PLAYERS_CACHE_MAX_AGE_SECONDS
    )
    if cache_is_fresh and not force_refresh:
        with open(PLAYERS_CACHE_PATH) as f:
            return json.load(f)

    print("Fetching full player dictionary from Sleeper (cached for 24h)...")
    players = sc.get_all_players()
    os.makedirs(os.path.dirname(PLAYERS_CACHE_PATH), exist_ok=True)
    with open(PLAYERS_CACHE_PATH, "w") as f:
        json.dump(players, f)
    return players


def load_owner_names(league_id: str) -> dict:
    """owner_id (user_id) -> display name, for THIS season's league_id.
    (For a fully accurate historical name at the time, you'd want to pull
    users per season-specific league_id in the chain — this sanity check
    only needs current names, since you're eyeballing 2026 anyway.)"""
    users = sc.get_users(league_id)
    return {u["user_id"]: u.get("display_name") or u.get("username") for u in users}


def main():
    rules = load_keeper_rules()
    league_id = rules["league_id"]

    ledger = pd.read_parquet("data/processed/keeper_ledger.parquet")
    keepers_2026 = ledger[(ledger["season"] == 2026) & (ledger["is_keeper"])].copy()

    if keepers_2026.empty:
        print("No 2026 keeper rows found in the ledger. Did build_ledger.py run successfully?")
        return

    players = load_players()
    owner_names = load_owner_names(league_id)

    def player_label(player_id: str) -> str:
        p = players.get(str(player_id))
        if not p:
            return f"Unknown player (id {player_id})"
        name = p.get("full_name") or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
        pos = p.get("position", "?")
        team = p.get("team") or "FA"
        return f"{name} ({pos}, {team})"

    keepers_2026["player_name"] = keepers_2026["player_id"].apply(player_label)
    keepers_2026["kept_by"] = keepers_2026["owner_id"].map(owner_names).fillna(keepers_2026["owner_id"])

    display = keepers_2026[[
        "player_name", "kept_by", "round_paid_actual", "original_round", "formula_mismatch"
    ]].rename(columns={
        "round_paid_actual": "round_paid_2026",
        "original_round": "originally_drafted_round",
    }).sort_values("kept_by")

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", None)
    print(f"\n{len(display)} keepers in 2026:\n")
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
