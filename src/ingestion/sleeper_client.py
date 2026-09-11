"""
Thin wrapper around the Sleeper public API (https://docs.sleeper.com/).
Read-only, no auth required. Sleeper asks that callers stay well under
1000 requests/minute — this client sleeps briefly between calls as a
courtesy, not because it's rate-limited in practice.

NOTE: This module makes live HTTP calls and therefore cannot run inside
a network-sandboxed environment without egress to api.sleeper.app. Run it
somewhere with internet access (your machine, Claude Code, CI, etc.).
"""

import time
import requests

BASE_URL = "https://api.sleeper.app/v1"
_SLEEP_BETWEEN_CALLS = 0.15  # seconds, politeness delay


def _get(path: str) -> dict | list:
    resp = requests.get(f"{BASE_URL}{path}", timeout=15)
    resp.raise_for_status()
    time.sleep(_SLEEP_BETWEEN_CALLS)
    return resp.json()


def get_league(league_id: str) -> dict:
    """Season metadata for one league_id, including previous_league_id
    (the chain link back to last year's league) and draft_id."""
    return _get(f"/league/{league_id}")


def get_league_chain(league_id: str, max_seasons: int = 15) -> list[dict]:
    """
    Sleeper mints a NEW league_id every season and links them via
    `previous_league_id`. Walk that chain backward to assemble full
    multi-year history from a single current league_id.

    Returns list of league dicts, most recent season first.
    """
    chain = []
    current_id = league_id
    for _ in range(max_seasons):
        if not current_id:
            break
        league = get_league(current_id)
        chain.append(league)
        current_id = league.get("previous_league_id")
    return chain


def get_users(league_id: str) -> list[dict]:
    """League members for a given season. user_id is stable across seasons
    and across leagues — this is the identity to key teams on, NOT roster_id."""
    return _get(f"/league/{league_id}/users")


def get_rosters(league_id: str) -> list[dict]:
    """Roster objects for a season. Each has roster_id (season-local, NOT
    stable year over year) and owner_id (the stable user_id to join on)."""
    return _get(f"/league/{league_id}/rosters")


def get_drafts_for_league(league_id: str) -> list[dict]:
    return _get(f"/league/{league_id}/drafts")


def get_draft_picks(draft_id: str) -> list[dict]:
    """Every pick in a draft: player_id, roster_id, round, pick_no, is_keeper."""
    return _get(f"/draft/{draft_id}/picks")


def get_traded_picks(league_id: str) -> list[dict]:
    """Future draft picks that were traded (not the same as player trades —
    see get_transactions for player-for-player trades)."""
    return _get(f"/league/{league_id}/traded_picks")


def get_transactions(league_id: str, week: int) -> list[dict]:
    """Trades/waivers/free-agent moves for a given week. Sleeper requires
    a week parameter; call across weeks 1-18 to get a full season."""
    return _get(f"/league/{league_id}/transactions/{week}")


def get_all_players() -> dict:
    """
    Full NFL player dictionary, keyed by player_id. This is a large
    payload (several MB) and Sleeper explicitly asks callers to use it
    sparingly — at most once a day. Cache the result locally (see
    scripts/sanity_check_2026_keepers.py) rather than calling this on
    every run.
    """
    return _get("/players/nfl")
