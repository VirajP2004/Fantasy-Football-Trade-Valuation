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


def get_matchups(league_id: str, week: int) -> list[dict]:
    """
    Actual starting lineups for a given week: one entry per roster, with
    a `starters` list of player_ids in the SAME ORDER as the league's
    `roster_positions` (excluding BN/taxi/IR slots). Zip the two together
    to know which slot (QB/RB/WR/TE/FLEX/K/DEF) each starter actually
    filled that week — this is what lets you measure the real FLEX
    position split instead of assuming one.
    """
    return _get(f"/league/{league_id}/matchups/{week}")


def get_stats_week(season: int, week: int) -> dict:
    """
    Raw weekly stats for every player AND team unit, keyed by id. This
    endpoint mixes several different kinds of entries under one dict —
    numeric player_ids, `TEAM_<ABBR>` entries (that team's own aggregate
    OFFENSIVE box score — pass_yd, rush_yd, etc., NOT defense), and plain
    `<ABBR>` entries (that team's actual DEFENSE/special-teams stat line).

    This wrapper filters down to ONLY the plain team-abbreviation keys —
    real team defense — dropping every numeric player_id and every
    `TEAM_`-prefixed entry. Filtered by PATTERN (non-numeric, not
    `TEAM_`-prefixed), not a hardcoded list of current team abbreviations
    — Sleeper uses ERA-ACCURATE codes for relocated/renamed franchises
    (OAK not LV pre-2020, SD not LAC pre-2017, STL not LAR pre-2016), so
    a hardcoded current-team allowlist would silently drop those
    franchises' historical seasons. Confirmed clean across 2009/2015/
    2020/2025: exactly 32 real team codes every time, nothing stray.

    Also confirmed via a real example: `TEAM_DET`'s "td" and plain
    `DET`'s "td" were identical (both mirroring the team's total
    OFFENSIVE touchdowns) — "td" on the plain entry is NOT a
    defense-only stat, so don't use it for defensive scoring; sum the
    specific sub-fields instead (def_td, def_pr_td, def_kr_td,
    blk_kick_ret_td, blk_pr_td, def_st_td).

    Returns {} for a season/week this endpoint has no data for (e.g.
    2008 and earlier — confirmed empty across every week checked) rather
    than raising, since an empty dict is a legitimate "no data" signal
    here, not an error.
    """
    data = _get(f"/stats/nfl/regular/{season}/{week}")
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if not k.isdigit() and not k.startswith("TEAM_")}
