"""
Trade Engine -- Trade Log.

Persists every completed `evaluate_trade` verdict the Streamlit app
(`app.py`) produces, so past trades stay auditable instead of vanishing
the moment a browser tab closes. SQLite is the deliberate v1 choice --
it keeps this shippable without a new infra decision (Supabase or
otherwise) getting made mid-build; see `roadmap.md`'s Phase 8 section.

Split the same way `net_value.py`/`draft_capital_curve.py` already
split pure orchestration from live I/O: `build_trade_log_row` is pure
and fully unit-tested (no SQLite, no disk) -- it turns one
`TradeEvaluationResult` plus the two team names into a single flat,
JSON-serializable row. `log_trade`/`get_trade_history` are the
SQLite-backed half that actually persists/reads it; not unit tested
directly, exercised for real by `app.py`.

`model_version` is never computed in here -- callers pass
`src/trade_engine/config.py`'s `MODEL_VERSION` through explicitly, so
this module has no hardcoded knowledge of which model generation
produced any given verdict.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src.trade_engine.evaluate_trade import TradeEvaluationResult

DEFAULT_DB_PATH = Path("data/processed/trade_log.sqlite")

TRADE_LOG_COLUMNS = [
    "trade_id",
    "timestamp",
    "team_a",
    "team_a_players",
    "team_b",
    "team_b_players",
    "fairness_score",
    "fairness_method",
    "net_kvs_delta_a",
    "net_kvs_delta_b",
    "caveats",
    "model_version",
]

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS trade_log (
    trade_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    team_a TEXT NOT NULL,
    team_a_players TEXT NOT NULL,
    team_b TEXT NOT NULL,
    team_b_players TEXT NOT NULL,
    fairness_score REAL NOT NULL,
    fairness_method TEXT NOT NULL,
    net_kvs_delta_a REAL NOT NULL,
    net_kvs_delta_b REAL NOT NULL,
    caveats TEXT NOT NULL,
    model_version TEXT NOT NULL
);
"""

_INSERT_SQL = f"""
INSERT INTO trade_log ({", ".join(TRADE_LOG_COLUMNS)})
VALUES ({", ".join(f":{c}" for c in TRADE_LOG_COLUMNS)})
"""


# ---------------------------------------------------------------------------
# Pure core -- fully unit-testable, no SQLite/disk dependency.
# ---------------------------------------------------------------------------

def build_trade_log_row(
    team_a: str,
    team_b: str,
    result: TradeEvaluationResult,
    model_version: str,
    *,
    trade_id: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> dict:
    """Turns one `evaluate_trade` result into a flat row dict ready for
    `log_trade`. `trade_id`/`timestamp` are injectable (default to a
    fresh `uuid4` and the current UTC time) specifically so this stays
    deterministic and testable without freezing real time or randomness
    in the test suite itself.

    `team_a_players`/`team_b_players`/`caveats` are JSON-encoded --
    SQLite has no native list/dict column type, and encoding here (not
    at the SQLite boundary) keeps `log_trade` a dumb, row-shaped insert."""
    row = {
        "trade_id": trade_id or str(uuid.uuid4()),
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "team_a": team_a,
        "team_a_players": json.dumps([p.player_name for p in result.team_a_players]),
        "team_b": team_b,
        "team_b_players": json.dumps([p.player_name for p in result.team_b_players]),
        "fairness_score": result.fairness.team_a_share_pct,
        "fairness_method": result.fairness.fairness_method,
        "net_kvs_delta_a": result.team_a_total,
        "net_kvs_delta_b": result.team_b_total,
        "caveats": json.dumps([
            {"side": c.side, "player_name": c.player_name, "caveat": c.caveat}
            for c in result.caveats
        ]),
        "model_version": model_version,
    }
    return row


# ---------------------------------------------------------------------------
# SQLite-backed I/O -- not unit tested directly, exercised for real by app.py.
# ---------------------------------------------------------------------------

def log_trade(row: dict, db_path: Path = DEFAULT_DB_PATH) -> None:
    """Inserts one already-built row (see `build_trade_log_row`) into the
    SQLite log, creating the table on first use."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(_CREATE_TABLE_SQL)
        conn.execute(_INSERT_SQL, row)
        conn.commit()


def get_trade_history(db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Every logged trade, most recent first. Returns an empty (but
    correctly-columned) DataFrame if the log doesn't exist yet -- so
    `app.py`'s History tab has something sane to render before the
    first trade is ever evaluated."""
    db_path = Path(db_path)
    if not db_path.exists():
        return pd.DataFrame(columns=TRADE_LOG_COLUMNS)
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM trade_log ORDER BY timestamp DESC", conn)
