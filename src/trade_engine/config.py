"""
Trade Engine -- Config.

Single source of truth for values that need to be bumped in exactly one
place when they change, rather than existing as a magic string
scattered across `app.py`, `trade_log.py`, and anywhere else that needs
to know which model generation produced a verdict. Currently just
`MODEL_VERSION`; a future retrain bumps it here, once.
"""

MODEL_VERSION = "2026-v1"
