"""SQLite persistence for signals, trades, experiments and the test-period lock."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_utc TEXT NOT NULL,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    params_json TEXT,
    data_hash TEXT,
    results_json TEXT
);
CREATE TABLE IF NOT EXISTS holdout_registry (
    period_key TEXT PRIMARY KEY,
    first_used_utc TEXT NOT NULL,
    experiment_id INTEGER,
    strategy_key TEXT,
    uses INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_utc TEXT NOT NULL,
    bar_time TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    strategy TEXT,
    confidence REAL,
    entry REAL, sl REAL, tp REAL, rr REAL,
    est_net_profit REAL,
    lots REAL,
    decision TEXT,
    reason TEXT,
    mode TEXT,
    UNIQUE(bar_time, symbol, direction, strategy)
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER,
    mode TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    lots REAL NOT NULL,
    open_utc TEXT NOT NULL,
    entry REAL NOT NULL,
    sl REAL, tp REAL,
    close_utc TEXT,
    exit REAL,
    pnl REAL,
    r REAL,
    ticket INTEGER,
    status TEXT NOT NULL DEFAULT 'open',
    comment TEXT
);
"""


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def file_hash(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()[:16]


class Store:
    def __init__(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # experiments -----------------------------------------------------------
    def log_experiment(self, kind: str, name: str, params: dict[str, Any], results: dict[str, Any],
                       data_hash: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO experiments(created_utc, kind, name, params_json, data_hash, results_json) VALUES (?,?,?,?,?,?)",
            (now_utc(), kind, name, json.dumps(params, default=str), data_hash, json.dumps(results, default=str)),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def experiments(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM experiments ORDER BY id", self.conn)

    # holdout lock ----------------------------------------------------------
    def holdout_status(self, period_key: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM holdout_registry WHERE period_key=?", (period_key,)).fetchone()
        return dict(row) if row else None

    def burn_holdout(self, period_key: str, experiment_id: int, strategy_key: str) -> None:
        if self.holdout_status(period_key):
            self.conn.execute("UPDATE holdout_registry SET uses = uses + 1 WHERE period_key=?", (period_key,))
        else:
            self.conn.execute(
                "INSERT INTO holdout_registry(period_key, first_used_utc, experiment_id, strategy_key) VALUES (?,?,?,?)",
                (period_key, now_utc(), experiment_id, strategy_key),
            )
        self.conn.commit()

    # signals and trades ------------------------------------------------------
    def record_signal(self, sig: dict[str, Any]) -> int | None:
        cols = ["bar_time", "symbol", "direction", "strategy", "confidence", "entry", "sl", "tp", "rr",
                "est_net_profit", "lots", "decision", "reason", "mode"]
        try:
            cur = self.conn.execute(
                f"INSERT INTO signals(created_utc, {', '.join(cols)}) VALUES (?, {', '.join('?' * len(cols))})",
                (now_utc(), *[sig.get(c) for c in cols]),
            )
        except sqlite3.IntegrityError:
            return None  # duplicate signal for the same bar/direction/strategy
        self.conn.commit()
        return int(cur.lastrowid)

    def open_trade(self, t: dict[str, Any]) -> int:
        cols = ["signal_id", "mode", "symbol", "direction", "lots", "open_utc", "entry", "sl", "tp", "ticket", "comment"]
        cur = self.conn.execute(
            f"INSERT INTO trades({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", tuple(t.get(c) for c in cols)
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def close_trade(self, trade_id: int, close_utc: str, exit_price: float, pnl: float, r: float) -> None:
        self.conn.execute(
            "UPDATE trades SET close_utc=?, exit=?, pnl=?, r=?, status='closed' WHERE id=?",
            (close_utc, exit_price, pnl, r, trade_id),
        )
        self.conn.commit()

    def update_stop(self, trade_id: int, sl: float) -> None:
        self.conn.execute("UPDATE trades SET sl=? WHERE id=?", (sl, trade_id))
        self.conn.commit()

    def open_trades(self, mode: str | None = None) -> pd.DataFrame:
        q = "SELECT * FROM trades WHERE status='open'"
        args: tuple = ()
        if mode:
            q += " AND mode=?"
            args = (mode,)
        return pd.read_sql_query(q, self.conn, params=args)

    def table(self, name: str) -> pd.DataFrame:
        if name not in ("signals", "trades", "experiments", "holdout_registry"):
            raise ValueError(name)
        return pd.read_sql_query(f"SELECT * FROM {name} ORDER BY 1", self.conn)

    def export(self, out_dir: str | Path) -> list[Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = []
        for name in ("signals", "trades", "experiments", "holdout_registry"):
            df = self.table(name)
            p_csv = out / f"{name}.csv"
            p_json = out / f"{name}.json"
            df.to_csv(p_csv, index=False)
            df.to_json(p_json, orient="records", indent=1)
            paths += [p_csv, p_json]
        return paths
