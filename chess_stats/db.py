"""
Parquet-based storage layer for chess game data.

Layout under data_dir/:
  games.parquet      — all parsed games, one row per game
  fetch_log.parquet  — which (username, year, month) have been fetched
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

# ── Schema ──────────────────────────────────────────────────────────────────

GAMES_SCHEMA = {
    "game_id": pl.Utf8,
    "username": pl.Utf8,
    "color": pl.Utf8,
    "opponent": pl.Utf8,
    "my_rating": pl.Int32,
    "opponent_rating": pl.Int32,
    "result": pl.Utf8,        # "win" | "loss" | "draw"
    "result_detail": pl.Utf8, # raw value e.g. "checkmated", "timeout"
    "time_class": pl.Utf8,
    "time_control": pl.Utf8,
    "end_time": pl.Int64,     # unix timestamp (seconds, UTC)
    "end_datetime": pl.Utf8,  # ISO 8601 UTC string
    "hour_of_day": pl.Int8,   # 0–23
    "day_of_week": pl.Int8,   # 0=Mon … 6=Sun
    "eco_code": pl.Utf8,      # e.g. "B20" (nullable)
    "num_moves": pl.Int32,
    "my_accuracy": pl.Float32,      # nullable
    "opponent_accuracy": pl.Float32, # nullable
    "pgn": pl.Utf8,
    "fen": pl.Utf8,
}

FETCH_LOG_SCHEMA = {
    "username": pl.Utf8,
    "year": pl.Int32,
    "month": pl.Int32,
    "fetched_at": pl.Int64,
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _games_path(data_dir: Path) -> Path:
    return data_dir / "games.parquet"


def _fetch_log_path(data_dir: Path) -> Path:
    return data_dir / "fetch_log.parquet"


def load_games(data_dir: Path) -> pl.DataFrame:
    p = _games_path(data_dir)
    if not p.exists():
        return pl.DataFrame(schema=GAMES_SCHEMA)
    return pl.read_parquet(p)


def load_fetch_log(data_dir: Path) -> pl.DataFrame:
    p = _fetch_log_path(data_dir)
    if not p.exists():
        return pl.DataFrame(schema=FETCH_LOG_SCHEMA)
    return pl.read_parquet(p)


def save_games(data_dir: Path, df: pl.DataFrame) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(_games_path(data_dir))


def save_fetch_log(data_dir: Path, df: pl.DataFrame) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(_fetch_log_path(data_dir))


def upsert_games(data_dir: Path, new_games: pl.DataFrame) -> pl.DataFrame:
    """Merge new_games into existing games, deduplicating on game_id."""
    existing = load_games(data_dir)
    if existing.is_empty():
        merged = new_games
    else:
        merged = (
            pl.concat([existing, new_games])
            .unique(subset=["game_id"], keep="last")
        )
    save_games(data_dir, merged)
    return merged


def record_fetch(data_dir: Path, username: str, year: int, month: int) -> None:
    """Append or update a fetch_log entry for (username, year, month)."""
    import time
    existing = load_fetch_log(data_dir)
    new_row = pl.DataFrame(
        {"username": [username], "year": [year], "month": [month], "fetched_at": [int(time.time())]},
        schema=FETCH_LOG_SCHEMA,
    )
    if existing.is_empty():
        merged = new_row
    else:
        merged = (
            pl.concat([existing, new_row])
            .unique(subset=["username", "year", "month"], keep="last")
        )
    save_fetch_log(data_dir, merged)


def fetched_months(data_dir: Path, username: str) -> set[tuple[int, int]]:
    """Return the set of (year, month) pairs already fetched for username."""
    log = load_fetch_log(data_dir)
    if log.is_empty():
        return set()
    filtered = log.filter(pl.col("username") == username)
    return {(row["year"], row["month"]) for row in filtered.iter_rows(named=True)}
