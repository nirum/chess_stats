"""
Fetch chess.com game history and store it locally as parquet files.

Usage (CLI):
    uv run python -m chess_stats.ingest --username nirum
    uv run python -m chess_stats.ingest --username nirum --data-dir ./data

The script performs a full fetch on first run, then only fetches months that
haven't been recorded in the fetch log (incremental updates). The current
calendar month is always re-fetched since it may not be complete yet.
"""

from __future__ import annotations

import argparse
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import requests

from chess_stats import db

_BASE = "https://api.chess.com/pub"
_USER_AGENT = "chess_stats/0.1 (github.com/nirum/chess_stats)"
_DEFAULT_TIME_CLASSES = {"rapid", "blitz"}
_REQUEST_DELAY = 0.5  # seconds between API calls


# ── HTTP helpers ─────────────────────────────────────────────────────────────

def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": _USER_AGENT, "Accept-Encoding": "gzip"})
    return s


def _get(session: requests.Session, url: str) -> dict | list:
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ── API calls ────────────────────────────────────────────────────────────────

def get_archive_months(username: str, session: requests.Session) -> list[tuple[int, int]]:
    """Return [(year, month), ...] for all available monthly archives."""
    data = _get(session, f"{_BASE}/player/{username}/games/archives")
    months = []
    for url in data.get("archives", []):
        # URL format: .../games/YYYY/MM
        parts = url.rstrip("/").split("/")
        months.append((int(parts[-2]), int(parts[-1])))
    return months


def fetch_monthly_games(
    username: str,
    year: int,
    month: int,
    session: requests.Session,
    time_classes: set[str],
) -> list[dict]:
    """Fetch and filter games for a single month from the chess.com API."""
    url = f"{_BASE}/player/{username}/games/{year:04d}/{month:02d}"
    data = _get(session, url)
    return [g for g in data.get("games", []) if g.get("time_class") in time_classes]


# ── Parsing ──────────────────────────────────────────────────────────────────

def _normalize_result(raw: str) -> str:
    """Map a chess.com result value to win / loss / draw."""
    wins = {"win"}
    draws = {"agreed", "repetition", "stalemate", "insufficient", "50move", "timevsinsufficient", "draw"}
    if raw in wins:
        return "win"
    if raw in draws:
        return "draw"
    return "loss"


def _parse_eco(eco_url: str | None) -> str | None:
    """Extract ECO code from a URL like https://www.chess.com/openings/B20-..."""
    if not eco_url:
        return None
    m = re.search(r"/openings/([A-E]\d+)", eco_url)
    return m.group(1) if m else None


def _count_moves(pgn: str) -> int:
    """Count the number of full moves in a PGN string."""
    # Remove header lines (start with '['), then count move numbers "1." "2." etc.
    body = re.sub(r"\[.*?\]\s*", "", pgn, flags=re.DOTALL)
    move_numbers = re.findall(r"\b\d+\.", body)
    return len(move_numbers)


def parse_game(raw: dict, username: str) -> dict | None:
    """
    Convert a raw chess.com game dict into a flat record for storage.
    Returns None if the game cannot be parsed (e.g. missing player data).
    """
    white = raw.get("white", {})
    black = raw.get("black", {})

    if not white or not black:
        return None

    white_name = white.get("username", "").lower()
    username_lower = username.lower()

    if white_name == username_lower:
        color = "white"
        my_side, opp_side = white, black
    else:
        color = "black"
        my_side, opp_side = black, white

    result_raw = my_side.get("result", "")
    result = _normalize_result(result_raw)

    end_ts = raw.get("end_time")
    if end_ts is None:
        return None

    end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)

    accuracies = raw.get("accuracies", {})
    my_acc = accuracies.get("white" if color == "white" else "black")
    opp_acc = accuracies.get("black" if color == "white" else "white")

    game_url = raw.get("url", "")
    game_id = game_url.rstrip("/").split("/")[-1] if game_url else ""

    pgn = raw.get("pgn", "")

    return {
        "game_id": game_id,
        "username": username,
        "color": color,
        "opponent": opp_side.get("username", ""),
        "my_rating": int(my_side.get("rating") or 0),
        "opponent_rating": int(opp_side.get("rating") or 0),
        "result": result,
        "result_detail": result_raw,
        "time_class": raw.get("time_class", ""),
        "time_control": raw.get("time_control", ""),
        "end_time": end_ts,
        "end_datetime": end_dt.isoformat(),
        "hour_of_day": end_dt.hour,
        "day_of_week": end_dt.weekday(),  # 0=Mon
        "eco_code": _parse_eco(raw.get("eco")),
        "num_moves": _count_moves(pgn) if pgn else 0,
        "my_accuracy": float(my_acc) if my_acc is not None else None,
        "opponent_accuracy": float(opp_acc) if opp_acc is not None else None,
        "pgn": pgn,
        "fen": raw.get("fen", ""),
    }


# ── Main ingest logic ─────────────────────────────────────────────────────────

def ingest(
    username: str,
    data_dir: Path = Path("data"),
    time_classes: set[str] = _DEFAULT_TIME_CLASSES,
) -> pl.DataFrame:
    """
    Fetch all new months for username and append them to the local parquet store.

    On first run: fetches all available months.
    On subsequent runs: only fetches months not in fetch_log, plus the
    current month (in case it wasn't complete last time).

    Returns the full games DataFrame after ingestion.
    """
    data_dir = Path(data_dir)
    session = _session()

    print(f"Fetching archive list for '{username}'...")
    all_months = get_archive_months(username, session)
    if not all_months:
        print("No archive months found.")
        return db.load_games(data_dir)

    already_fetched = db.fetched_months(data_dir, username)

    now = datetime.now(tz=timezone.utc)
    current_month = (now.year, now.month)

    # Always re-fetch the current month (it may have grown since last run)
    to_fetch = [
        (y, m) for y, m in all_months
        if (y, m) not in already_fetched or (y, m) == current_month
    ]

    if not to_fetch:
        print("Already up to date.")
        return db.load_games(data_dir)

    print(f"Fetching {len(to_fetch)} month(s)...")

    all_records: list[dict] = []
    for i, (year, month) in enumerate(to_fetch):
        print(f"  [{i+1}/{len(to_fetch)}] {year}/{month:02d}", end=" ", flush=True)
        try:
            raw_games = fetch_monthly_games(username, year, month, session, time_classes)
            records = [r for raw in raw_games if (r := parse_game(raw, username))]
            all_records.extend(records)
            db.record_fetch(data_dir, username, year, month)
            print(f"→ {len(records)} games")
        except requests.HTTPError as e:
            print(f"→ HTTP error: {e}")
        except Exception as e:
            print(f"→ error: {e}")

        if i < len(to_fetch) - 1:
            time.sleep(_REQUEST_DELAY)

    if all_records:
        new_df = pl.DataFrame(all_records, schema=db.GAMES_SCHEMA)
        db.upsert_games(data_dir, new_df)
        print(f"\nAdded {len(all_records)} game(s). Total: {len(db.load_games(data_dir))} games.")
    else:
        print("\nNo new games found.")

    return db.load_games(data_dir)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest chess.com game history")
    parser.add_argument("--username", default="nirum", help="chess.com username")
    parser.add_argument("--data-dir", default="data", help="Directory for parquet files")
    parser.add_argument(
        "--time-classes",
        nargs="+",
        default=list(_DEFAULT_TIME_CLASSES),
        choices=["rapid", "blitz", "bullet", "daily"],
        help="Game types to include",
    )
    args = parser.parse_args()
    ingest(args.username, Path(args.data_dir), set(args.time_classes))


if __name__ == "__main__":
    main()
