"""
Python query API for the local chess game parquet store.

Example:
    from chess_stats import GameDB

    db = GameDB("data")
    df = db.games(time_class="rapid")
    print(db.summary())
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import polars.selectors as cs

from chess_stats.db import load_games

_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


class GameDB:
    """
    Read-only interface to the local parquet game store.

    All query methods return polars DataFrames, making it easy to join with
    external data sources (health metrics, sleep logs, etc.).
    """

    def __init__(self, data_dir: str | Path = "data"):
        self._data_dir = Path(data_dir)
        self._df: pl.LazyFrame = load_games(self._data_dir).lazy()

    def reload(self) -> None:
        """Re-read the parquet file (call after running ingest)."""
        self._df = load_games(self._data_dir).lazy()

    # ── Core accessor ─────────────────────────────────────────────────────────

    def games(
        self,
        *,
        username: str | None = None,
        time_class: str | None = None,
        color: str | None = None,
        result: str | None = None,
        start_date: str | None = None,   # ISO date string, e.g. "2024-01-01"
        end_date: str | None = None,
        include_pgn: bool = False,
    ) -> pl.DataFrame:
        """
        Return a filtered DataFrame of games.

        Parameters
        ----------
        username : filter to a specific player (default: all)
        time_class : "rapid" | "blitz" | ...
        color : "white" | "black"
        result : "win" | "loss" | "draw"
        start_date : inclusive lower bound on end_datetime (ISO date string)
        end_date : inclusive upper bound on end_datetime (ISO date string)
        include_pgn : if False (default), the pgn column is dropped for brevity
        """
        lf = self._df

        if username is not None:
            lf = lf.filter(pl.col("username") == username)
        if time_class is not None:
            lf = lf.filter(pl.col("time_class") == time_class)
        if color is not None:
            lf = lf.filter(pl.col("color") == color)
        if result is not None:
            lf = lf.filter(pl.col("result") == result)
        if start_date is not None:
            lf = lf.filter(pl.col("end_datetime") >= start_date)
        if end_date is not None:
            lf = lf.filter(pl.col("end_datetime") <= end_date + "Z")

        df = lf.collect()

        if not include_pgn and "pgn" in df.columns:
            df = df.drop("pgn")

        return df.sort("end_time")

    # ── Aggregated views ──────────────────────────────────────────────────────

    def rating_history(self, time_class: str = "rapid") -> pl.DataFrame:
        """
        Return (end_datetime, my_rating) sorted by time for the given time class.
        Useful for plotting rating trajectory.
        """
        return (
            self._df
            .filter(pl.col("time_class") == time_class)
            .select(["end_datetime", "end_time", "my_rating", "color", "result"])
            .sort("end_time")
            .collect()
        )

    def performance_by_hour(self, time_class: str | None = None) -> pl.DataFrame:
        """
        Return win rate and game count broken down by hour of day (UTC).

        Columns: hour_of_day, n_games, win_rate, wins, losses, draws
        """
        lf = self._df
        if time_class is not None:
            lf = lf.filter(pl.col("time_class") == time_class)

        return (
            lf.group_by("hour_of_day")
            .agg(
                pl.len().alias("n_games"),
                (pl.col("result") == "win").sum().alias("wins"),
                (pl.col("result") == "loss").sum().alias("losses"),
                (pl.col("result") == "draw").sum().alias("draws"),
            )
            .with_columns(
                (pl.col("wins") / pl.col("n_games")).alias("win_rate")
            )
            .sort("hour_of_day")
            .collect()
        )

    def performance_by_day(self, time_class: str | None = None) -> pl.DataFrame:
        """
        Return win rate and game count broken down by day of week.

        Columns: day_of_week, day_name, n_games, win_rate, wins, losses, draws
        """
        lf = self._df
        if time_class is not None:
            lf = lf.filter(pl.col("time_class") == time_class)

        day_map = pl.DataFrame(
            {"day_of_week": list(range(7)), "day_name": _DAYS},
            schema={"day_of_week": pl.Int8, "day_name": pl.Utf8},
        )

        return (
            lf.group_by("day_of_week")
            .agg(
                pl.len().alias("n_games"),
                (pl.col("result") == "win").sum().alias("wins"),
                (pl.col("result") == "loss").sum().alias("losses"),
                (pl.col("result") == "draw").sum().alias("draws"),
            )
            .with_columns(
                (pl.col("wins") / pl.col("n_games")).alias("win_rate")
            )
            .collect()
            .join(day_map, on="day_of_week", how="left")
            .sort("day_of_week")
        )

    def opening_stats(self, min_games: int = 5, time_class: str | None = None) -> pl.DataFrame:
        """
        Return per-ECO-code performance statistics.

        Columns: eco_code, n_games, win_rate, avg_my_accuracy
        """
        lf = self._df.filter(pl.col("eco_code").is_not_null())
        if time_class is not None:
            lf = lf.filter(pl.col("time_class") == time_class)

        return (
            lf.group_by("eco_code")
            .agg(
                pl.len().alias("n_games"),
                (pl.col("result") == "win").sum().alias("wins"),
                pl.col("my_accuracy").drop_nulls().mean().alias("avg_my_accuracy"),
                pl.col("num_moves").mean().alias("avg_moves"),
            )
            .with_columns(
                (pl.col("wins") / pl.col("n_games")).alias("win_rate")
            )
            .filter(pl.col("n_games") >= min_games)
            .sort("n_games", descending=True)
            .collect()
        )

    def summary(self) -> dict:
        """Return a dict of high-level statistics across all stored games."""
        df = self._df.collect()

        if df.is_empty():
            return {"total_games": 0}

        total = len(df)
        wins = (df["result"] == "win").sum()
        losses = (df["result"] == "loss").sum()
        draws = (df["result"] == "draw").sum()

        dates = df.sort("end_time")["end_datetime"]

        by_class = (
            df.group_by("time_class")
            .agg(pl.len().alias("n_games"))
            .sort("n_games", descending=True)
        )

        return {
            "total_games": total,
            "win_rate": round(wins / total, 3),
            "wins": int(wins),
            "losses": int(losses),
            "draws": int(draws),
            "first_game": dates[0],
            "last_game": dates[-1],
            "usernames": df["username"].unique().to_list(),
            "games_by_type": dict(zip(by_class["time_class"], by_class["n_games"])),
            "avg_my_accuracy": round(df["my_accuracy"].drop_nulls().mean() or 0, 1),
        }
