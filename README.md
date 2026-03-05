# chess_stats

Explore your chess.com game history and correlate performance with external factors (time of day, health data, sleep, etc.).

## How it works

```
chess.com API  →  ingest.py  →  data/games.parquet
                                      ↓
                               query.py (GameDB)
                                      ↓
                          notebooks/tutorial.ipynb
```

Game data is fetched via the [chess.com PubAPI](https://www.chess.com/news/view/published-data-api) and stored locally as parquet files. The query API returns [polars](https://pola.rs) DataFrames so you can join with any external dataset.

## Setup

Requires [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

## Ingest game history

```bash
uv run python -m chess_stats.ingest --username nirum
```

Options:
- `--username` — chess.com username (default: `nirum`)
- `--data-dir` — where to store parquet files (default: `data/`)
- `--time-classes` — game types to include (default: `rapid blitz`)

The first run fetches all available months. Subsequent runs only fetch new months (incremental). The current calendar month is always re-fetched in case new games were played.

## Query API

```python
from chess_stats import GameDB

db = GameDB("data")

# Summary
db.summary()

# All games as a polars DataFrame
db.games()
db.games(time_class="rapid", color="white", result="win")
db.games(start_date="2024-01-01", end_date="2024-12-31")

# Aggregated views
db.rating_history(time_class="rapid")
db.performance_by_hour()
db.performance_by_day()
db.opening_stats(min_games=10)
```

## Tutorial notebook

```bash
uv run jupyter notebook notebooks/tutorial.ipynb
```

Covers: summary stats, rating trajectory, win rate by hour/day, opening analysis, and a template for joining with health data.

## Data

Parquet files are written to `data/` (git-ignored):
- `data/games.parquet` — one row per game
- `data/fetch_log.parquet` — tracks which months have been fetched
