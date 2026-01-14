"""
DuckDB database layer for querying parquet files.
"""

from pathlib import Path
from functools import lru_cache

import duckdb
import polars as pl

DATA_DIR = Path(__file__).parent.parent / "data"
MATCH_PATTERN = str(DATA_DIR / "match_metadata" / "match_player_*.parquet")


def get_connection() -> duckdb.DuckDBPyConnection:
    """Get a configured DuckDB connection."""
    conn = duckdb.connect(":memory:")
    return conn


@lru_cache(maxsize=1)
def get_heroes() -> pl.DataFrame:
    """Get hero id/name mappings."""
    path = DATA_DIR / "heroes.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Heroes file not found: {path}")

    return pl.read_parquet(path)


@lru_cache(maxsize=1)
def get_items() -> pl.DataFrame:
    """Get item metadata (id, name, tier, slot, cost)."""
    path = DATA_DIR / "items.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Items file not found: {path}")

    return pl.read_parquet(path)


def get_hero_matches(hero_id: int, won_only: bool = True) -> pl.DataFrame:
    """
    Get match data for a specific hero with item arrays.

    Returns DataFrame with columns:
    - match_id
    - hero_id
    - won
    - item_ids (list of item IDs)
    - purchase_times (list of purchase times in seconds)
    """
    conn = get_connection()

    won_filter = "AND won = true" if won_only else ""

    query = f"""
    SELECT
        match_id,
        hero_id,
        won,
        "items.item_id" as item_ids,
        "items.game_time_s" as purchase_times
    FROM '{MATCH_PATTERN}'
    WHERE hero_id = {hero_id}
      {won_filter}
      AND length("items.item_id") > 3
    """

    result = conn.execute(query).pl()
    conn.close()
    return result


def get_all_hero_matches(won_only: bool = True) -> pl.DataFrame:
    """Get match data for all heroes (for stats page)."""
    conn = get_connection()

    won_filter = "AND won = true" if won_only else ""

    query = f"""
    SELECT
        match_id,
        hero_id,
        won,
        "items.item_id" as item_ids
    FROM '{MATCH_PATTERN}'
    WHERE length("items.item_id") > 3
    {won_filter}
    """

    result = conn.execute(query).pl()
    conn.close()
    return result


def get_hero_stats() -> pl.DataFrame:
    """Get win rate and match count stats per hero."""
    conn = get_connection()

    query = f"""
    SELECT
        hero_id,
        COUNT(*) as total_matches,
        SUM(CASE WHEN won THEN 1 ELSE 0 END) as wins,
        AVG(CASE WHEN won THEN 1.0 ELSE 0.0 END) as win_rate
    FROM '{MATCH_PATTERN}'
    GROUP BY hero_id
    ORDER BY total_matches DESC
    """

    result = conn.execute(query).pl()
    conn.close()
    return result


def get_hero_item_stats(hero_id: int) -> pl.DataFrame:
    """Get item win rates for a specific hero."""
    conn = get_connection()

    query = f"""
    WITH item_usage AS (
        SELECT
            match_id,
            won,
            UNNEST("items.item_id") as item_id
        FROM '{MATCH_PATTERN}'
        WHERE hero_id = {hero_id}
          AND length("items.item_id") > 3
    )
    SELECT
        item_id,
        COUNT(*) as times_bought,
        SUM(CASE WHEN won THEN 1 ELSE 0 END) as wins,
        AVG(CASE WHEN won THEN 1.0 ELSE 0.0 END) as win_rate
    FROM item_usage
    GROUP BY item_id
    ORDER BY times_bought DESC
    """

    result = conn.execute(query).pl()
    conn.close()
    return result
