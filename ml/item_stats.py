"""
Item statistics computation for Deadlock Build Optimizer.

Computes per-hero item statistics:
- win_rate: Probability of winning when item is purchased
- pick_rate: Frequency of item purchase relative to hero matches
- total_matches: Number of matches where item was purchased
- wins: Number of wins when item was purchased
"""

from pathlib import Path

import duckdb
import polars as pl

DATA_DIR = Path(__file__).parent.parent / "data"

# Pattern for match data - supports both sharded and full dump formats
SHARD_PATTERN = str(DATA_DIR / "match_metadata" / "match_player_*.parquet")
FULL_DUMP_FILE = DATA_DIR / "player_match_history.parquet"


def get_match_data_source() -> str:
    """
    Determine the best data source for match data.
    Prefers full dump if available, falls back to shards.
    """
    if FULL_DUMP_FILE.exists():
        return str(FULL_DUMP_FILE)
    return SHARD_PATTERN


def compute_hero_item_stats(hero_id: int) -> pl.DataFrame:
    """
    Compute comprehensive item statistics for a specific hero.

    Returns DataFrame with columns:
    - item_id: Item identifier
    - total_matches: Times this item was purchased by this hero
    - wins: Wins when this item was purchased
    - total_hero_matches: Total matches for this hero (for pick rate calc)
    - win_rate: wins / total_matches
    - pick_rate: total_matches / total_hero_matches

    Uses winning match weighting is NOT applied here - we want raw statistics.
    Win weighting is applied in the Markov chain models separately.
    """
    conn = duckdb.connect(":memory:")
    data_source = get_match_data_source()

    query = f"""
    WITH hero_matches AS (
        SELECT
            match_id,
            won,
            UNNEST("items.item_id") as item_id
        FROM '{data_source}'
        WHERE hero_id = {hero_id}
          AND length("items.item_id") > 0
    ),
    hero_total AS (
        SELECT COUNT(DISTINCT match_id) as total_hero_matches
        FROM '{data_source}'
        WHERE hero_id = {hero_id}
    )
    SELECT
        item_id,
        COUNT(DISTINCT match_id) as total_matches,
        SUM(CASE WHEN won THEN 1 ELSE 0 END)::BIGINT as wins,
        (SELECT total_hero_matches FROM hero_total)::BIGINT as total_hero_matches
    FROM hero_matches
    GROUP BY item_id
    HAVING COUNT(DISTINCT match_id) >= 10  -- Minimum sample size
    ORDER BY total_matches DESC
    """

    result = conn.execute(query).pl()
    conn.close()

    # Compute derived metrics
    if not result.is_empty():
        result = result.with_columns([
            (pl.col("wins") / pl.col("total_matches")).round(4).alias("win_rate"),
            (pl.col("total_matches") / pl.col("total_hero_matches")).round(4).alias("pick_rate"),
        ])

    return result


def compute_all_hero_item_stats(min_matches: int = 100) -> dict[int, pl.DataFrame]:
    """
    Compute item statistics for all heroes with sufficient data.

    Args:
        min_matches: Minimum total matches required for a hero to be included

    Returns:
        Dictionary mapping hero_id to item stats DataFrame
    """
    from lib.db import get_heroes

    heroes = get_heroes()
    stats: dict[int, pl.DataFrame] = {}

    for row in heroes.iter_rows(named=True):
        hero_id = row["id"]
        hero_name = row.get("name", f"Hero {hero_id}")

        hero_stats = compute_hero_item_stats(hero_id)

        if hero_stats.is_empty():
            continue

        # Check if hero has enough matches
        total_matches = hero_stats["total_hero_matches"][0]
        if total_matches < min_matches:
            continue

        stats[hero_id] = hero_stats
        print(f"  {hero_name}: {len(hero_stats)} items, {total_matches} matches")

    return stats


def item_stats_to_json(stats_df: pl.DataFrame) -> list[dict]:
    """
    Convert item stats DataFrame to JSON-serializable list for Rust consumption.

    Returns list of dicts with keys:
    - item_id, total_matches, wins, win_rate, pick_rate
    """
    if stats_df.is_empty():
        return []

    return [
        {
            "item_id": int(row["item_id"]),
            "total_matches": int(row["total_matches"]),
            "wins": int(row["wins"]),
            "win_rate": float(row["win_rate"]),
            "pick_rate": float(row["pick_rate"]),
        }
        for row in stats_df.iter_rows(named=True)
    ]


def get_global_item_stats() -> pl.DataFrame:
    """
    Compute global item statistics across all heroes.

    Useful for baseline comparison (is item good overall vs for specific hero).
    """
    conn = duckdb.connect(":memory:")
    data_source = get_match_data_source()

    query = f"""
    WITH all_items AS (
        SELECT
            match_id,
            won,
            UNNEST("items.item_id") as item_id
        FROM '{data_source}'
        WHERE length("items.item_id") > 0
    ),
    total AS (
        SELECT COUNT(DISTINCT match_id) as total_matches
        FROM '{data_source}'
    )
    SELECT
        item_id,
        COUNT(DISTINCT match_id) as total_matches,
        SUM(CASE WHEN won THEN 1 ELSE 0 END)::BIGINT as wins,
        (SELECT total_matches FROM total)::BIGINT as global_matches
    FROM all_items
    GROUP BY item_id
    HAVING COUNT(DISTINCT match_id) >= 100
    ORDER BY total_matches DESC
    """

    result = conn.execute(query).pl()
    conn.close()

    if not result.is_empty():
        result = result.with_columns([
            (pl.col("wins") / pl.col("total_matches")).round(4).alias("win_rate"),
            (pl.col("total_matches") / pl.col("global_matches")).round(4).alias("pick_rate"),
        ])

    return result


if __name__ == "__main__":
    # Quick test
    print("Testing item statistics computation...")
    print(f"Data source: {get_match_data_source()}")

    # Test with hero ID 6 (Infernus)
    stats = compute_hero_item_stats(6)
    print(f"\nHero 6 item stats: {len(stats)} items")
    if not stats.is_empty():
        print(stats.head(10))
