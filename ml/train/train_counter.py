#!/usr/bin/env python3
"""
Generate counter-picking data: hero matchup matrix and item counter scores.

This script computes:
1. Hero matchup matrix: win_rate[hero_A][vs_hero_B]
   - For each hero pair, calculates win rate when facing each other
2. Item counter scores: item_effectiveness[item_id][vs_hero_id]
   - For each item, calculates how effective it is against specific heroes

Outputs to JSON format for Rust consumption.
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, UTC
from pathlib import Path

import polars as pl

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MATCH_PATTERN = str(DATA_DIR / "match_metadata" / "match_player_*.parquet")


def get_connection():
    """Get DuckDB connection."""
    import duckdb
    return duckdb.connect(":memory:")


def load_hero_metadata() -> dict:
    """Load hero ID to index mapping and names."""
    heroes_df = pl.read_parquet(DATA_DIR / "heroes.parquet")
    hero_ids = sorted(heroes_df["id"].to_list())

    # Try to get hero names if available
    hero_names = {}
    if "name" in heroes_df.columns:
        for row in heroes_df.iter_rows(named=True):
            hero_names[int(row["id"])] = row["name"]

    return {
        "hero_to_idx": {int(hero_id): idx for idx, hero_id in enumerate(hero_ids)},
        "idx_to_hero": {idx: int(hero_id) for idx, hero_id in enumerate(hero_ids)},
        "hero_names": hero_names,
        "num_heroes": len(hero_ids),
        "hero_ids": hero_ids,
    }


def load_item_metadata() -> dict:
    """Load item metadata."""
    items_df = pl.read_parquet(DATA_DIR / "items.parquet")
    item_ids = sorted(items_df["id"].to_list())

    # Try to get item names if available
    item_names = {}
    if "name" in items_df.columns:
        for row in items_df.iter_rows(named=True):
            item_names[int(row["id"])] = row["name"]

    return {
        "item_to_idx": {int(item_id): idx for idx, item_id in enumerate(item_ids)},
        "idx_to_item": {idx: int(item_id) for idx, item_id in enumerate(item_ids)},
        "item_names": item_names,
        "num_items": len(item_ids),
        "item_ids": item_ids,
    }


def compute_hero_matchup_matrix(
    hero_meta: dict,
    sample_size: int = 500000,
    min_games: int = 100,
    verbose: bool = True,
) -> dict:
    """
    Compute hero vs hero win rate matrix.

    For each hero pair (A, B), we count games where:
    - hero_A_wins: A won when facing B
    - hero_A_games: total games where A faced B

    Returns dict with:
    - matchup_matrix[hero_A_id][hero_B_id] = {wins, games, win_rate}
    """
    conn = get_connection()

    if verbose:
        print("  Loading match data for hero matchups...")

    # Get all players in matches with their team and outcome
    # We need to join players in the same match to find opponents
    query = f"""
    SELECT
        match_id,
        hero_id,
        team,
        won
    FROM '{MATCH_PATTERN}'
    LIMIT {sample_size * 12}
    """

    df = conn.execute(query).pl()
    conn.close()

    if verbose:
        print(f"  Loaded {df.height} player records")

    # Group by match_id to get all players in each match
    matches = df.group_by("match_id").agg([
        pl.col("hero_id").alias("heroes"),
        pl.col("team").alias("teams"),
        pl.col("won").alias("outcomes"),
    ])

    if verbose:
        print(f"  Processing {matches.height} matches...")

    # Track matchup statistics
    # matchups[hero_A][hero_B] = {wins: int, games: int}
    matchups = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "games": 0}))

    processed = 0
    for row in matches.iter_rows(named=True):
        heroes = row["heroes"]
        teams = row["teams"]
        outcomes = row["outcomes"]

        if heroes is None or len(heroes) < 2:
            continue

        # Separate teams
        team_0_heroes = []
        team_1_heroes = []
        team_0_won = None

        for i, (hero, team, won) in enumerate(zip(heroes, teams, outcomes)):
            if team == "Team0":
                team_0_heroes.append(hero)
                if team_0_won is None:
                    team_0_won = won
            else:
                team_1_heroes.append(hero)

        if not team_0_heroes or not team_1_heroes or team_0_won is None:
            continue

        # Record matchups: each hero on team 0 vs each hero on team 1
        for hero_a in team_0_heroes:
            for hero_b in team_1_heroes:
                # hero_a vs hero_b: hero_a won if team_0_won
                matchups[hero_a][hero_b]["games"] += 1
                if team_0_won:
                    matchups[hero_a][hero_b]["wins"] += 1

                # hero_b vs hero_a: hero_b won if not team_0_won
                matchups[hero_b][hero_a]["games"] += 1
                if not team_0_won:
                    matchups[hero_b][hero_a]["wins"] += 1

        processed += 1
        if verbose and processed % 50000 == 0:
            print(f"    Processed {processed} matches...")

    if verbose:
        print(f"  Processed {processed} total matches")

    # Convert to output format with win rates
    matrix = {}
    hero_ids = hero_meta["hero_ids"]

    for hero_a in hero_ids:
        matrix[str(hero_a)] = {}
        for hero_b in hero_ids:
            if hero_a == hero_b:
                # No self-matchup
                matrix[str(hero_a)][str(hero_b)] = {
                    "wins": 0,
                    "games": 0,
                    "win_rate": 0.5,
                }
            else:
                data = matchups[hero_a][hero_b]
                games = data["games"]
                wins = data["wins"]

                # Only include if enough games
                if games >= min_games:
                    win_rate = wins / games if games > 0 else 0.5
                else:
                    win_rate = 0.5  # Default to 50% if insufficient data

                matrix[str(hero_a)][str(hero_b)] = {
                    "wins": wins,
                    "games": games,
                    "win_rate": round(win_rate, 4),
                }

    return matrix


def compute_item_counter_scores(
    hero_meta: dict,
    item_meta: dict,
    sample_size: int = 500000,
    min_games: int = 50,
    verbose: bool = True,
) -> dict:
    """
    Compute item effectiveness against each enemy hero.

    For each (item, enemy_hero) pair, we calculate:
    - How often players with this item win against that enemy hero
    - Compared to the baseline win rate for that item

    Returns dict with:
    - item_scores[item_id][enemy_hero_id] = {wins, games, win_rate, effectiveness}
    """
    conn = get_connection()

    if verbose:
        print("  Loading match data for item counter scores...")

    # Get players with their items and team info
    query = f"""
    SELECT
        match_id,
        hero_id,
        team,
        won,
        "items.item_id" as item_ids
    FROM '{MATCH_PATTERN}'
    WHERE length("items.item_id") > 3
    LIMIT {sample_size * 12}
    """

    df = conn.execute(query).pl()
    conn.close()

    if verbose:
        print(f"  Loaded {df.height} player records")

    # Group by match
    matches = df.group_by("match_id").agg([
        pl.col("hero_id").alias("heroes"),
        pl.col("team").alias("teams"),
        pl.col("won").alias("outcomes"),
        pl.col("item_ids").alias("all_items"),
    ])

    if verbose:
        print(f"  Processing {matches.height} matches...")

    # Track item performance against each enemy hero
    # item_vs_hero[item_id][enemy_hero_id] = {wins: int, games: int}
    item_vs_hero = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "games": 0}))

    # Track overall item win rates (baseline)
    item_overall = defaultdict(lambda: {"wins": 0, "games": 0})

    processed = 0
    for row in matches.iter_rows(named=True):
        heroes = row["heroes"]
        teams = row["teams"]
        outcomes = row["outcomes"]
        all_items = row["all_items"]

        if heroes is None or len(heroes) < 2:
            continue

        # Organize by team
        team_data = {"Team0": [], "Team1": []}

        for i, (hero, team, won, items) in enumerate(zip(heroes, teams, outcomes, all_items)):
            if items is None:
                items = []
            team_data[team].append({
                "hero": hero,
                "won": won,
                "items": set(items),
            })

        if not team_data["Team0"] or not team_data["Team1"]:
            continue

        # For each player, record item effectiveness vs enemy team
        for own_team, enemy_team in [("Team0", "Team1"), ("Team1", "Team0")]:
            enemy_heroes = [p["hero"] for p in team_data[enemy_team]]

            for player in team_data[own_team]:
                won = player["won"]
                items = player["items"]

                for item_id in items:
                    # Overall item stats
                    item_overall[item_id]["games"] += 1
                    if won:
                        item_overall[item_id]["wins"] += 1

                    # Item vs each enemy hero
                    for enemy_hero in enemy_heroes:
                        item_vs_hero[item_id][enemy_hero]["games"] += 1
                        if won:
                            item_vs_hero[item_id][enemy_hero]["wins"] += 1

        processed += 1
        if verbose and processed % 50000 == 0:
            print(f"    Processed {processed} matches...")

    if verbose:
        print(f"  Processed {processed} total matches")

    # Compute effectiveness scores
    # effectiveness = (win_rate_vs_hero - baseline_win_rate) / baseline_win_rate
    # Positive = item is better than average against this hero
    # Negative = item is worse than average against this hero

    item_scores = {}

    for item_id in item_meta["item_ids"]:
        baseline = item_overall[item_id]
        baseline_wr = baseline["wins"] / baseline["games"] if baseline["games"] > 0 else 0.5

        item_scores[str(item_id)] = {
            "baseline_win_rate": round(baseline_wr, 4),
            "baseline_games": baseline["games"],
            "vs_heroes": {},
        }

        for hero_id in hero_meta["hero_ids"]:
            data = item_vs_hero[item_id][hero_id]
            games = data["games"]
            wins = data["wins"]

            if games >= min_games:
                win_rate = wins / games
                # Effectiveness: how much better/worse than baseline
                effectiveness = (win_rate - baseline_wr) / baseline_wr if baseline_wr > 0 else 0
            else:
                win_rate = baseline_wr  # Use baseline if insufficient data
                effectiveness = 0.0

            item_scores[str(item_id)]["vs_heroes"][str(hero_id)] = {
                "wins": wins,
                "games": games,
                "win_rate": round(win_rate, 4),
                "effectiveness": round(effectiveness, 4),
            }

    return item_scores


def export_counter_matrix(
    hero_matchups: dict,
    item_scores: dict,
    hero_meta: dict,
    item_meta: dict,
    output_path: Path,
    verbose: bool = True,
) -> None:
    """
    Export counter-picking data to JSON.
    """
    output = {
        "model_type": "counter_matrix",
        "generated_at": datetime.now(UTC).isoformat(),
        "metadata": {
            "num_heroes": hero_meta["num_heroes"],
            "num_items": item_meta["num_items"],
            "hero_ids": [int(h) for h in hero_meta["hero_ids"]],
            "item_ids": [int(i) for i in item_meta["item_ids"]],
            "hero_names": {str(k): v for k, v in hero_meta["hero_names"].items()},
            "item_names": {str(k): v for k, v in item_meta["item_names"].items()},
        },
        "hero_matchups": hero_matchups,
        "item_counter_scores": item_scores,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"  Writing to {output_path}...")

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Report file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    if verbose:
        print(f"  Output file size: {size_mb:.2f} MB")


def main(args: argparse.Namespace = None) -> int:
    """Main entry point."""
    # Load metadata
    print("Loading metadata...")
    hero_meta = load_hero_metadata()
    item_meta = load_item_metadata()
    print(f"  {hero_meta['num_heroes']} heroes")
    print(f"  {item_meta['num_items']} items")

    sample_size = args.sample_size if args else 100000
    min_games = args.min_games if args else 100

    # Compute hero matchup matrix
    print("\nComputing hero matchup matrix...")
    hero_matchups = compute_hero_matchup_matrix(
        hero_meta,
        sample_size=sample_size,
        min_games=min_games,
        verbose=True,
    )

    # Count valid matchups
    valid_matchups = sum(
        1 for hero_a in hero_matchups.values()
        for data in hero_a.values()
        if data["games"] >= min_games
    )
    print(f"  Computed {valid_matchups} valid matchups (>= {min_games} games)")

    # Compute item counter scores
    print("\nComputing item counter scores...")
    item_scores = compute_item_counter_scores(
        hero_meta,
        item_meta,
        sample_size=sample_size,
        min_games=min_games // 2,  # Lower threshold for item-hero pairs
        verbose=True,
    )

    # Export
    output_path = MODELS_DIR / "counter_matrix.json"
    print(f"\nExporting to {output_path}...")
    export_counter_matrix(
        hero_matchups,
        item_scores,
        hero_meta,
        item_meta,
        output_path,
        verbose=True,
    )

    print("\nDone!")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate counter-picking data"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100000,
        help="Max matches to process (default: 100000)",
    )
    parser.add_argument(
        "--min-games",
        type=int,
        default=100,
        help="Minimum games for reliable statistics (default: 100)",
    )

    args = parser.parse_args()
    sys.exit(main(args))
