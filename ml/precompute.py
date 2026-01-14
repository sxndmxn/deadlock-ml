#!/usr/bin/env python3
"""
Pre-compute ML models for each hero.
Saves models to disk for fast loading in Streamlit.
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import polars as pl

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.db import get_heroes, get_hero_matches
from ml.association import build_association_model
from ml.markov import build_markov_model

MODELS_DIR = Path(__file__).parent.parent / "models"
MIN_MATCHES = 100


def precompute_hero(hero_id: int, hero_name: str, verbose: bool = True) -> dict | None:
    """
    Precompute models for a single hero.

    Returns dict with model data, or None if insufficient data.
    """
    def log(msg: str):
        if verbose:
            print(f"    {msg}")

    # Get matches for this hero
    log("Fetching match data...")
    matches_df = get_hero_matches(hero_id, won_only=False)
    log(f"Found {matches_df.height} matches")

    if matches_df.height < MIN_MATCHES:
        return None

    # Build association rules (from winning matches only)
    log("Filtering winning matches...")
    winning_matches = matches_df.filter(pl.col("won") == True)
    log(f"Found {winning_matches.height} winning matches")

    log("Building item matrix for association rules...")
    from ml.association import build_item_matrix, find_frequent_itemsets, generate_rules
    matrix = build_item_matrix(winning_matches)
    log(f"Item matrix: {matrix.shape[0]} matches x {matrix.shape[1]} items")

    log("Finding frequent itemsets (FP-Growth)...")
    itemsets = find_frequent_itemsets(matrix, min_support=0.03)
    log(f"Found {len(itemsets)} frequent itemsets")

    log("Generating association rules...")
    association_rules = generate_rules(itemsets, min_confidence=0.4)
    log(f"Generated {len(association_rules)} rules")

    # Build Markov chain model (using all matches, weighted by win)
    log("Building Markov transition matrix...")
    markov_model = build_markov_model(matches_df, win_weight=2.0)
    n_states = len(markov_model["item_to_idx"])
    log(f"Markov model: {n_states} states")
    log("Done!")
    return {
        "hero_id": hero_id,
        "hero_name": hero_name,
        "match_count": matches_df.height,
        "win_count": winning_matches.height,
        "association_rules": association_rules,
        "markov_model": markov_model,
    }


def save_model(model: dict, hero_id: int) -> None:
    """Save model to pickle file."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{hero_id}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(hero_id: int) -> dict | None:
    """Load model from pickle file."""
    path = MODELS_DIR / f"{hero_id}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def find_hero(heroes_df: pl.DataFrame, query: str) -> pl.DataFrame:
    """Find hero by name (case-insensitive) or ID."""
    # Try as ID first
    if query.isdigit():
        result = heroes_df.filter(pl.col("id") == int(query))
        if not result.is_empty():
            return result

    # Try exact name match (case-insensitive)
    result = heroes_df.filter(pl.col("name").str.to_lowercase() == query.lower())
    if not result.is_empty():
        return result

    # Try partial name match
    result = heroes_df.filter(pl.col("name").str.to_lowercase().str.contains(query.lower()))
    return result


def main(args: argparse.Namespace = None) -> int:
    """Main entry point."""
    start_time = time.time()

    # Get heroes
    try:
        heroes_df = get_heroes()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run scripts/fetch_data.py first to download data files.")
        return 1

    # List heroes if requested
    if args and args.list_heroes:
        print("Available heroes:")
        for row in heroes_df.sort("name").iter_rows(named=True):
            print(f"  {row['id']:3d}  {row['name']}")
        return 0

    # Filter to single hero if specified
    if args and args.hero:
        heroes_df = find_hero(heroes_df, args.hero)
        if heroes_df.is_empty():
            print(f"Hero '{args.hero}' not found. Use --list-heroes to see available heroes.")
            return 1
        if heroes_df.height > 1:
            print(f"Multiple heroes match '{args.hero}':")
            for row in heroes_df.iter_rows(named=True):
                print(f"  {row['id']:3d}  {row['name']}")
            print("Please be more specific.")
            return 1

    total = heroes_df.height
    processed = 0
    skipped = 0

    print(f"Processing {total} heroes...")

    for row in heroes_df.iter_rows(named=True):
        hero_id = row["id"]
        hero_name = row.get("name", f"Hero {hero_id}")

        print(f"[{processed + 1}/{total}] Processing: {hero_name} (ID: {hero_id})")

        model = precompute_hero(hero_id, hero_name)

        if model is None:
            print(f"  Skipped (insufficient data, <{MIN_MATCHES} matches)")
            skipped += 1
        else:
            save_model(model, hero_id)
            print(f"  Saved: {model['match_count']} matches, {len(model['association_rules'])} rules")

        processed += 1

    elapsed = time.time() - start_time
    print(f"\nDone! Processed {processed} heroes, skipped {skipped}")
    print(f"Total time: {elapsed:.1f}s")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-compute ML models for Deadlock heroes"
    )
    parser.add_argument(
        "--hero",
        type=str,
        help="Process single hero by name or ID (e.g., --hero Infernus)",
    )
    parser.add_argument(
        "--list-heroes",
        action="store_true",
        help="List available heroes and exit",
    )

    args = parser.parse_args()
    sys.exit(main(args))
