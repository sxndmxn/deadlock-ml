#!/usr/bin/env python3
"""
Train an optimal 12-item full build recommender.

This model uses constrained beam search with a neural scorer to find optimal
complete builds given:
- Hero ID
- Enemy team composition (optional)
- Game progression (tier-aware transitions)

Key features:
- Slot balance constraints (weapon/vitality/spirit)
- Tier-aware item transitions
- Synergy scoring between items
- XGBoost-based build quality scorer

Outputs model in JSON format for Rust consumption.
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed, using fallback mode")

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MATCH_PATTERN = str(DATA_DIR / "match_metadata" / "match_player_*.parquet")


# Constants for build constraints
MAX_ITEMS = 12
SLOTS = ["weapon", "vitality", "spirit"]
SLOT_SOFT_MAX = 5  # Soft limit per slot type
SLOT_HARD_MAX = 6  # Hard limit per slot type
MIN_ITEMS_PER_SLOT = 2  # Minimum items from each slot in final build


class ItemInfo(NamedTuple):
    """Item metadata for build optimization."""
    id: int
    name: str
    tier: int
    slot: str
    cost: int


@dataclass
class BuildCandidate:
    """A candidate build being explored by beam search."""
    items: list[int]  # Item IDs in purchase order
    score: float
    slot_counts: dict[str, int]
    tier_sum: int
    total_cost: int

    def clone(self) -> "BuildCandidate":
        return BuildCandidate(
            items=self.items.copy(),
            score=self.score,
            slot_counts=self.slot_counts.copy(),
            tier_sum=self.tier_sum,
            total_cost=self.total_cost,
        )


def get_connection():
    """Get DuckDB connection."""
    import duckdb
    return duckdb.connect(":memory:")


def load_item_metadata() -> dict[int, ItemInfo]:
    """Load item metadata with tier and slot information."""
    items_df = pl.read_parquet(DATA_DIR / "items.parquet")

    items = {}
    for row in items_df.iter_rows(named=True):
        item_id = int(row["id"])
        tier = row.get("tier")
        item_type = row.get("type")
        slot_type = row.get("slot_type")
        cost = row.get("cost")
        name = row.get("name", f"item_{item_id}")

        # Only include purchasable upgrade items with valid tier
        if item_type != b"upgrade" or tier is None:
            continue

        # Decode slot type
        if slot_type == b"weapon":
            slot = "weapon"
        elif slot_type == b"vitality":
            slot = "vitality"
        elif slot_type == b"spirit":
            slot = "spirit"
        else:
            continue

        items[item_id] = ItemInfo(
            id=item_id,
            name=name,
            tier=int(tier),
            slot=slot,
            cost=int(cost) if cost else 0,
        )

    return items


def load_hero_metadata() -> dict[int, str]:
    """Load hero ID to name mapping."""
    heroes_df = pl.read_parquet(DATA_DIR / "heroes.parquet")
    return {
        int(row["id"]): row.get("name", f"hero_{row['id']}")
        for row in heroes_df.iter_rows(named=True)
    }


def load_winning_builds(
    hero_id: int,
    items_meta: dict[int, ItemInfo],
    sample_size: int = 50000,
) -> list[list[int]]:
    """
    Load winning builds for a hero.

    Returns list of item ID lists (purchase order).
    """
    conn = get_connection()

    query = f"""
    SELECT
        "items.item_id" as item_ids
    FROM '{MATCH_PATTERN}'
    WHERE hero_id = {hero_id}
      AND won = true
      AND length("items.item_id") >= 8
    LIMIT {sample_size}
    """

    df = conn.execute(query).pl()
    conn.close()

    builds = []
    for row in df.iter_rows(named=True):
        item_ids = row.get("item_ids")
        if item_ids is None:
            continue

        # Filter to only purchasable items we know about
        valid_items = [
            item_id for item_id in item_ids
            if item_id in items_meta
        ]

        if len(valid_items) >= 6:
            builds.append(valid_items)

    return builds


def compute_item_cooccurrence(
    builds: list[list[int]],
    items_meta: dict[int, ItemInfo],
) -> dict[tuple[int, int], float]:
    """
    Compute item co-occurrence scores for synergy detection.

    Returns dict mapping (item_a, item_b) -> synergy_score
    where synergy_score is log odds ratio of co-occurrence.
    """
    item_counts: dict[int, int] = defaultdict(int)
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    total_builds = len(builds)

    for build in builds:
        item_set = set(build)
        for item_id in item_set:
            item_counts[item_id] += 1

        # Count pairs (unordered)
        items = sorted(item_set)
        for i, item_a in enumerate(items):
            for item_b in items[i+1:]:
                pair_counts[(item_a, item_b)] += 1

    # Compute synergy scores using log odds ratio
    synergy: dict[tuple[int, int], float] = {}

    for (item_a, item_b), count in pair_counts.items():
        if count < 10:  # Minimum observations
            continue

        p_a = item_counts[item_a] / total_builds
        p_b = item_counts[item_b] / total_builds
        p_ab = count / total_builds

        # Expected co-occurrence under independence
        expected = p_a * p_b

        if expected > 0:
            # Log odds ratio (positive = synergy, negative = anti-synergy)
            ratio = p_ab / expected
            synergy[(item_a, item_b)] = np.log(max(ratio, 0.01))

    return synergy


def compute_item_winrates(
    hero_id: int,
    items_meta: dict[int, ItemInfo],
    min_games: int = 50,
) -> dict[int, float]:
    """
    Compute item win rates for a hero.

    Returns dict mapping item_id -> win_rate
    """
    conn = get_connection()

    query = f"""
    WITH item_stats AS (
        SELECT
            UNNEST("items.item_id") as item_id,
            won
        FROM '{MATCH_PATTERN}'
        WHERE hero_id = {hero_id}
          AND length("items.item_id") > 0
    )
    SELECT
        item_id,
        AVG(CASE WHEN won THEN 1.0 ELSE 0.0 END) as win_rate,
        COUNT(*) as games
    FROM item_stats
    GROUP BY item_id
    HAVING COUNT(*) >= {min_games}
    """

    df = conn.execute(query).pl()
    conn.close()

    return {
        int(row["item_id"]): float(row["win_rate"])
        for row in df.iter_rows(named=True)
        if row["item_id"] in items_meta
    }


def prepare_build_features(
    build: list[int],
    items_meta: dict[int, ItemInfo],
    synergy_scores: dict[tuple[int, int], float],
    item_winrates: dict[int, float],
    num_items: int,
) -> np.ndarray:
    """
    Create feature vector for a build.

    Features:
    - Item one-hot encoding (num_items dimensions)
    - Slot balance features (3 dimensions)
    - Tier distribution (4 dimensions)
    - Total synergy score (1 dimension)
    - Average item win rate (1 dimension)
    - Total cost normalized (1 dimension)
    """
    # Create item index mapping
    item_ids = sorted(items_meta.keys())
    item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}

    feature_dim = num_items + 3 + 4 + 1 + 1 + 1  # one-hot + slots + tiers + synergy + winrate + cost
    features = np.zeros(feature_dim, dtype=np.float32)

    # One-hot encode items
    for item_id in build:
        if item_id in item_to_idx:
            features[item_to_idx[item_id]] = 1.0

    # Slot balance (normalized counts)
    slot_counts = {"weapon": 0, "vitality": 0, "spirit": 0}
    tier_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    total_cost = 0

    for item_id in build:
        if item_id in items_meta:
            info = items_meta[item_id]
            slot_counts[info.slot] += 1
            tier_counts[info.tier] += 1
            total_cost += info.cost

    offset = num_items
    features[offset] = slot_counts["weapon"] / MAX_ITEMS
    features[offset + 1] = slot_counts["vitality"] / MAX_ITEMS
    features[offset + 2] = slot_counts["spirit"] / MAX_ITEMS

    # Tier distribution
    offset += 3
    for tier in range(1, 5):
        features[offset + tier - 1] = tier_counts[tier] / MAX_ITEMS

    # Synergy score
    offset += 4
    total_synergy = 0.0
    items_sorted = sorted(build)
    for i, item_a in enumerate(items_sorted):
        for item_b in items_sorted[i+1:]:
            key = (min(item_a, item_b), max(item_a, item_b))
            total_synergy += synergy_scores.get(key, 0.0)
    features[offset] = total_synergy / max(len(build), 1)

    # Average win rate
    offset += 1
    winrates = [item_winrates.get(item_id, 0.5) for item_id in build if item_id in items_meta]
    features[offset] = np.mean(winrates) if winrates else 0.5

    # Total cost normalized (assuming max ~80k souls for full build)
    offset += 1
    features[offset] = total_cost / 80000.0

    return features


def train_build_scorer(
    builds: list[list[int]],
    items_meta: dict[int, ItemInfo],
    synergy_scores: dict[tuple[int, int], float],
    item_winrates: dict[int, float],
    use_gpu: bool = True,
    verbose: bool = True,
) -> xgb.Booster | None:
    """
    Train XGBoost model to score build quality.

    Uses winning builds as positive examples and randomly shuffled
    builds as negative examples.
    """
    if not HAS_XGBOOST:
        return None

    num_items = len(items_meta)

    X_list = []
    y_list = []

    if verbose:
        print(f"  Preparing training data from {len(builds)} builds...")

    for build in builds:
        # Positive example: actual winning build
        features = prepare_build_features(
            build, items_meta, synergy_scores, item_winrates, num_items
        )
        X_list.append(features)
        y_list.append(1.0)

        # Negative example: shuffled/modified build
        if len(build) >= 4:
            # Create a worse build by removing good items and adding random ones
            shuffled = build.copy()
            np.random.shuffle(shuffled)
            # Replace some items with random ones
            all_items = list(items_meta.keys())
            for i in range(min(3, len(shuffled))):
                shuffled[i] = np.random.choice(all_items)

            features = prepare_build_features(
                shuffled, items_meta, synergy_scores, item_winrates, num_items
            )
            X_list.append(features)
            y_list.append(0.0)

    if len(X_list) < 100:
        if verbose:
            print("  Insufficient training data")
        return None

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    if verbose:
        print(f"  Created {len(X)} training examples")

    # Train/val split
    n_samples = len(X)
    n_train = int(n_samples * 0.8)
    indices = np.random.permutation(n_samples)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx])
    dval = xgb.DMatrix(X[val_idx], label=y[val_idx])

    params = {
        "objective": "binary:logistic",
        "max_depth": 5,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": ["logloss", "auc"],
        "seed": 42,
    }

    if use_gpu:
        try:
            params["device"] = "cuda"
            params["tree_method"] = "hist"
            if verbose:
                print("  Using GPU acceleration (CUDA)")
        except Exception:
            if verbose:
                print("  GPU not available, using CPU")

    evals = [(dtrain, "train"), (dval, "val")]
    evals_result = {}

    if verbose:
        print("  Training build scorer...")

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=10,
        verbose_eval=verbose,
    )

    if verbose:
        best_auc = max(evals_result["val"]["auc"])
        print(f"  Best validation AUC: {best_auc:.4f}")

    return model


def beam_search_build(
    hero_id: int,
    items_meta: dict[int, ItemInfo],
    synergy_scores: dict[tuple[int, int], float],
    item_winrates: dict[int, float],
    scorer: xgb.Booster | None,
    beam_width: int = 10,
    max_items: int = MAX_ITEMS,
) -> list[int]:
    """
    Use beam search to find optimal 12-item build.

    Constraints:
    - Tier-aware: early items should be lower tier
    - Slot balance: 2-5 items per slot
    - No duplicates

    Returns list of item IDs in purchase order.
    """
    num_items_total = len(items_meta)
    item_ids = sorted(items_meta.keys())

    # Group items by tier
    items_by_tier: dict[int, list[int]] = {1: [], 2: [], 3: [], 4: []}
    for item_id, info in items_meta.items():
        items_by_tier[info.tier].append(item_id)

    # Initialize beam with empty builds
    beam: list[BuildCandidate] = [
        BuildCandidate(
            items=[],
            score=0.0,
            slot_counts={"weapon": 0, "vitality": 0, "spirit": 0},
            tier_sum=0,
            total_cost=0,
        )
    ]

    # Tier schedule: which tiers are available at each build position
    # Early game: tier 1-2, mid game: tier 2-3, late game: tier 3-4
    tier_schedule = [
        {1, 2},      # Item 0-1
        {1, 2},
        {1, 2, 3},   # Item 2-4
        {1, 2, 3},
        {2, 3},
        {2, 3, 4},   # Item 5-8
        {2, 3, 4},
        {3, 4},
        {3, 4},
        {3, 4},      # Item 9-11
        {3, 4},
        {3, 4},
    ]

    for position in range(max_items):
        allowed_tiers = tier_schedule[position] if position < len(tier_schedule) else {3, 4}
        candidates: list[BuildCandidate] = []

        for build in beam:
            owned_set = set(build.items)

            # Get candidate items
            candidate_items = []
            for tier in allowed_tiers:
                for item_id in items_by_tier[tier]:
                    if item_id in owned_set:
                        continue

                    info = items_meta[item_id]

                    # Check slot constraints
                    if build.slot_counts[info.slot] >= SLOT_HARD_MAX:
                        continue

                    candidate_items.append(item_id)

            # Score and add candidates
            for item_id in candidate_items:
                new_build = build.clone()
                new_build.items.append(item_id)

                info = items_meta[item_id]
                new_build.slot_counts[info.slot] += 1
                new_build.tier_sum += info.tier
                new_build.total_cost += info.cost

                # Score the build
                if scorer is not None:
                    features = prepare_build_features(
                        new_build.items, items_meta, synergy_scores,
                        item_winrates, num_items_total
                    )
                    dmatrix = xgb.DMatrix(features.reshape(1, -1))
                    new_build.score = float(scorer.predict(dmatrix)[0])
                else:
                    # Fallback scoring: use item win rates, synergy, and slot balance
                    wr = item_winrates.get(item_id, 0.5)
                    syn = sum(
                        synergy_scores.get((min(item_id, other), max(item_id, other)), 0.0)
                        for other in owned_set
                    )

                    # Slot balance bonus: prefer slots with fewer items
                    slot_count = new_build.slot_counts[info.slot]
                    min_slot_count = min(new_build.slot_counts.values())
                    slot_balance_bonus = 0.0
                    if slot_count == min_slot_count:
                        slot_balance_bonus = 0.2  # Strong bonus for filling lowest slot
                    elif slot_count <= 2:
                        slot_balance_bonus = 0.1  # Encourage underrepresented slots
                    elif slot_count <= 4:
                        slot_balance_bonus = 0.02
                    elif slot_count >= SLOT_SOFT_MAX:
                        slot_balance_bonus = -0.15  # Penalize overrepresented slots

                    # Calculate slot variance penalty (prefer balanced distributions)
                    counts = list(new_build.slot_counts.values())
                    mean_count = sum(counts) / 3
                    variance = sum((c - mean_count) ** 2 for c in counts) / 3
                    balance_penalty = variance * 0.05  # Penalty for imbalance

                    new_build.score = build.score + wr + syn * 0.1 + slot_balance_bonus - balance_penalty

                candidates.append(new_build)

        # Keep top-k candidates
        candidates.sort(key=lambda b: b.score, reverse=True)
        beam = candidates[:beam_width]

        if not beam:
            break

    # Filter final builds for slot balance
    valid_builds = []
    for build in beam:
        if all(build.slot_counts[slot] >= MIN_ITEMS_PER_SLOT for slot in SLOTS):
            valid_builds.append(build)

    if valid_builds:
        return valid_builds[0].items
    elif beam:
        return beam[0].items
    else:
        return []


def generate_precomputed_builds(
    hero_id: int,
    items_meta: dict[int, ItemInfo],
    synergy_scores: dict[tuple[int, int], float],
    item_winrates: dict[int, float],
    scorer: xgb.Booster | None,
    num_variants: int = 5,
) -> list[dict]:
    """
    Generate multiple build variants for a hero.

    Returns list of build dicts with items and metadata.
    """
    builds = []

    # Run beam search with different random seeds
    for variant in range(num_variants):
        np.random.seed(42 + variant)

        build_items = beam_search_build(
            hero_id,
            items_meta,
            synergy_scores,
            item_winrates,
            scorer,
            beam_width=10 + variant * 2,  # Vary beam width
        )

        if not build_items:
            continue

        # Calculate build metadata
        slot_counts = {"weapon": 0, "vitality": 0, "spirit": 0}
        tier_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        total_cost = 0
        item_names = []

        for item_id in build_items:
            info = items_meta.get(item_id)
            if info:
                slot_counts[info.slot] += 1
                tier_counts[info.tier] += 1
                total_cost += info.cost
                item_names.append(info.name)

        # Score the build
        if scorer is not None:
            features = prepare_build_features(
                build_items, items_meta, synergy_scores,
                item_winrates, len(items_meta)
            )
            dmatrix = xgb.DMatrix(features.reshape(1, -1))
            score = float(scorer.predict(dmatrix)[0])
        else:
            score = sum(item_winrates.get(item_id, 0.5) for item_id in build_items) / len(build_items)

        builds.append({
            "variant": variant,
            "items": build_items,
            "item_names": item_names,
            "score": round(score, 4),
            "slot_counts": slot_counts,
            "tier_counts": tier_counts,
            "total_cost": total_cost,
        })

    # Sort by score
    builds.sort(key=lambda b: b["score"], reverse=True)
    return builds


def export_full_build_model(
    hero_id: int,
    items_meta: dict[int, ItemInfo],
    synergy_scores: dict[tuple[int, int], float],
    item_winrates: dict[int, float],
    builds: list[dict],
    scorer_model: xgb.Booster | None,
    output_path: Path,
) -> None:
    """
    Export full build model to JSON.
    """
    # Convert synergy scores to serializable format
    synergy_list = [
        {"item_a": a, "item_b": b, "score": round(score, 4)}
        for (a, b), score in synergy_scores.items()
    ]

    # Convert item metadata
    items_list = [
        {
            "id": info.id,
            "name": info.name,
            "tier": info.tier,
            "slot": info.slot,
            "cost": info.cost,
        }
        for info in items_meta.values()
    ]

    # Convert win rates
    winrates_list = [
        {"item_id": item_id, "win_rate": round(wr, 4)}
        for item_id, wr in item_winrates.items()
    ]

    output = {
        "model_type": "full_build_optimizer",
        "hero_id": hero_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "config": {
            "max_items": MAX_ITEMS,
            "slots": SLOTS,
            "slot_soft_max": SLOT_SOFT_MAX,
            "slot_hard_max": SLOT_HARD_MAX,
            "min_items_per_slot": MIN_ITEMS_PER_SLOT,
        },
        "items": items_list,
        "synergy_scores": synergy_list,
        "item_winrates": winrates_list,
        "precomputed_builds": builds,
    }

    # Add scorer model if available
    if scorer_model is not None:
        output["scorer_model"] = json.loads(scorer_model.save_raw("json"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


def main(args: argparse.Namespace = None) -> int:
    """Main entry point."""
    hero_id = args.hero_id if args else 6
    sample_size = args.sample_size if args else 30000
    use_gpu = not (args and args.no_gpu)

    print(f"Training full build optimizer for hero {hero_id}...")

    # Load metadata
    print("\nLoading metadata...")
    items_meta = load_item_metadata()
    heroes = load_hero_metadata()
    hero_name = heroes.get(hero_id, f"Hero {hero_id}")
    print(f"  Hero: {hero_name}")
    print(f"  {len(items_meta)} purchasable items")

    # Load winning builds
    print("\nLoading winning builds...")
    builds = load_winning_builds(hero_id, items_meta, sample_size)
    print(f"  Found {len(builds)} winning builds")

    if len(builds) < 100:
        print("Error: Insufficient winning builds for training")
        return 1

    # Compute synergy scores
    print("\nComputing item synergies...")
    synergy_scores = compute_item_cooccurrence(builds, items_meta)
    print(f"  Found {len(synergy_scores)} item synergies")

    # Compute item win rates
    print("\nComputing item win rates...")
    item_winrates = compute_item_winrates(hero_id, items_meta)
    print(f"  Win rates for {len(item_winrates)} items")

    # Train build scorer
    print("\nTraining build scorer...")
    scorer = None
    if HAS_XGBOOST:
        scorer = train_build_scorer(
            builds, items_meta, synergy_scores, item_winrates,
            use_gpu=use_gpu, verbose=True
        )
    else:
        print("  Skipping scorer training (XGBoost not available)")

    # Generate optimized builds
    print("\nGenerating optimized builds via beam search...")
    optimized_builds = generate_precomputed_builds(
        hero_id, items_meta, synergy_scores, item_winrates, scorer,
        num_variants=5,
    )
    print(f"  Generated {len(optimized_builds)} build variants")

    for i, build in enumerate(optimized_builds[:3]):
        print(f"\n  Build {i + 1} (score: {build['score']:.3f}):")
        print(f"    Slots: W={build['slot_counts']['weapon']}, V={build['slot_counts']['vitality']}, S={build['slot_counts']['spirit']}")
        print(f"    Cost: {build['total_cost']} souls")
        print(f"    Items: {', '.join(build['item_names'][:6])}...")

    # Export model
    output_path = MODELS_DIR / f"full_build_{hero_id}.json"
    print(f"\nExporting model to {output_path}...")
    export_full_build_model(
        hero_id, items_meta, synergy_scores, item_winrates,
        optimized_builds, scorer, output_path
    )

    print("Done!")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train optimal 12-item full build model"
    )
    parser.add_argument(
        "--hero-id",
        type=int,
        default=6,
        help="Hero ID to train model for (default: 6 = Infernus)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=30000,
        help="Max number of matches to sample (default: 30000)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration",
    )

    args = parser.parse_args()
    sys.exit(main(args))
