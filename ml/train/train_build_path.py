#!/usr/bin/env python3
"""
Train a dynamic build path recommender using XGBoost.

This model predicts the next best item to buy given:
- Currently owned items (one-hot encoded)
- Current game time
- Enemy team heroes (one-hot encoded)

Outputs model in JSON format for Rust consumption.
"""

import argparse
import json
import sys
from datetime import datetime, UTC
from pathlib import Path

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


def get_connection():
    """Get DuckDB connection."""
    import duckdb
    return duckdb.connect(":memory:")


def load_item_metadata() -> dict:
    """Load item ID to index mapping."""
    items_df = pl.read_parquet(DATA_DIR / "items.parquet")
    item_ids = sorted(items_df["id"].to_list())
    return {
        "item_to_idx": {int(item_id): idx for idx, item_id in enumerate(item_ids)},
        "idx_to_item": {idx: int(item_id) for idx, item_id in enumerate(item_ids)},
        "num_items": len(item_ids),
    }


def load_hero_metadata() -> dict:
    """Load hero ID to index mapping."""
    heroes_df = pl.read_parquet(DATA_DIR / "heroes.parquet")
    hero_ids = sorted(heroes_df["id"].to_list())
    return {
        "hero_to_idx": {int(hero_id): idx for idx, hero_id in enumerate(hero_ids)},
        "idx_to_hero": {idx: int(hero_id) for idx, hero_id in enumerate(hero_ids)},
        "num_heroes": len(hero_ids),
    }


def prepare_training_data(
    hero_id: int,
    item_meta: dict,
    hero_meta: dict,
    sample_size: int = 100000,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare training data for build path model.

    For each match, we create training examples at each item purchase:
    - Features: [owned_items_one_hot, game_time_normalized, enemy_heroes_one_hot]
    - Label: next item to purchase (item index)

    Returns (X, y) arrays.
    """
    conn = get_connection()

    if verbose:
        print(f"  Loading matches for hero {hero_id}...")

    # Get winning matches with item purchase sequences
    query = f"""
    SELECT
        match_id,
        hero_id,
        won,
        "items.item_id" as item_ids,
        "items.game_time_s" as purchase_times
    FROM '{MATCH_PATTERN}'
    WHERE hero_id = {hero_id}
      AND won = true
      AND length("items.item_id") > 5
    LIMIT {sample_size}
    """

    matches_df = conn.execute(query).pl()
    conn.close()

    if verbose:
        print(f"  Found {matches_df.height} matches")

    if matches_df.height < 100:
        if verbose:
            print(f"  Insufficient matches for training")
        return None, None

    num_items = item_meta["num_items"]
    num_heroes = hero_meta["num_heroes"]
    item_to_idx = item_meta["item_to_idx"]

    # Feature dimensions:
    # - owned_items: num_items (one-hot)
    # - game_time: 1 (normalized 0-1)
    # - enemy_heroes: num_heroes (one-hot, placeholder for now)
    feature_dim = num_items + 1 + num_heroes

    X_list = []
    y_list = []

    for row in matches_df.iter_rows(named=True):
        item_ids = row["item_ids"]
        purchase_times = row["purchase_times"]

        if item_ids is None or len(item_ids) < 2:
            continue

        # Create training examples for each purchase in the sequence
        owned_items = set()
        for i in range(len(item_ids) - 1):
            current_item = item_ids[i]
            next_item = item_ids[i + 1]
            game_time = purchase_times[i] if purchase_times and i < len(purchase_times) else 0

            # Skip if items not in our vocabulary
            if current_item not in item_to_idx or next_item not in item_to_idx:
                owned_items.add(current_item)
                continue

            # Build feature vector
            features = np.zeros(feature_dim, dtype=np.float32)

            # One-hot encode owned items
            for owned in owned_items:
                if owned in item_to_idx:
                    features[item_to_idx[owned]] = 1.0

            # Normalized game time (assuming max 45 min = 2700s)
            features[num_items] = min(game_time / 2700.0, 1.0)

            # Enemy heroes placeholder (zeros for now, would need match data)
            # features[num_items + 1:] = 0  # Already zero

            X_list.append(features)
            y_list.append(item_to_idx[next_item])

            # Update owned items for next iteration
            owned_items.add(current_item)

    if len(X_list) < 100:
        if verbose:
            print(f"  Insufficient training examples ({len(X_list)})")
        return None, None

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    if verbose:
        print(f"  Created {len(X)} training examples")

    return X, y


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    use_gpu: bool = True,
    verbose: bool = True,
) -> xgb.Booster:
    """
    Train XGBoost multi-class classifier.
    """
    if not HAS_XGBOOST:
        raise RuntimeError("XGBoost not installed")

    # Split train/validation
    n_samples = len(X)
    n_train = int(n_samples * 0.8)
    indices = np.random.permutation(n_samples)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # XGBoost parameters
    params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "mlogloss",
        "seed": 42,
    }

    # Use GPU if available
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
        print("  Training XGBoost model...")

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
        best_score = evals_result["val"]["mlogloss"][-1]
        print(f"  Best validation loss: {best_score:.4f}")

    return model


def export_model_json(
    model: xgb.Booster,
    hero_id: int,
    item_meta: dict,
    hero_meta: dict,
    output_path: Path,
) -> None:
    """
    Export trained model to JSON format for Rust.

    The JSON format stores:
    - Feature metadata (item/hero indices)
    - Model trees (for inference)
    - Top-k predictions per common state (for fast lookup)
    """
    # Get model JSON
    model_json = json.loads(model.save_raw("json"))

    # Build output
    output = {
        "hero_id": hero_id,
        "model_type": "build_path_xgb",
        "generated_at": datetime.now(UTC).isoformat(),
        "feature_config": {
            "num_items": item_meta["num_items"],
            "num_heroes": hero_meta["num_heroes"],
            "item_to_idx": item_meta["item_to_idx"],
            "idx_to_item": {str(k): v for k, v in item_meta["idx_to_item"].items()},
        },
        "xgboost_model": model_json,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f)


def main(args: argparse.Namespace = None) -> int:
    """Main entry point."""
    if not HAS_XGBOOST:
        print("Error: XGBoost not installed. Install with: pip install xgboost")
        return 1

    # Load metadata
    print("Loading metadata...")
    item_meta = load_item_metadata()
    hero_meta = load_hero_metadata()
    print(f"  {item_meta['num_items']} items, {hero_meta['num_heroes']} heroes")

    # Get hero to train
    hero_id = args.hero_id if args else 6  # Default to Infernus (ID 6)

    print(f"\nTraining build path model for hero {hero_id}...")

    # Prepare training data
    X, y = prepare_training_data(
        hero_id,
        item_meta,
        hero_meta,
        sample_size=args.sample_size if args else 50000,
        verbose=True,
    )

    if X is None:
        print("Failed to prepare training data")
        return 1

    # Train model
    model = train_model(
        X, y,
        num_classes=item_meta["num_items"],
        use_gpu=not (args and args.no_gpu),
        verbose=True,
    )

    # Export model
    output_path = MODELS_DIR / f"build_path_{hero_id}.json"
    print(f"\nExporting model to {output_path}...")
    export_model_json(model, hero_id, item_meta, hero_meta, output_path)
    print("Done!")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train build path recommender model"
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
        default=50000,
        help="Max number of matches to sample (default: 50000)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration",
    )

    args = parser.parse_args()
    sys.exit(main(args))
