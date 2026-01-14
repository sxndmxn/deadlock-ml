#!/usr/bin/env python3
"""
Train a win probability predictor using XGBoost binary classification.

This model predicts the probability of winning given:
- Ally team heroes (one-hot encoded)
- Enemy team heroes (one-hot encoded)
- Item counts per category (weapon/vitality/spirit)
- Gold/soul advantage

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


def load_hero_metadata() -> dict:
    """Load hero ID to index mapping."""
    heroes_df = pl.read_parquet(DATA_DIR / "heroes.parquet")
    hero_ids = sorted(heroes_df["id"].to_list())
    return {
        "hero_to_idx": {int(hero_id): idx for idx, hero_id in enumerate(hero_ids)},
        "idx_to_hero": {idx: int(hero_id) for idx, hero_id in enumerate(hero_ids)},
        "num_heroes": len(hero_ids),
    }


def load_item_metadata() -> dict:
    """Load item metadata with slot categorization."""
    items_df = pl.read_parquet(DATA_DIR / "items.parquet")

    # Categorize items by slot
    weapon_items = set()
    vitality_items = set()
    spirit_items = set()

    for row in items_df.iter_rows(named=True):
        item_id = int(row["id"])
        slot = row.get("slot", "").lower() if row.get("slot") else ""

        if "weapon" in slot:
            weapon_items.add(item_id)
        elif "vitality" in slot or "armor" in slot:
            vitality_items.add(item_id)
        elif "spirit" in slot:
            spirit_items.add(item_id)

    return {
        "weapon_items": weapon_items,
        "vitality_items": vitality_items,
        "spirit_items": spirit_items,
        "all_items": set(items_df["id"].to_list()),
    }


def prepare_training_data(
    hero_meta: dict,
    item_meta: dict,
    sample_size: int = 100000,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare training data for win prediction model.

    For each match participant, we create a training example:
    - Features: [ally_heroes_one_hot, enemy_heroes_one_hot, item_counts, net_worth_normalized]
    - Label: 1 if won, 0 if lost

    Returns (X, y) arrays.
    """
    conn = get_connection()

    if verbose:
        print("  Loading match data...")

    # Get matches with player data
    query = f"""
    SELECT
        match_id,
        hero_id,
        won,
        net_worth,
        "items.item_id" as item_ids
    FROM '{MATCH_PATTERN}'
    WHERE length("items.item_id") > 3
    LIMIT {sample_size * 2}
    """

    matches_df = conn.execute(query).pl()
    conn.close()

    if verbose:
        print(f"  Found {matches_df.height} player records")

    if matches_df.height < 1000:
        if verbose:
            print("  Insufficient data for training")
        return None, None

    num_heroes = hero_meta["num_heroes"]
    hero_to_idx = hero_meta["hero_to_idx"]

    weapon_items = item_meta["weapon_items"]
    vitality_items = item_meta["vitality_items"]
    spirit_items = item_meta["spirit_items"]

    # Feature dimensions:
    # - own_hero: num_heroes (one-hot for player's hero)
    # - item_counts: 3 (weapon, vitality, spirit item counts)
    # - net_worth: 1 (normalized)
    # Note: Full ally/enemy team would need match grouping which is complex
    # For now, we use individual player features
    feature_dim = num_heroes + 3 + 1

    X_list = []
    y_list = []

    for row in matches_df.iter_rows(named=True):
        hero_id = row["hero_id"]
        won = row["won"]
        net_worth = row["net_worth"] or 0
        item_ids = row["item_ids"] or []

        if hero_id not in hero_to_idx:
            continue

        # Build feature vector
        features = np.zeros(feature_dim, dtype=np.float32)

        # One-hot encode hero
        features[hero_to_idx[hero_id]] = 1.0

        # Count items by category
        weapon_count = sum(1 for item in item_ids if item in weapon_items)
        vitality_count = sum(1 for item in item_ids if item in vitality_items)
        spirit_count = sum(1 for item in item_ids if item in spirit_items)

        features[num_heroes] = weapon_count / 12.0  # Normalize by max items
        features[num_heroes + 1] = vitality_count / 12.0
        features[num_heroes + 2] = spirit_count / 12.0

        # Normalized net worth (assuming max ~100k souls)
        features[num_heroes + 3] = min(net_worth / 100000.0, 1.0)

        X_list.append(features)
        y_list.append(1 if won else 0)

    if len(X_list) < 1000:
        if verbose:
            print(f"  Insufficient training examples ({len(X_list)})")
        return None, None

    # Sample to target size
    if len(X_list) > sample_size:
        indices = np.random.choice(len(X_list), sample_size, replace=False)
        X_list = [X_list[i] for i in indices]
        y_list = [y_list[i] for i in indices]

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    if verbose:
        print(f"  Created {len(X)} training examples")
        print(f"  Win rate in data: {y.mean():.2%}")

    return X, y


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    use_gpu: bool = True,
    verbose: bool = True,
) -> xgb.Booster:
    """
    Train XGBoost binary classifier for win prediction.
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

    # XGBoost parameters for binary classification
    params = {
        "objective": "binary:logistic",
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": ["logloss", "auc"],
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
        best_auc = max(evals_result["val"]["auc"])
        best_loss = min(evals_result["val"]["logloss"])
        print(f"  Best validation AUC: {best_auc:.4f}")
        print(f"  Best validation loss: {best_loss:.4f}")

    return model


def export_model_json(
    model: xgb.Booster,
    hero_meta: dict,
    output_path: Path,
) -> None:
    """
    Export trained model to JSON format for Rust.
    """
    # Get model JSON
    model_json = json.loads(model.save_raw("json"))

    # Build output
    output = {
        "model_type": "win_predictor_xgb",
        "generated_at": datetime.now(UTC).isoformat(),
        "feature_config": {
            "num_heroes": hero_meta["num_heroes"],
            "hero_to_idx": hero_meta["hero_to_idx"],
            "idx_to_hero": {str(k): v for k, v in hero_meta["idx_to_hero"].items()},
            "feature_names": [
                "hero_one_hot",
                "weapon_item_count",
                "vitality_item_count",
                "spirit_item_count",
                "net_worth_normalized",
            ],
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
    hero_meta = load_hero_metadata()
    item_meta = load_item_metadata()
    print(f"  {hero_meta['num_heroes']} heroes")
    print(f"  {len(item_meta['weapon_items'])} weapon items")
    print(f"  {len(item_meta['vitality_items'])} vitality items")
    print(f"  {len(item_meta['spirit_items'])} spirit items")

    print("\nPreparing training data...")

    # Prepare training data
    X, y = prepare_training_data(
        hero_meta,
        item_meta,
        sample_size=args.sample_size if args else 50000,
        verbose=True,
    )

    if X is None:
        print("Failed to prepare training data")
        return 1

    # Train model
    model = train_model(
        X, y,
        use_gpu=not (args and args.no_gpu),
        verbose=True,
    )

    # Export model
    output_path = MODELS_DIR / "win_predictor.json"
    print(f"\nExporting model to {output_path}...")
    export_model_json(model, hero_meta, output_path)
    print("Done!")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train win probability predictor model"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50000,
        help="Max number of samples to use (default: 50000)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration",
    )

    args = parser.parse_args()
    sys.exit(main(args))
