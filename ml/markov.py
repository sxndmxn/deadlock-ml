"""
Markov chain ML module for item purchase sequence modeling.
Builds transition probability matrices and finds optimal paths.
"""

from typing import Optional

import numpy as np
import polars as pl

# Special state for first item in sequence
START_STATE = -1


def build_transition_matrix(
    sequences: list[list[int]],
    item_ids: Optional[list[int]] = None,
) -> tuple[np.ndarray, dict[int, int], dict[int, int]]:
    """
    Build transition probability matrix from item purchase sequences.

    Args:
        sequences: List of item purchase sequences (ordered by purchase time)
        item_ids: Optional list of all possible item IDs. If None, inferred from data.

    Returns:
        Tuple of:
        - Transition matrix P where P[i,j] = P(buy item j after item i)
        - item_to_idx: Mapping from item ID to matrix index
        - idx_to_item: Mapping from matrix index to item ID
    """
    if not sequences:
        return np.array([]), {}, {}

    # Get all unique items if not provided
    if item_ids is None:
        all_items = set()
        for seq in sequences:
            all_items.update(seq)
        item_ids = sorted(all_items)

    # Include START state
    all_states = [START_STATE] + list(item_ids)
    n_states = len(all_states)

    # Build mappings
    item_to_idx = {item: idx for idx, item in enumerate(all_states)}
    idx_to_item = {idx: item for item, idx in item_to_idx.items()}

    # Count transitions
    transition_counts = np.zeros((n_states, n_states))

    for seq in sequences:
        if not seq:
            continue

        # Transition from START to first item
        first_item = seq[0]
        if first_item in item_to_idx:
            transition_counts[item_to_idx[START_STATE], item_to_idx[first_item]] += 1

        # Transitions between consecutive items
        for i in range(len(seq) - 1):
            from_item = seq[i]
            to_item = seq[i + 1]
            if from_item in item_to_idx and to_item in item_to_idx:
                transition_counts[item_to_idx[from_item], item_to_idx[to_item]] += 1

    # Normalize to probabilities
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    transition_matrix = transition_counts / row_sums

    return transition_matrix, item_to_idx, idx_to_item


def build_win_weighted_matrix(
    sequences: list[list[int]],
    outcomes: list[bool],
    item_ids: Optional[list[int]] = None,
    win_weight: float = 2.0,
) -> tuple[np.ndarray, dict[int, int], dict[int, int]]:
    """
    Build transition matrix weighted by win rate.

    Args:
        sequences: List of item purchase sequences
        outcomes: List of win/loss outcomes (True=win) for each sequence
        item_ids: Optional list of all possible item IDs
        win_weight: Multiplier for transitions from winning games

    Returns:
        Same as build_transition_matrix()
    """
    if not sequences or len(sequences) != len(outcomes):
        return np.array([]), {}, {}

    # Get all unique items if not provided
    if item_ids is None:
        all_items = set()
        for seq in sequences:
            all_items.update(seq)
        item_ids = sorted(all_items)

    # Include START state
    all_states = [START_STATE] + list(item_ids)
    n_states = len(all_states)

    # Build mappings
    item_to_idx = {item: idx for idx, item in enumerate(all_states)}
    idx_to_item = {idx: item for item, idx in item_to_idx.items()}

    # Count weighted transitions
    transition_counts = np.zeros((n_states, n_states))

    for seq, won in zip(sequences, outcomes):
        if not seq:
            continue

        weight = win_weight if won else 1.0

        # Transition from START to first item
        first_item = seq[0]
        if first_item in item_to_idx:
            transition_counts[item_to_idx[START_STATE], item_to_idx[first_item]] += weight

        # Transitions between consecutive items
        for i in range(len(seq) - 1):
            from_item = seq[i]
            to_item = seq[i + 1]
            if from_item in item_to_idx and to_item in item_to_idx:
                transition_counts[item_to_idx[from_item], item_to_idx[to_item]] += weight

    # Normalize to probabilities
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    transition_matrix = transition_counts / row_sums

    return transition_matrix, item_to_idx, idx_to_item


def get_next_item_probabilities(
    current_item: int,
    matrix: np.ndarray,
    item_to_idx: dict[int, int],
    idx_to_item: dict[int, int],
    top_k: int = 10,
) -> list[tuple[int, float]]:
    """
    Get ranked next item recommendations given current item.

    Args:
        current_item: Current item ID (or START_STATE for first item)
        matrix: Transition probability matrix
        item_to_idx: Item to index mapping
        idx_to_item: Index to item mapping
        top_k: Number of recommendations to return

    Returns:
        List of (item_id, probability) tuples, sorted by probability descending
    """
    if matrix.size == 0 or current_item not in item_to_idx:
        return []

    current_idx = item_to_idx[current_item]
    probabilities = matrix[current_idx]

    # Get top-k indices by probability
    top_indices = np.argsort(probabilities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        item_id = idx_to_item[idx]
        prob = probabilities[idx]
        if prob > 0 and item_id != START_STATE:
            results.append((item_id, float(prob)))

    return results


def find_optimal_path(
    matrix: np.ndarray,
    item_to_idx: dict[int, int],
    idx_to_item: dict[int, int],
    start_item: int = START_STATE,
    length: int = 6,
    exclude_items: Optional[set[int]] = None,
) -> list[tuple[int, float]]:
    """
    Find optimal (highest probability) item sequence using greedy path.

    Args:
        matrix: Transition probability matrix
        item_to_idx: Item to index mapping
        idx_to_item: Index to item mapping
        start_item: Starting item ID (default: START_STATE)
        length: Number of items in path
        exclude_items: Optional set of item IDs to exclude from path

    Returns:
        List of (item_id, probability) tuples representing optimal path
    """
    if matrix.size == 0 or start_item not in item_to_idx:
        return []

    if exclude_items is None:
        exclude_items = set()

    path = []
    current_item = start_item
    visited = set()

    for _ in range(length):
        current_idx = item_to_idx[current_item]
        probabilities = matrix[current_idx].copy()

        # Zero out visited items, START_STATE, and excluded items
        for item_id in visited | {START_STATE} | exclude_items:
            if item_id in item_to_idx:
                probabilities[item_to_idx[item_id]] = 0

        # Find best next item
        if probabilities.max() == 0:
            break

        best_idx = np.argmax(probabilities)
        best_item = idx_to_item[best_idx]
        best_prob = float(probabilities[best_idx])

        path.append((best_item, best_prob))
        visited.add(best_item)
        current_item = best_item

    return path


def get_slot_filtered_sequences(
    sequences: list[list[int]],
    slot_items: set[int],
) -> list[list[int]]:
    """
    Filter sequences to only include items from a specific slot.

    Args:
        sequences: Original item purchase sequences
        slot_items: Set of item IDs belonging to the target slot

    Returns:
        Filtered sequences containing only slot items (order preserved)
    """
    filtered = []
    for seq in sequences:
        filtered_seq = [item for item in seq if item in slot_items]
        if filtered_seq:
            filtered.append(filtered_seq)
    return filtered


def build_markov_model(
    matches_df: pl.DataFrame,
    win_weight: float = 2.0,
) -> dict:
    """
    Build complete Markov chain model from match data.

    Args:
        matches_df: Polars DataFrame with 'item_ids', 'purchase_times', and 'won' columns

    Returns:
        Dict containing:
        - matrix: Transition probability matrix
        - item_to_idx: Item to index mapping
        - idx_to_item: Index to item mapping
    """
    if matches_df.is_empty():
        return {"matrix": np.array([]), "item_to_idx": {}, "idx_to_item": {}}

    # Build sequences ordered by purchase time
    sequences = []
    outcomes = []

    for row in matches_df.iter_rows(named=True):
        item_ids = row.get("item_ids", [])
        purchase_times = row.get("purchase_times", [])
        won = row.get("won", True)

        if not item_ids or not purchase_times:
            continue

        # Sort by purchase time
        if len(item_ids) == len(purchase_times):
            sorted_items = [x for _, x in sorted(zip(purchase_times, item_ids))]
        else:
            sorted_items = list(item_ids)

        sequences.append(sorted_items)
        outcomes.append(won)

    matrix, item_to_idx, idx_to_item = build_win_weighted_matrix(
        sequences, outcomes, win_weight=win_weight
    )

    return {
        "matrix": matrix,
        "item_to_idx": item_to_idx,
        "idx_to_item": idx_to_item,
    }
