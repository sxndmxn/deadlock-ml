"""
Association rules ML module using FP-Growth algorithm.
Finds item synergies from winning match item sets.
Uses GPU-accelerated cuML for FP-Growth, mlxtend for rule generation.
"""

import pandas as pd
import polars as pl
import numpy as np

try:
    import cudf
    from cuml.frequent_pattern import FPGrowth as CuFPGrowth
    HAS_CUML = True
except Exception:
    cudf = None
    CuFPGrowth = None
    HAS_CUML = False

# CPU rule generation (fast enough, cuML doesn't have this)
from mlxtend.frequent_patterns import association_rules, fpgrowth as mlxtend_fpgrowth


def build_item_matrix(matches_df: pl.DataFrame) -> pd.DataFrame:
    """
    Convert item arrays to binary matrix for FP-Growth.

    Args:
        matches_df: Polars DataFrame with 'item_ids' column containing lists of item IDs

    Returns:
        Binary pandas DataFrame where columns are item IDs, rows are matches,
        values are 1 if item was purchased, 0 otherwise.
    """
    if matches_df.is_empty():
        return pd.DataFrame()

    # Get all unique items across all matches
    all_items = set()
    for items in matches_df["item_ids"].to_list():
        if items is not None:
            all_items.update(items)

    if not all_items:
        return pd.DataFrame()

    # Build binary matrix
    item_list = sorted(all_items)
    matrix_data = []

    for items in matches_df["item_ids"].to_list():
        if items is None:
            items = []
        item_set = set(items)
        row = [1 if item in item_set else 0 for item in item_list]
        matrix_data.append(row)

    return pd.DataFrame(matrix_data, columns=item_list)


def find_frequent_itemsets(
    matrix: pd.DataFrame,
    min_support: float = 0.05,
) -> pd.DataFrame:
    """
    Find frequent item combinations using GPU-accelerated FP-Growth.

    Args:
        matrix: Binary item matrix from build_item_matrix()
        min_support: Minimum support threshold (0-1)

    Returns:
        DataFrame with columns: support, itemsets
    """
    if matrix.empty:
        return pd.DataFrame(columns=["support", "itemsets"])

    if HAS_CUML:
        try:
            # Convert to cuDF for GPU processing
            gdf = cudf.DataFrame.from_pandas(matrix.astype(bool))

            # Run FP-Growth on GPU
            fp = CuFPGrowth(min_support=min_support)
            fp.fit(gdf)

            # Get results back as pandas
            itemsets = fp.frequent_itemsets_.to_pandas()

            # cuML returns 'items' column, mlxtend expects 'itemsets'
            if "items" in itemsets.columns:
                itemsets = itemsets.rename(columns={"items": "itemsets"})

            # Convert item indices back to frozensets of column names for mlxtend compatibility
            col_names = matrix.columns.tolist()
            itemsets["itemsets"] = itemsets["itemsets"].apply(
                lambda x: frozenset(col_names[i] for i in x) if hasattr(x, "__iter__") else frozenset([col_names[x]])
            )

            return itemsets
        except Exception as e:
            print(f"    GPU FP-Growth error: {e}")

    # Fallback to CPU FP-Growth (mlxtend)
    return mlxtend_fpgrowth(matrix.astype(bool), min_support=min_support, use_colnames=True)


def generate_rules(
    itemsets: pd.DataFrame,
    min_confidence: float = 0.5,
) -> pd.DataFrame:
    """
    Generate association rules from frequent itemsets.

    Args:
        itemsets: DataFrame from find_frequent_itemsets()
        min_confidence: Minimum confidence threshold (0-1)

    Returns:
        DataFrame with columns: antecedents, consequents, support,
        confidence, lift
    """
    if itemsets.empty:
        return pd.DataFrame(
            columns=["antecedents", "consequents", "support", "confidence", "lift"]
        )

    try:
        rules = association_rules(
            itemsets,
            metric="confidence",
            min_threshold=min_confidence,
        )
        if rules.empty:
            return pd.DataFrame(
                columns=["antecedents", "consequents", "support", "confidence", "lift"]
            )

        return rules[
            ["antecedents", "consequents", "support", "confidence", "lift"]
        ].copy()
    except Exception as e:
        print(f"    Rule generation error: {e}")
        return pd.DataFrame(
            columns=["antecedents", "consequents", "support", "confidence", "lift"]
        )


def get_items_that_pair_with(item_id: int, rules_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get items that associate with a given item.

    Args:
        item_id: The item ID to find pairings for
        rules_df: Association rules DataFrame

    Returns:
        DataFrame of rules where item_id appears in antecedents,
        sorted by lift descending.
    """
    if rules_df.empty:
        return rules_df

    mask = rules_df["antecedents"].apply(lambda x: item_id in x)
    filtered = rules_df[mask].copy()

    if filtered.empty:
        return filtered

    return filtered.sort_values("lift", ascending=False)


def build_association_model(
    matches_df: pl.DataFrame,
    min_support: float = 0.05,
    min_confidence: float = 0.5,
) -> pd.DataFrame:
    """
    Build complete association rules model from match data.

    Args:
        matches_df: Polars DataFrame with 'item_ids' column
        min_support: Minimum support for frequent itemsets
        min_confidence: Minimum confidence for rules

    Returns:
        Association rules DataFrame
    """
    matrix = build_item_matrix(matches_df)
    itemsets = find_frequent_itemsets(matrix, min_support)
    rules = generate_rules(itemsets, min_confidence)
    return rules
