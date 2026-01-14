"""
Item metadata helper for lookups and categorization.
"""

from functools import lru_cache
from typing import Optional

import polars as pl

from lib.db import get_items


@lru_cache(maxsize=1)
def _load_items() -> pl.DataFrame:
    """Load and cache items DataFrame."""
    return get_items()


@lru_cache(maxsize=1)
def _get_item_lookup() -> dict[int, dict]:
    """Build item ID to metadata lookup dict."""
    df = _load_items()
    lookup = {}
    for row in df.iter_rows(named=True):
        lookup[row["id"]] = row
    return lookup


def get_item_name(item_id: int) -> str:
    """Get item name by ID. Returns 'Unknown' if not found."""
    lookup = _get_item_lookup()
    item = lookup.get(item_id)
    if item is None:
        return "Unknown"
    return item.get("name", "Unknown")


def get_items_by_tier(tier: int) -> pl.DataFrame:
    """Get items filtered by tier (1-4)."""
    df = _load_items()
    return df.filter(pl.col("tier") == tier)


def get_items_by_slot(slot: str) -> pl.DataFrame:
    """Get items filtered by slot (weapon/vitality/spirit)."""
    df = _load_items()
    return df.filter(pl.col("slot").str.to_lowercase() == slot.lower())


def is_ability(item_id: int) -> bool:
    """Check if item is an ability upgrade."""
    lookup = _get_item_lookup()
    item = lookup.get(item_id)
    if item is None:
        return False
    slot = item.get("slot", "").lower()
    return slot == "ability"


def get_item_slot(item_id: int) -> Optional[str]:
    """Get the slot type for an item."""
    lookup = _get_item_lookup()
    item = lookup.get(item_id)
    if item is None:
        return None
    return item.get("slot")


def get_item_tier(item_id: int) -> Optional[int]:
    """Get the tier for an item."""
    lookup = _get_item_lookup()
    item = lookup.get(item_id)
    if item is None:
        return None
    return item.get("tier")


def get_all_item_ids() -> list[int]:
    """Get list of all item IDs."""
    df = _load_items()
    return df["id"].to_list()


def get_item_names_map() -> dict[int, str]:
    """Get mapping of item ID to name."""
    lookup = _get_item_lookup()
    return {item_id: info.get("name", "Unknown") for item_id, info in lookup.items()}
