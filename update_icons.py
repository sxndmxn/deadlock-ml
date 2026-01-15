#!/usr/bin/env python3
"""Fetch hero and item icon URLs from the assets API and update metadata.json."""

import json
import subprocess
from pathlib import Path

ASSETS_API = "https://assets.deadlock-api.com/v2"
MODELS_DIR = Path(__file__).parent / "models"

def fetch_json(url):
    """Fetch JSON from URL using curl."""
    result = subprocess.run(
        ["curl", "-s", url],
        capture_output=True,
        text=True,
        check=True
    )
    return json.loads(result.stdout)

def fetch_heroes():
    """Fetch all heroes with their image URLs."""
    data = fetch_json(f"{ASSETS_API}/heroes")
    heroes = []
    for hero in data:
        heroes.append({
            "id": hero["id"],
            "name": hero["name"],
            "icon_url": hero.get("images", {}).get("icon_image_small", "")
        })
    return sorted(heroes, key=lambda h: h["name"])

def fetch_items():
    """Fetch all upgrade items with their image URLs."""
    data = fetch_json(f"{ASSETS_API}/items/by-type/upgrade")
    items = []
    for item in data:
        # Only include shopable items (skip internal/hidden items)
        if not item.get("shopable", False):
            continue

        # Get slot from item_slot_type field
        slot = item.get("item_slot_type", "vitality")

        # Get tier from item_tier field (defaults to 1)
        tier = item.get("item_tier", 1)

        # Prefer shop_image_small for icons
        icon_url = item.get("shop_image_small", "") or item.get("image", "")

        items.append({
            "id": item["id"],
            "name": item["name"],
            "slot": slot,
            "tier": tier,
            "icon_url": icon_url
        })
    return items

def main():
    print("Fetching heroes from assets API...")
    heroes = fetch_heroes()
    print(f"  Found {len(heroes)} heroes")

    print("Fetching items from assets API...")
    items = fetch_items()
    print(f"  Found {len(items)} items")

    # Load existing metadata
    metadata_path = MODELS_DIR / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {}

    # Update with new data
    metadata["heroes"] = heroes
    metadata["items"] = items
    metadata["generated_at"] = "2026-01-14T19:00:00Z"

    # Save
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Updated {metadata_path}")
    print(f"  Heroes: {len(heroes)}")
    print(f"  Items: {len(items)}")

if __name__ == "__main__":
    main()
