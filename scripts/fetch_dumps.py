#!/usr/bin/env python3
"""
Data fetch script for Deadlock Build Optimizer.
Downloads parquet files from deadlock-api.com.

Supports two modes:
- Full dumps: Download complete database snapshots for offline resilience
- Incremental: Download sharded match_player files for updates
"""

import argparse
import asyncio
import sys
from pathlib import Path

import httpx

# API endpoints
DUMPS_URL = "https://files.deadlock-api.com/buckets/db-snapshot/public"
SHARDS_URL = "https://files.deadlock-api.com/api/buckets/db-snapshot/objects/public"

DATA_DIR = Path(__file__).parent.parent / "data"
MATCH_DIR = DATA_DIR / "match_metadata"

# Full dump files (primary data source)
FULL_DUMP_FILES = [
    ("player_match_history.parquet", "Historical match records (~5.7GB)"),
    ("active_matches.parquet", "Current match data (~3.4GB)"),
    ("player_card.parquet", "Player profiles (~600KB)"),
    ("match_salts.parquet", "Match identifiers (~200MB)"),
    ("heroes.parquet", "Hero metadata (~1KB)"),
    ("items.parquet", "Item metadata (~12KB)"),
]

# Metadata files (always downloaded)
METADATA_FILES = ["heroes.parquet", "items.parquet"]


def parse_shard_range(shard_arg: str) -> list[int]:
    """Parse shard range string like '46-50' or '48' into list of ints."""
    if "-" in shard_arg:
        start, end = shard_arg.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(shard_arg)]


async def download_file(
    client: httpx.AsyncClient,
    url: str,
    dest: Path,
    force: bool = False,
    description: str = "",
) -> bool:
    """Download a file with progress display. Returns True on success."""
    if dest.exists() and not force:
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  Skipping {dest.name} ({size_mb:.1f} MB exists, use --force to redownload)")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)

    if description:
        print(f"  {description}")

    try:
        async with client.stream("GET", url) as response:
            if response.status_code == 404:
                print(f"  Not found: {url}")
                return False
            response.raise_for_status()

            total = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(dest, "wb") as f:
                async for chunk in response.aiter_bytes(chunk_size=65536):
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total > 0:
                        pct = downloaded / total * 100
                        bar_len = 40
                        filled = int(bar_len * downloaded / total)
                        bar = "█" * filled + "░" * (bar_len - filled)
                        size_mb = downloaded / (1024 * 1024)
                        total_mb = total / (1024 * 1024)
                        print(
                            f"\r  [{bar}] {pct:5.1f}% ({size_mb:.1f}/{total_mb:.1f} MB)",
                            end="",
                            flush=True,
                        )

            print(f"\r  ✓ Downloaded {dest.name}" + " " * 50)
            return True

    except httpx.HTTPError as e:
        print(f"\n  ✗ Error downloading {dest.name}: {e}")
        if dest.exists():
            dest.unlink()
        return False


async def fetch_full_dumps(
    client: httpx.AsyncClient,
    force: bool,
    skip_large: bool = False,
) -> bool:
    """Download full database dumps for offline resilience."""
    print("\n=== Downloading Full Database Dumps ===")
    print(f"Source: {DUMPS_URL}")
    print()

    success = True

    for filename, description in FULL_DUMP_FILES:
        # Skip large files if requested
        if skip_large and filename in ("player_match_history.parquet", "active_matches.parquet"):
            print(f"  Skipping {filename} (--skip-large)")
            continue

        url = f"{DUMPS_URL}/{filename}"
        dest = DATA_DIR / filename

        if not await download_file(client, url, dest, force, description):
            success = False

    return success


async def fetch_metadata(client: httpx.AsyncClient, force: bool) -> bool:
    """Download heroes.parquet and items.parquet from shards API."""
    print("\n=== Fetching Metadata Files ===")
    success = True

    for filename in METADATA_FILES:
        url = f"{SHARDS_URL}/{filename}"
        dest = DATA_DIR / filename
        if not await download_file(client, url, dest, force):
            success = False

    return success


async def fetch_shards(
    client: httpx.AsyncClient,
    shards: list[int],
    force: bool,
) -> bool:
    """Download match_player shard files for incremental updates."""
    print(f"\n=== Fetching Match Player Shards: {shards[0]}-{shards[-1]} ===")
    success = True

    for shard in shards:
        filename = f"match_player_{shard}.parquet"
        url = f"{SHARDS_URL}/match_metadata/{filename}"
        dest = MATCH_DIR / filename

        if not await download_file(client, url, dest, force):
            success = False

    return success


async def main(args: argparse.Namespace) -> int:
    """Main entry point."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MATCH_DIR.mkdir(parents=True, exist_ok=True)

    # Longer timeout for large files
    timeout = httpx.Timeout(60.0, read=600.0, connect=30.0)

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        if args.full:
            # Download full database dumps
            if not await fetch_full_dumps(client, args.force, args.skip_large):
                print("\nWarning: Some dump files failed to download")
                return 1
        else:
            # Default: fetch metadata
            if not await fetch_metadata(client, args.force):
                print("Warning: Some metadata files failed to download")

            # Fetch shards if specified
            if args.shards:
                shards = parse_shard_range(args.shards)
                if not await fetch_shards(client, shards, args.force):
                    print("Warning: Some shard files failed to download")
                    return 1

    print("\n✓ Done!")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Deadlock data files from deadlock-api.com",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download full database dumps (recommended for offline use)
  python scripts/fetch_dumps.py --full

  # Download full dumps but skip large files (>1GB)
  python scripts/fetch_dumps.py --full --skip-large

  # Download specific match shards for incremental updates
  python scripts/fetch_dumps.py --shards 48-60

  # Force re-download existing files
  python scripts/fetch_dumps.py --full --force
        """,
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Download full database dumps (player_match_history, active_matches, etc.)",
    )
    parser.add_argument(
        "--shards",
        type=str,
        help="Shard range to download (e.g., '46-50' or '48')",
    )
    parser.add_argument(
        "--skip-large",
        action="store_true",
        help="Skip large files (>1GB) when using --full",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they exist",
    )

    args = parser.parse_args()

    if not args.full and not args.shards:
        print("Note: No action specified. Use --full for dumps or --shards for incremental.")
        print("Run with --help for more options.")
        sys.exit(0)

    sys.exit(asyncio.run(main(args)))
