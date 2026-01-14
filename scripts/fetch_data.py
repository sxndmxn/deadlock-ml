#!/usr/bin/env python3
"""
Data fetch script for Deadlock Build Optimizer.
Downloads parquet files from deadlock-api.com.
"""

import argparse
import asyncio
import sys
from pathlib import Path

import httpx

BASE_URL = "https://files.deadlock-api.com/api/buckets/db-snapshot/objects/public"
DATA_DIR = Path(__file__).parent.parent / "data"
MATCH_DIR = DATA_DIR / "match_metadata"

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
) -> bool:
    """Download a file with progress display. Returns True on success."""
    if dest.exists() and not force:
        print(f"  Skipping {dest.name} (exists, use --force to redownload)")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        async with client.stream("GET", url) as response:
            if response.status_code == 404:
                print(f"  Not found: {url}")
                return False
            response.raise_for_status()

            total = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(dest, "wb") as f:
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total > 0:
                        pct = downloaded / total * 100
                        bar_len = 30
                        filled = int(bar_len * downloaded / total)
                        bar = "=" * filled + "-" * (bar_len - filled)
                        size_mb = downloaded / (1024 * 1024)
                        total_mb = total / (1024 * 1024)
                        print(
                            f"\r  [{bar}] {pct:5.1f}% ({size_mb:.1f}/{total_mb:.1f} MB)",
                            end="",
                            flush=True,
                        )

            print(f"\r  Downloaded {dest.name}" + " " * 40)
            return True

    except httpx.HTTPError as e:
        print(f"\n  Error downloading {dest.name}: {e}")
        if dest.exists():
            dest.unlink()
        return False


async def fetch_metadata(client: httpx.AsyncClient, force: bool) -> bool:
    """Download heroes.parquet and items.parquet."""
    print("Fetching metadata files...")
    success = True

    for filename in METADATA_FILES:
        url = f"{BASE_URL}/{filename}"
        dest = DATA_DIR / filename
        if not await download_file(client, url, dest, force):
            success = False

    return success


async def fetch_shards(
    client: httpx.AsyncClient,
    shards: list[int],
    force: bool,
) -> bool:
    """Download match_player shard files."""
    print(f"Fetching match_player shards: {shards[0]}-{shards[-1]}...")
    success = True

    for shard in shards:
        filename = f"match_player_{shard}.parquet"
        url = f"{BASE_URL}/match_metadata/{filename}"
        dest = MATCH_DIR / filename

        if not await download_file(client, url, dest, force):
            success = False

    return success


async def main(args: argparse.Namespace) -> int:
    """Main entry point."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MATCH_DIR.mkdir(parents=True, exist_ok=True)

    timeout = httpx.Timeout(30.0, read=300.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        # Always fetch metadata
        if not await fetch_metadata(client, args.force):
            print("Warning: Some metadata files failed to download")

        # Fetch shards if specified
        if args.shards:
            shards = parse_shard_range(args.shards)
            if not await fetch_shards(client, shards, args.force):
                print("Warning: Some shard files failed to download")
                return 1

    print("Done!")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Deadlock data files from deadlock-api.com"
    )
    parser.add_argument(
        "--shards",
        type=str,
        help="Shard range to download (e.g., '46-50' or '48')",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they exist",
    )

    args = parser.parse_args()
    sys.exit(asyncio.run(main(args)))
