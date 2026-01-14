#!/bin/bash
#
# Daily refresh script for Deadlock Build Optimizer
# Fetches new data and recomputes ML models
#
# Usage: ./scripts/refresh.sh [--shards START-END]
#
# Cron setup (daily at 4am):
#   0 4 * * * cd /path/to/project && ./scripts/refresh.sh --shards 48-50 >> logs/refresh.log 2>&1
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"
BACKUP_DIR="$PROJECT_DIR/models_backup"

# Default shard range (recent data)
SHARDS="${1:-48-50}"
if [[ "$1" == "--shards" ]]; then
    SHARDS="$2"
fi

# Log start time
echo "=========================================="
echo "Refresh started at $(date)"
echo "Project directory: $PROJECT_DIR"
echo "Shard range: $SHARDS"
echo "=========================================="

# Change to project directory
cd "$PROJECT_DIR"

# Backup existing models
if [ -d "$MODELS_DIR" ] && [ "$(ls -A "$MODELS_DIR" 2>/dev/null)" ]; then
    echo ""
    echo "Backing up existing models..."
    rm -rf "$BACKUP_DIR"
    cp -r "$MODELS_DIR" "$BACKUP_DIR"
    echo "Backup saved to $BACKUP_DIR"
fi

# Fetch new data
echo ""
echo "Fetching data..."
uv run python scripts/fetch_data.py --shards "$SHARDS" --force

# Recompute models
echo ""
echo "Recomputing ML models..."
uv run python ml/precompute.py

# Log end time
echo ""
echo "=========================================="
echo "Refresh completed at $(date)"
echo "=========================================="

exit 0
