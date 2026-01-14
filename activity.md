# Activity Log

## Session Start - 2026-01-14

### Previously Completed (Phase 1.1-1.3)
- Phase 1.1: Data download script (scripts/fetch_dumps.py) - DONE
- Phase 1.2: Item statistics module (ml/item_stats.py) - DONE
- Phase 1.3: Integrate item_stats into precompute.py - DONE

### Current State
- Completed Task 1 (Phase 1.4): Update Rust models.rs with ItemStats struct
- 17 tasks remaining in plan.md

---

## Task 1 Completed - 2026-01-14

### Update Rust models.rs with ItemStats struct

**Changes made:**
- Added `ItemStats` struct with fields: `item_id`, `total_matches`, `wins`, `win_rate`, `pick_rate`
- Added `#[serde(default)] item_stats: Vec<ItemStats>` field to `HeroModel` struct
- Verified compilation with `cargo check` - passed with only expected unused field warnings

**File modified:** `rust-server/src/models.rs`

---

## Task 2 Completed - 2026-01-14

### Update Rust handlers.rs to use real item stats

**Changes made:**
- Replaced mock random data generation in `all_items` handler (lines 380-418)
- Added lookup of hero model via `store.get_hero(hero_id)`
- Built `item_stats_map: HashMap<i64, &ItemStats>` from `hero_model.item_stats`
- Updated `DisplayItem` mapping to use real stats from lookup with fallback to zeros
- Verified compilation with `cargo check` - passed with only expected unused field warnings

**File modified:** `rust-server/src/handlers.rs`

---

## Task 3 Completed - 2026-01-14

### Add DuckDB dependency to Rust server

**Changes made:**
- Added `duckdb = { version = "1.0", features = ["bundled"] }` to Cargo.toml
- Verified `parking_lot = "0.12"` was already present (no changes needed)
- Ran `cargo check` - compilation succeeded with only expected unused field warnings
- DuckDB bundled feature compiles the full DuckDB C++ library (~18 min first build)

**File modified:** `rust-server/Cargo.toml`

---

## Task 4 Completed - 2026-01-14

### Create database module (db.rs)

**Changes made:**
- Created `rust-server/src/db.rs` with DuckDB integration for match history queries
- Added `MatchSummary` struct with fields: `match_id`, `hero_id`, `account_id`, `won`, `kills`, `deaths`, `assists`, `net_worth`
- Added `MatchListQuery` struct for filtering with: `hero_id`, `account_id`, `won`, `limit`, `offset`
- Implemented `Database` struct with:
  - `new(data_dir)` - creates in-memory DuckDB connection with parquet support
  - `get_match_list(query)` - queries match_metadata parquet files with dynamic filters
  - `get_match_count(query)` - returns total count for pagination
- Added `mod db;` declaration to `main.rs`
- Verified compilation with `cargo check` - passed with only expected unused field warnings

**Files modified:**
- `rust-server/src/db.rs` (new file)
- `rust-server/src/main.rs` (added mod declaration)

---

## Task 5 Completed - 2026-01-14

### Add match routes to main.rs

**Changes made:**
- Created `AppState` struct holding both `Arc<ModelStore>` and `Arc<Database>`
- Updated all existing handlers to use `State(state): State<AppState>` instead of `State(store): State<Arc<ModelStore>>`
- Added data_dir initialization for DuckDB data directory
- Initialized Database in main() and created shared AppState
- Added routes:
  - `/htmx/matches` -> `handlers::match_list`
  - `/htmx/matches/{match_id}` -> `handlers::match_detail`
- Added placeholder handlers `match_list` and `match_detail` with stub HTML responses
- Added `MatchListQuery` struct with `hero_id`, `account_id`, `outcome`, `page` params
- Verified compilation with `cargo check` - passed with only expected unused field warnings

**Files modified:**
- `rust-server/src/main.rs` (AppState struct, database init, routes)
- `rust-server/src/handlers.rs` (updated all handlers to use AppState, added match handlers)

---
