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

## Task 6 Completed - 2026-01-14

### Add match list handler

**Changes made:**
- Updated `MatchListQuery` struct with `hero_id`, `account_id`, `outcome`, `page` params (already existed from Task 5)
- Created `MatchListTemplate` struct with:
  - `matches: Vec<DisplayMatch>` - list of matches to display
  - `heroes: Vec<HeroInfo>` - for hero filter dropdown
  - `total_matches: u64` - total count for pagination info
  - `current_page: u32` / `total_pages: u32` - pagination state
  - `selected_hero_id: i32` - current hero filter (0 = all)
  - `selected_outcome: String` - current outcome filter ("", "win", "loss")
- Created `DisplayMatch` struct with:
  - `match_id`, `hero_id`, `hero_name`, `account_id`
  - `won`, `kills`, `deaths`, `assists`, `net_worth`
  - `kda` - formatted KDA string with ratio
- Implemented full `match_list` handler:
  - Parses query params and converts outcome string to boolean
  - Builds `DbMatchListQuery` for database layer
  - Queries database for matches and count
  - Converts to display format with hero name lookup and KDA calculation
  - Handles database errors gracefully
- Created `templates/partials/match_list.html` template:
  - Hero and outcome filter dropdowns with HTMX triggers
  - Match cards with win/loss styling classes
  - KDA and net worth display
  - Pagination controls with filter preservation
- Verified compilation with `cargo check` - passed with only expected unused field warnings

**Files modified:**
- `rust-server/src/handlers.rs` (MatchListTemplate, DisplayMatch, match_list handler)
- `rust-server/templates/partials/match_list.html` (new file)

---

## Task 7 Completed - 2026-01-14

### Create match list template (verification)

**Verification performed:**
- Template `templates/partials/match_list.html` was already created in Task 6
- Ran `cargo check` - compilation passed successfully
- Started server and tested `/htmx/matches` endpoint via curl
- Template renders correctly with:
  - Hero filter dropdown populated with all 53 heroes
  - Outcome filter dropdown (All Outcomes, Wins, Losses)
  - Match cards displaying with `match-win`/`match-loss` CSS classes
  - Match summary showing 8.6M+ matches from DuckDB
  - Pagination controls with page 1 of 431,107 pages
  - KDA calculation showing ratio (e.g., "8/1/10 (18.00)")
  - Net worth displayed in souls
  - HTMX triggers on filter dropdowns (hx-get, hx-target)

**Template features verified:**
1. Filter dropdowns with `hx-get="/htmx/matches"` triggers
2. Match cards with conditional win/loss styling via Tera templating
3. Pagination controls that preserve filter state in URLs
4. Empty state message for no matches

**Files verified:**
- `rust-server/templates/partials/match_list.html`

---

## Task 8 Completed - 2026-01-14

### Add leaderboard database queries

**Changes made:**
- Added `PlayerRanking` struct with fields:
  - `account_id: i64`
  - `matches: u32`, `wins: u32`, `win_rate: f64`
  - `total_kills: u32`, `total_deaths: u32`, `total_assists: u32`
  - `kda: f64`
- Added `LeaderboardSort` enum with variants: `WinRate`, `Matches`, `Kills`, `Kda`
  - Includes `from_str()` parser and `to_sql_order()` for query building
- Added `LeaderboardQuery` struct with:
  - `sort_by: LeaderboardSort`
  - `min_matches: u32` (minimum games filter)
  - `limit: u32`, `offset: u32` (pagination)
- Implemented `get_overall_leaderboard()`:
  - Aggregates all players across all heroes
  - Groups by account_id with SUM/AVG for stats
  - Calculates KDA with zero-death handling
  - Supports dynamic sorting and pagination
- Implemented `get_hero_leaderboard(hero_id)`:
  - Same aggregation but filtered by specific hero
  - Parameterized hero_id for safe queries
- Implemented `get_leaderboard_count()`:
  - Returns count of unique players meeting min_matches
  - Supports optional hero_id filter for pagination
- Verified compilation with `cargo check` - passed

**File modified:** `rust-server/src/db.rs`

---

## Task 9 Completed - 2026-01-14

### Add leaderboard routes and handlers

**Changes made:**
- Added routes to `main.rs`:
  - `/htmx/leaderboards/overall` -> `handlers::leaderboard_overall`
  - `/htmx/leaderboards/hero/{hero_id}` -> `handlers::leaderboard_hero`
- Added imports for `LeaderboardQuery` and `LeaderboardSort` from db module
- Created `LeaderboardParams` query struct with:
  - `sort: Option<String>` (win_rate, matches, kills, kda)
  - `min_matches: Option<u32>`
  - `page: Option<u32>`
- Created `DisplayRanking` struct for template rendering:
  - `rank`, `account_id`, `matches`, `wins`, `win_rate`, `total_kills`, `total_deaths`, `total_assists`, `kda`
- Created `LeaderboardTemplate` struct with Askama template binding
- Implemented `leaderboard_overall` handler:
  - Parses query params for sorting and filtering
  - Queries DuckDB for aggregated player stats
  - Converts to display format with percentage formatting
  - Handles pagination (50 players per page)
- Implemented `leaderboard_hero` handler:
  - Same as overall but filters by hero_id
  - Lower default min_matches (5 vs 10)
- Created `templates/partials/leaderboard.html`:
  - Hero filter dropdown
  - Sort dropdown (Win Rate, Matches, Total Kills, KDA)
  - Min matches filter (5, 10, 25, 50, 100)
  - Rankings table with rank badges (gold/silver/bronze)
  - Pagination controls
- Verified compilation with `cargo check` - passed

**Files modified:**
- `rust-server/src/main.rs` (routes)
- `rust-server/src/handlers.rs` (handlers, templates, structs)
- `rust-server/templates/partials/leaderboard.html` (new file)

---

## Task 10 Completed - 2026-01-14

### Create leaderboard template (verification)

**Verification performed:**
- Template `templates/partials/leaderboard.html` was created in Task 9
- Verified all required features are present:
  1. Rankings table with columns: Rank, Player, Matches, Wins, Win Rate, KDA
  2. Rank badges for top 3 (gold/silver/bronze with `rank-badge` CSS classes)
  3. Sort controls dropdown (Win Rate, Matches, Total Kills, KDA)
  4. Hero filter dropdown for hero-specific leaderboards
  5. Min matches filter (5, 10, 25, 50, 100)
  6. Pagination controls with filter preservation
- Ran `cargo check` and `cargo build` - both passed
  - Askama validates templates at compile time, confirming template syntax is correct
- Server starts and responds on leaderboard endpoint
  - Query is slow due to aggregation over 8.6M+ rows (expected)

**Template features verified:**
1. Rankings table with conditional rank badge styling via `else if` conditionals
2. HTMX triggers on all filter dropdowns with `hx-include` for state preservation
3. Hero mastery via hero_id dropdown switching between `/overall` and `/hero/{id}`
4. Pagination controls preserve sort and filter state in URL params

**Files verified:**
- `rust-server/templates/partials/leaderboard.html`

---

## Task 11 Completed - 2026-01-14

### Create dynamic build path ML model

**Changes made:**
- Created `ml/train/` directory structure
- Created `ml/train/__init__.py`
- Created `ml/train/train_build_path.py` with:
  - XGBoost multi-class classifier for next-item prediction
  - GPU acceleration support via `device: cuda` parameter
  - Feature engineering:
    - `owned_items`: one-hot encoded (num_items dimensions)
    - `game_time`: normalized to 0-1 range (max 45 min)
    - `enemy_heroes`: one-hot placeholder (num_heroes dimensions)
  - Training data preparation from parquet match history
  - Train/validation split (80/20) with early stopping
  - JSON export format for Rust consumption including:
    - Feature configuration (item/hero index mappings)
    - Full XGBoost model in JSON format
- Added `xgboost>=2.0.0` to pyproject.toml dependencies
- Verified Python syntax with `py_compile`

**Model architecture:**
- Input: owned_items_one_hot + game_time + enemy_heroes_one_hot
- Output: softmax probabilities over all items
- Uses `multi:softprob` objective for multi-class classification

**Files created:**
- `ml/train/__init__.py`
- `ml/train/train_build_path.py`

**Files modified:**
- `pyproject.toml` (added xgboost dependency)

---

## Task 12 Completed - 2026-01-14

### Create win probability predictor

**Changes made:**
- Created `ml/train/train_win_predictor.py` with:
  - XGBoost binary classifier for win prediction
  - `binary:logistic` objective with AUC evaluation
  - GPU acceleration support via `device: cuda`
  - Feature engineering:
    - `hero_one_hot`: player's hero (num_heroes dimensions)
    - `weapon_item_count`: normalized count of weapon items
    - `vitality_item_count`: normalized count of vitality items
    - `spirit_item_count`: normalized count of spirit items
    - `net_worth_normalized`: normalized to 0-1 range (max 100k)
  - Item categorization by slot (weapon/vitality/spirit)
  - Train/validation split (80/20) with early stopping
  - JSON export format for Rust consumption
- Verified Python syntax with `py_compile`

**Model architecture:**
- Input: hero_one_hot + item_category_counts + net_worth
- Output: probability of winning (0-1)
- Uses `binary:logistic` objective
- Evaluation metrics: logloss, AUC

**Files created:**
- `ml/train/train_win_predictor.py`

---
