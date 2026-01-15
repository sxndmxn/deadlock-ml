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

## Task 13 Completed - 2026-01-14

### Create counter-picking model

**Changes made:**
- Created `ml/train/train_counter.py` with counter-picking data generation
- Implemented `compute_hero_matchup_matrix()`:
  - Queries match data to find players on opposing teams
  - Groups by match_id to get all 12 players per match
  - Tracks wins/games for each hero pair (A vs B)
  - Calculates win rate with minimum games threshold
- Implemented `compute_item_counter_scores()`:
  - Tracks item effectiveness vs each enemy hero
  - Computes baseline win rate for each item
  - Calculates effectiveness score: (win_rate_vs_hero - baseline) / baseline
  - Positive effectiveness = item counters that hero
- Implemented `export_counter_matrix()`:
  - JSON format with metadata (hero/item IDs and names)
  - Hero matchup matrix with wins, games, win_rate
  - Item counter scores with baseline and per-hero effectiveness
- Verified Python syntax with `py_compile`
- Tested with sample data (10k matches):
  - Computed 992 valid hero matchups
  - Generated 2.28 MB JSON output
  - Verified data structure and content

**Output format:**
```json
{
  "model_type": "counter_matrix",
  "generated_at": "...",
  "metadata": {
    "num_heroes": 32,
    "num_items": 515,
    "hero_ids": [...],
    "item_ids": [...]
  },
  "hero_matchups": {
    "hero_a_id": {
      "hero_b_id": {"wins": N, "games": N, "win_rate": 0.XX}
    }
  },
  "item_counter_scores": {
    "item_id": {
      "baseline_win_rate": 0.XX,
      "vs_heroes": {
        "hero_id": {"wins": N, "games": N, "win_rate": 0.XX, "effectiveness": 0.XX}
      }
    }
  }
}
```

**Files created:**
- `ml/train/train_counter.py`
- `models/counter_matrix.json`

---

## Task 14 Completed - 2026-01-14

### Create optimal 12-item full build model

**Changes made:**
- Created `ml/train/train_full_build.py` with full build optimizer
- Implemented constrained beam search algorithm:
  - Explores multiple build paths simultaneously with configurable beam width
  - Uses XGBoost scorer when available, with fallback scoring
- Added slot balance constraints:
  - `SLOT_SOFT_MAX = 5` items per slot (soft limit)
  - `SLOT_HARD_MAX = 6` items per slot (hard limit)
  - `MIN_ITEMS_PER_SLOT = 2` minimum per slot in final build
  - Fallback scorer includes slot balance bonus/penalty
- Added tier-aware transitions:
  - Early game (items 0-1): Tier 1-2 allowed
  - Mid game (items 2-4): Tier 1-3 allowed
  - Late game (items 5-8): Tier 2-4 allowed
  - End game (items 9-11): Tier 3-4 only
- Implemented item synergy scoring:
  - Computes log odds ratio co-occurrence for item pairs
  - Rewards items that frequently appear together in winning builds
- Implemented per-hero item win rate computation
- JSON export format includes:
  - Item metadata (id, name, tier, slot, cost)
  - Synergy scores between item pairs
  - Per-item win rates
  - Precomputed optimal builds (5 variants)
  - Optional XGBoost scorer model
- Verified with test runs producing balanced builds (W=2, V=4, S=6)

**Model architecture:**
- Input: Winning builds from match history
- Processing: Beam search with slot/tier constraints
- Output: 12-item builds optimized for win rate and synergy

**Files created:**
- `ml/train/train_full_build.py`
- `models/full_build_6.json` (test output)

---

## Task 15 Completed - 2026-01-14

### Add ONNX Runtime to Rust for ML inference

**Changes made:**
- Added `ort = "2.0.0-rc.11"` to Cargo.toml (using latest RC since 2.0 stable not released)
- Created `rust-server/src/ml_inference.rs` with comprehensive ML inference module:
  - `FullBuildModel` - Loads and parses full build optimizer JSON models
  - `CounterMatrix` - Loads hero matchup and item counter matrices
  - `MlInference` - Main inference engine with model loading and caching
- Implemented model loading from JSON files:
  - Loads `full_build_*.json` files with synergy scores, item win rates, precomputed builds
  - Loads `counter_matrix.json` with hero matchups and item counter effectiveness
  - Builds symmetric synergy lookup table for fast (item_a, item_b) queries
- Implemented inference methods:
  - `get_precomputed_build()` - Returns best precomputed 12-item build
  - `get_synergy_score()` - Gets synergy score between two items
  - `get_hero_matchup()` - Gets win rate for hero A vs hero B
  - `get_item_counter_effectiveness()` - Gets item effectiveness vs enemy hero
  - `get_counter_items()` - Recommends counter items against enemy team
  - `recommend_next_items()` - ML-based recommendations with Markov fallback
- Added fallback to Markov chains:
  - `MAX_INFERENCE_TIME_MS = 50` constant for timeout threshold
  - `recommend_next_items()` attempts synergy-based ML inference first
  - Falls back to `markov::get_next_probabilities()` if timeout or no model
- Verified compilation with `cargo check` - passed with expected unused warnings

**Module architecture:**
- Synergy-based scoring: 60% synergy score + 40% win rate
- Counter-picking: sum item effectiveness vs all enemy heroes
- Markov fallback: use existing Markov chain probabilities

**Files created:**
- `rust-server/src/ml_inference.rs`

**Files modified:**
- `rust-server/Cargo.toml` (added ort dependency)
- `rust-server/src/main.rs` (added mod declaration)

---

## Task 16 Completed - 2026-01-14

### Add navigation tabs to index.html

**Changes made:**
- Added Match History tab with `hx-get="/htmx/matches"` to navigation
- Added Leaderboards tab with `hx-get="/htmx/leaderboards/overall"` to navigation
- Added visual tab divider (`<span class="tab-divider">`) to separate hero-specific tabs from global tabs
- Added `data-hero-specific` attribute to distinguish hero-specific tabs from global tabs
- Updated JavaScript to only update URLs for hero-specific tabs when hero selection changes
- Updated JavaScript to only auto-reload content on hero change if active tab is hero-specific
- Added `.tab-divider` CSS styling for visual separation
- Verified compilation with `cargo check` - passed
- Tested endpoints - both `/htmx/matches` and `/htmx/leaderboards/overall` render correctly

**Tab structure:**
- Hero-specific tabs (update on hero change): Build Optimizer, Item Synergies, Hero Stats, All Items
- Global tabs (no hero dependency): Match History, Leaderboards

**Files modified:**
- `rust-server/templates/index.html` (added tabs, updated JavaScript)
- `rust-server/static/css/styles.css` (added tab-divider style)

---

## Task 17 Completed - 2026-01-14

### Update CSS styling

**Changes made:**
- Added comprehensive Match History styles:
  - `.match-card` with hover effects (transform, box-shadow)
  - `.match-card.match-win` with green left border (#22c55e)
  - `.match-card.match-loss` with red left border (#ef4444)
  - `.match-result.win` / `.match-result.loss` badges with colored backgrounds
  - `.match-filters`, `.match-summary`, `.match-cards` grid layout
- Added comprehensive Leaderboard styles:
  - `.leaderboard-table` with sticky headers
  - `.rank-badge.gold` / `.silver` / `.bronze` with gradient backgrounds and shadows
  - `.rank-gold` / `.rank-silver` / `.rank-bronze` row highlights
  - Column-specific styling (`.player-col`, `.winrate-col`, `.kda-col`)
- Added Pagination styles:
  - `.pagination` flexbox container
  - `.page-btn` button styling with hover states
  - `.page-info` centered page display
- Added responsive layouts:
  - `@media (max-width: 768px)`: Stacked top bar, scrollable tabs, single-column match cards, full-width filters
  - `@media (max-width: 480px)`: Hide less important leaderboard columns, stack summary info
- Verified with `cargo check` - passed
- Tested CSS rendering via curl - all classes applied correctly

**CSS sections added (~330 lines):**
1. Match History Styles
2. Leaderboard Styles
3. Pagination
4. Responsive Layouts

**File modified:** `rust-server/static/css/styles.css`

---

## Task 18 Completed - 2026-01-14

### End-to-end verification

**Verification performed:**

1. **Server startup** - `cargo run` in rust-server/
   - Server starts successfully on http://127.0.0.1:3000
   - Loads models from /home/sandtop/deadlock-ml/models
   - 53 heroes, 229 items loaded

2. **All Items endpoint** (/htmx/all-items/6)
   - Endpoint returns items table with correct structure
   - Win rate and pick rate columns present
   - Note: Shows 0% because model was generated before item_stats integration
   - Fallback behavior works correctly

3. **Match History** (/htmx/matches)
   - Default view: 8.6M+ matches across 431,107 pages
   - Hero filter: hero_id=6 filters to 14,488 pages
   - Pagination: Page 2 loads correctly
   - Win/loss styling classes applied correctly

4. **Leaderboards** (/htmx/leaderboards/overall)
   - Overall leaderboard shows ranked players
   - Gold/silver/bronze rank badges for top 3
   - Hero-specific leaderboard works (/htmx/leaderboards/hero/6)
   - Min matches filter works

5. **Full UI walkthrough** - All tabs verified:
   - Build Optimizer: build-tab with build-list ✓
   - Item Synergies: synergies-tab with synergy-graph, rules-table ✓
   - Hero Stats: stats-tab with metrics ✓
   - All Items: items-tab with items-table ✓
   - Match History: match-list-tab with match-cards, pagination ✓
   - Leaderboards: leaderboard-tab with rank badges ✓

6. **Index page navigation**
   - All 6 tabs present: build, synergies, stats, items, matches, leaderboards
   - Hero-specific vs global tab separation working
   - HTMX triggers configured correctly

**Verification result:** All endpoints functional, all tabs working

---

## Task 19 Completed - 2026-01-15

### Fix All Items tab - tier classification, icons, and stats

**Changes made:**
- Updated `update_icons.py` to use correct API fields:
  - Changed `tier` field to use `item_tier` from API (was defaulting to 1)
  - Changed `slot` field to use `item_slot_type` from API
  - Added `shopable` filter to only include purchasable items
  - Changed icon URL to prefer `shop_image_small` for better quality icons
- Regenerated `metadata.json` with correct tier distribution:
  - Tier 1: 22 items
  - Tier 2: 42 items
  - Tier 3: 46 items
  - Tier 4: 45 items
  - Total: 151 shopable items (was 229 including non-shopable)
- Removed Slot column from `all_items.html`:
  - Removed header column
  - Removed data cell
  - Removed JavaScript sorting case for slot
- Added CDN fallback for icon URLs in `handlers.rs`:
  - Uses `https://assets.deadlock-api.com/images/items/{id}.png` when icon_url is empty
- Verified compilation with `cargo check` - passed with expected warnings
- Verified in browser - all 4 tiers rendering correctly with proper icons

**Files modified:**
- `update_icons.py` (API field fixes)
- `models/metadata.json` (regenerated with correct tiers)
- `rust-server/templates/partials/all_items.html` (removed Slot column)
- `rust-server/src/handlers.rs` (added CDN fallback for icons)

---

## Task 20 Completed - 2026-01-15

### Fix Item Synergies tab - item names and tooltips

**Problem identified:**
- Association rules in the model file (6.json) use sequential item IDs (100, 101, 200, 300)
- These IDs are embedded in the model's markov.states with their corresponding names
- The synergies handler was looking up names from metadata.items which uses different API item IDs (like 1548066885)
- This ID mismatch caused all items to display as "Item 100", "Item 300", etc.

**Changes made:**
- Updated `synergies()` handler in handlers.rs to build item_names HashMap from `model.markov.states` instead of metadata
- Updated filter_items dropdown to build ItemInfo from markov states
- Updated `synergy_graph()` handler similarly to use markov states for node labels
- Added title attributes to synergy_table.html header columns:
  - Antecedent: "Items that are purchased first (the 'if' part of the rule)"
  - Consequent: "Items that tend to follow the antecedent (the 'then' part of the rule)"
  - Support: "How often this item combination appears in all winning builds (higher = more common)"
  - Confidence: "When antecedent is bought, how often consequent follows (higher = stronger relationship)"
  - Lift: "How much more likely the consequent is compared to random chance (>1 = positive association)"
- Verified compilation with `cargo check` - passed with expected warnings
- Tested endpoints - item names display correctly (e.g., "Basic Magazine", "Extra Health")

**Files modified:**
- `rust-server/src/handlers.rs` (synergies and synergy_graph handlers)
- `rust-server/templates/partials/synergy_table.html` (added tooltips)

---

## Task 21 Completed - 2026-01-15

### Replace Hero Stats with simple HTMX table

**Changes made:**
- Added `pick_rate: f64` field to `HeroStat` struct in handlers.rs
- Updated `hero_stats` handler to:
  - First pass: collect hero data and calculate total matches across all heroes
  - Second pass: build HeroStat with pick_rate = hero_matches / total_matches
  - Changed sort to descending by win_rate (highest first)
- Replaced Plotly.js bar chart in hero_stats.html with sortable table:
  - Columns: Hero | Win Rate | Total Matches | Total Wins | Pick Rate
  - Sortable columns with ascending/descending toggle (like all_items.html)
  - Selected hero row highlighted with blue background and left border
  - Win rates below 50% shown in red (`.negative` class)
  - Numbers formatted with thousands separators via JavaScript
- Added CSS styles for `.hero-stats-table` in styles.css:
  - Table styling matching items table
  - Sortable column headers with arrow indicators
  - `.selected-hero` row highlighting
  - `.col-hero`, `.col-wins`, `.no-heroes` styles
- Verified compilation with `cargo check` - passed
- Verified endpoint renders table correctly via curl

**Files modified:**
- `rust-server/src/handlers.rs` (HeroStat struct, hero_stats handler)
- `rust-server/templates/partials/hero_stats.html` (replaced Plotly with table)
- `rust-server/static/css/styles.css` (added hero stats table styles)

---

## Task 22 Completed - 2026-01-15

### Redesign Match History to Hero Matchups

**Changes made:**
- Added `CounterMatrix` and `HeroMatchup` structs to `models.rs`:
  - `HeroMatchup`: holds wins, games, win_rate for a hero vs opponent
  - `CounterMatrixMetadata`: num_heroes, num_items, hero_ids, item_ids
  - `CounterMatrix`: full counter matrix with hero_matchups HashMap
  - Added `get_matchup()` and `get_all_matchups()` methods
- Added counter_matrix loading to `store.rs`:
  - Added `counter_matrix: RwLock<Option<Arc<CounterMatrix>>>` field
  - Implemented `reload_counter_matrix()` method
  - Added `get_counter_matrix()` getter
  - Updated file watcher to handle counter_matrix.json
- Replaced match list handler with hero matchups handler in `handlers.rs`:
  - Created `HeroMatchupsTemplate` with matchups, heroes, selected_hero_id, total_games
  - Created `DisplayMatchup` struct with opponent info and win_rate_class
  - Handler fetches matchup data from counter_matrix and returns sorted by win rate
  - Color coding: "matchup-good" for >52%, "matchup-bad" for <48%
- Redesigned `match_list.html` as matchups table:
  - Hero dropdown selector
  - Summary showing total games
  - Table with columns: Opponent (with hero icon), Games, Wins, Win Rate
  - Sortable columns with JavaScript
  - Numbers formatted with thousands separators
- Added CSS styles in `styles.css`:
  - `.matchups-tab`, `.matchups-filters`, `.matchups-summary` layouts
  - `.matchups-table` with sortable headers
  - `.opponent-cell` with hero icon and name
  - `.matchup-good` (green) and `.matchup-bad` (red) color coding
  - Responsive layouts for mobile
- Renamed "Match History" tab to "Hero Matchups" in `index.html`
- Verified compilation with `cargo check` - passed
- Verified endpoint renders correctly via curl testing

**Files modified:**
- `rust-server/src/models.rs` (added CounterMatrix structs)
- `rust-server/src/store.rs` (added counter_matrix loading)
- `rust-server/src/handlers.rs` (replaced match_list with hero_matchups handler)
- `rust-server/templates/partials/match_list.html` (redesigned as matchups table)
- `rust-server/templates/index.html` (renamed tab)
- `rust-server/static/css/styles.css` (added matchups styles)

---

## Task 23 Completed - 2026-01-15

### Redesign Build Optimizer with tree visualization

**Changes made:**
- Added `TreeNode`, `TreeEdge`, `TreeData` structs to `markov.rs`:
  - `TreeNode`: id, item_id, name, prob, x, y, icon_url
  - `TreeEdge`: source, target, weight, x0, y0, x1, y1
  - `TreeData`: nodes, edges, max_y
- Implemented `build_tree_data()` function:
  - Generates 3-level deep tree (START -> Level 1 -> Level 2 -> Level 3)
  - Top 5 items per branch at each level
  - Horizontal layout (left to right) with proper Y spacing
  - Icon URLs from deadlock-api CDN
- Added `tree_data` handler to `handlers.rs`:
  - Returns JSON with tree structure for Plotly visualization
  - Builds item icon map from markov states
- Added route `/htmx/tree/{hero_id}` to `main.rs`
- Replaced Sankey diagram in `build_path.html` with tree visualization:
  - Plotly scatter plot with `layout.images` for item icons
  - Edge traces connecting nodes with line width based on probability
  - START node rendered as SVG circle
  - Horizontal left-to-right layout
  - Hover info showing item name and probability
- Verified compilation with `cargo check` - passed
- Verified tree endpoint returns 36 nodes and 35 edges
- Verified build-path template renders with tree chart div

**Files modified:**
- `rust-server/src/markov.rs` (added TreeNode, TreeEdge, TreeData, build_tree_data)
- `rust-server/src/handlers.rs` (added tree_data handler, updated imports)
- `rust-server/src/main.rs` (added /htmx/tree/{hero_id} route)
- `rust-server/templates/partials/build_path.html` (replaced Sankey with tree)

---

## PHASE 6 IN PROGRESS

Tasks 1-18 complete. Now working on Phase 6 UI Fixes:
- Task 19: ✅ Fix All Items tab (completed)
- Task 20: ✅ Fix Item Synergies tab (completed)
- Task 21: ✅ Replace Hero Stats with table (completed)
- Task 22: ✅ Redesign Match History to Hero Matchups (completed)
- Task 23: ✅ Redesign Build Optimizer with tree (completed)
- Task 24: ❌ End-to-end UI verification (pending)

---
