# Task Plan

```json
{
  "tasks": [
    {
      "id": 1,
      "category": "phase1",
      "description": "Update Rust models.rs with ItemStats struct",
      "steps": [
        "Add ItemStats struct with item_id, total_matches, wins, win_rate, pick_rate fields",
        "Add #[serde(default)] item_stats: Vec<ItemStats> to HeroModel struct",
        "Run cargo check to verify compilation"
      ],
      "passes": true
    },
    {
      "id": 2,
      "category": "phase1",
      "description": "Update Rust handlers.rs to use real item stats",
      "steps": [
        "Replace mock random data in all_items handler (~line 380-418)",
        "Build item_stats_map HashMap from hero_model.item_stats",
        "Map DisplayItem to use real stats from lookup",
        "Run cargo check to verify compilation"
      ],
      "passes": true
    },
    {
      "id": 3,
      "category": "phase2",
      "description": "Add DuckDB dependency to Rust server",
      "steps": [
        "cd /home/sandtop/deadlock-ml/rust-server && cargo add duckdb --features bundled",
        "Add parking_lot dependency for Mutex",
        "Run cargo check to verify dependencies resolve"
      ],
      "passes": true
    },
    {
      "id": 4,
      "category": "phase2",
      "description": "Create database module (db.rs)",
      "steps": [
        "Create rust-server/src/db.rs with Database struct",
        "Implement new() with in-memory DuckDB connection",
        "Implement get_match_list() with hero_id, account_id, won filters",
        "Add MatchSummary struct with match_id, hero_id, account_id, won, kills, deaths, assists, net_worth",
        "Run cargo check"
      ],
      "passes": true
    },
    {
      "id": 5,
      "category": "phase2",
      "description": "Add match routes to main.rs",
      "steps": [
        "Add mod db; declaration",
        "Add Database to AppState",
        "Add route /htmx/matches for match_list handler",
        "Add route /htmx/matches/:match_id for match_detail handler",
        "Run cargo check"
      ],
      "passes": true
    },
    {
      "id": 6,
      "category": "phase2",
      "description": "Add match list handler",
      "steps": [
        "Add MatchListQuery struct with hero_id, account_id, outcome, page params",
        "Implement match_list handler querying Database",
        "Create MatchListTemplate struct",
        "Run cargo check"
      ],
      "passes": true
    },
    {
      "id": 7,
      "category": "phase2",
      "description": "Create match list template",
      "steps": [
        "Create templates/partials/match_list.html",
        "Add filter dropdowns for hero and outcome with hx-get triggers",
        "Add match cards with win/loss styling",
        "Add pagination controls",
        "Verify template renders"
      ],
      "passes": true
    },
    {
      "id": 8,
      "category": "phase3",
      "description": "Add leaderboard database queries",
      "steps": [
        "Add get_overall_leaderboard() to db.rs",
        "Add get_hero_leaderboard(hero_id) to db.rs",
        "Add PlayerRanking struct with account_id, matches, wins, win_rate, total_kills, kda",
        "Support sort_by, time_range, min_matches params",
        "Run cargo check"
      ],
      "passes": true
    },
    {
      "id": 9,
      "category": "phase3",
      "description": "Add leaderboard routes and handlers",
      "steps": [
        "Add route /htmx/leaderboards/overall",
        "Add route /htmx/leaderboards/hero/:hero_id",
        "Implement leaderboard_overall handler",
        "Implement leaderboard_hero handler",
        "Run cargo check"
      ],
      "passes": true
    },
    {
      "id": 10,
      "category": "phase3",
      "description": "Create leaderboard template",
      "steps": [
        "Create templates/partials/leaderboard.html",
        "Add rankings table with rank badges (gold/silver/bronze)",
        "Add sort/filter controls",
        "Add hero mastery tab option",
        "Verify template renders"
      ],
      "passes": true
    },
    {
      "id": 11,
      "category": "phase4",
      "description": "Create dynamic build path ML model",
      "steps": [
        "Create ml/train/train_build_path.py",
        "Implement XGBoost classifier with GPU acceleration",
        "Features: owned_items (one-hot), game_time, enemy_heroes",
        "Export model to JSON for Rust consumption",
        "Verify training completes and model exports"
      ],
      "passes": true
    },
    {
      "id": 12,
      "category": "phase4",
      "description": "Create win probability predictor",
      "steps": [
        "Create ml/train/train_win_predictor.py",
        "Implement XGBoost binary classifier",
        "Features: ally_heroes, enemy_heroes, item_counts, gold_advantage",
        "Export to models/win_predictor.json",
        "Verify training and export"
      ],
      "passes": true
    },
    {
      "id": 13,
      "category": "phase4",
      "description": "Create counter-picking model",
      "steps": [
        "Create ml/train/train_counter.py",
        "Precompute hero matchup matrix: win_rate[hero_A][vs_hero_B]",
        "Compute item counter scores per enemy hero",
        "Export to models/counter_matrix.json",
        "Verify export"
      ],
      "passes": true
    },
    {
      "id": 14,
      "category": "phase4",
      "description": "Create optimal 12-item build model",
      "steps": [
        "Create ml/train/train_full_build.py",
        "Implement constrained beam search with neural scorer",
        "Add slot balance constraints (weapon/vitality/spirit/ability)",
        "Add tier-aware transitions",
        "Export model and verify"
      ],
      "passes": true
    },
    {
      "id": 15,
      "category": "phase4",
      "description": "Add ONNX Runtime to Rust for ML inference",
      "steps": [
        "Add ort = 2.0 to Cargo.toml",
        "Create rust-server/src/ml_inference.rs",
        "Implement model loading and inference",
        "Add fallback to Markov chains if inference too slow",
        "Run cargo check"
      ],
      "passes": true
    },
    {
      "id": 16,
      "category": "phase5",
      "description": "Add navigation tabs to index.html",
      "steps": [
        "Add Match History tab with hx-get to /htmx/matches",
        "Add Leaderboards tab with hx-get to /htmx/leaderboards/overall",
        "Ensure tab switching works with HTMX",
        "Verify navigation works"
      ],
      "passes": true
    },
    {
      "id": 17,
      "category": "phase5",
      "description": "Update CSS styling",
      "steps": [
        "Add match card styles (win=green border, loss=red border)",
        "Add leaderboard table styles with rank badges",
        "Add responsive grid layouts",
        "Verify styling on different screen sizes"
      ],
      "passes": true
    },
    {
      "id": 18,
      "category": "verification",
      "description": "End-to-end verification",
      "steps": [
        "Run cargo run in rust-server",
        "Verify /htmx/all-items/6 shows real percentages",
        "Browse /htmx/matches, filter by hero, paginate",
        "Check /htmx/leaderboards/overall shows ranked players",
        "Full UI walkthrough of all tabs"
      ],
      "passes": true
    },
    {
      "id": 19,
      "category": "phase6",
      "description": "Fix All Items tab - tier classification, icons, and stats",
      "steps": [
        "Update update_icons.py to merge tier data from data/items.parquet with icon URLs from assets API",
        "Remove Slot column from all_items.html (header line 17, cell line 37, JS sorting)",
        "Fix icon URLs in handlers.rs to use https://assets.deadlock-api.com/images/items/{id}.png when icon_url is empty",
        "Verify item_stats are loaded from hero model JSON and stats columns populate",
        "Run cargo check and verify in browser"
      ],
      "passes": true
    },
    {
      "id": 20,
      "category": "phase6",
      "description": "Fix Item Synergies tab - item names and tooltips",
      "steps": [
        "Debug item name lookup in handlers.rs synergies() to identify missing item IDs",
        "Ensure precompute.py association rules only include items from metadata",
        "Add title attributes to synergy_table.html header columns (Antecedent, Consequent, Support, Confidence, Lift)",
        "Fix synergy graph node labels to show item names instead of IDs",
        "Run cargo check and verify tooltips appear on hover"
      ],
      "passes": true
    },
    {
      "id": 21,
      "category": "phase6",
      "description": "Replace Hero Stats with simple HTMX table",
      "steps": [
        "Add pick_rate field to HeroStat struct in handlers.rs",
        "Update hero_stats handler to calculate pick_rate (hero_matches / total_matches)",
        "Replace hero_stats.html Plotly chart with table: Hero | Win Rate | Total Matches | Total Wins | Pick Rate",
        "Add sortable column JavaScript similar to all_items.html",
        "Run cargo check and verify table renders"
      ],
      "passes": true
    },
    {
      "id": 22,
      "category": "phase6",
      "description": "Redesign Match History to Hero Matchups",
      "steps": [
        "Add CounterMatrix and HeroMatchup structs to models.rs",
        "Add counter_matrix loading to store.rs from models/counter_matrix.json",
        "Create hero_matchups handler in handlers.rs using counter_matrix data",
        "Redesign match_list.html as matchups table: Opponent | Games | Wins | Win Rate",
        "Add hero icons and color coding (green >52%, red <48%)",
        "Run cargo check and verify matchups display"
      ],
      "passes": true
    },
    {
      "id": 23,
      "category": "phase6",
      "description": "Redesign Build Optimizer with tree visualization",
      "steps": [
        "Add TreeData, TreeNode, TreeEdge structs to markov.rs",
        "Implement build_tree_data() function (3 levels deep, top 5 items per branch)",
        "Add tree_data handler and /htmx/tree/{hero_id} route",
        "Replace Sankey in build_path.html with scatter plot + layout.images for item icons",
        "Use horizontal layout (left to right) with edges as line traces",
        "Run cargo check and verify tree renders with icons"
      ],
      "passes": false
    },
    {
      "id": 24,
      "category": "phase6",
      "description": "End-to-end UI fixes verification",
      "steps": [
        "Run cargo run in rust-server",
        "Verify All Items tab: correct tiers, no Slot column, color icons, populated stats",
        "Verify Item Synergies: item names display, tooltips work",
        "Verify Hero Stats: table renders with sortable columns",
        "Verify Match History: hero matchups table with win rates",
        "Verify Build Optimizer: tree with item icons",
        "Full UI walkthrough of all tabs"
      ],
      "passes": false
    }
  ]
}
```
