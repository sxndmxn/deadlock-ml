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
      "passes": false
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
      "passes": false
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
      "passes": false
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
      "passes": false
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
      "passes": false
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
      "passes": false
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
      "passes": false
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
      "passes": false
    }
  ]
}
```
