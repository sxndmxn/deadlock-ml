//! Route handlers for all endpoints.

use crate::{
    db::{LeaderboardQuery, LeaderboardSort},
    markov::{build_sankey_data, build_tree_data, find_optimal_path, get_next_probabilities, ItemRecommendation, SankeyData, TreeData},
    models::{AssociationRule, HeroInfo, ItemInfo},
    AppState,
};
use askama::Template;
use axum::{
    extract::{Path, Query, State},
    response::{Html, IntoResponse},
    Json,
};
use serde::Deserialize;

// =============================================================================
// Templates
// =============================================================================

#[derive(Template)]
#[template(path = "index.html")]
pub struct IndexTemplate {
    pub heroes: Vec<HeroInfo>,
    pub selected_hero_id: i32,
}

#[derive(Template)]
#[template(path = "partials/build_path.html")]
pub struct BuildPathTemplate {
    pub hero_id: i32,
    pub hero_name: String,
    pub match_count: u64,
    pub win_count: u64,
    pub build_path: Vec<ItemRecommendation>,
    pub items: Vec<ItemInfo>,
}

#[derive(Template)]
#[template(path = "partials/next_items.html")]
pub struct NextItemsTemplate {
    pub next_items: Vec<ItemRecommendation>,
}

#[derive(Template)]
#[template(path = "partials/synergy_table.html")]
pub struct SynergyTableTemplate {
    pub hero_id: i32,
    pub rules: Vec<DisplayRule>,
    pub filter_items: Vec<ItemInfo>,
    pub total_rules: usize,
    pub avg_confidence: f64,
    pub avg_lift: f64,
}

#[derive(Clone)]
pub struct DisplayRule {
    pub antecedent_names: String,
    pub consequent_names: String,
    pub support: f64,
    pub confidence: f64,
    pub lift: f64,
}

#[derive(Template)]
#[template(path = "partials/hero_stats.html")]
pub struct HeroStatsTemplate {
    pub heroes: Vec<HeroStat>,
    pub selected_hero_id: i32,
}

pub struct HeroStat {
    pub id: i32,
    pub name: String,
    pub match_count: u64,
    pub win_count: u64,
    pub win_rate: f64,
    pub pick_rate: f64,
}

// =============================================================================
// Index
// =============================================================================

pub async fn index(State(state): State<AppState>) -> impl IntoResponse {
    let metadata = state.store.get_metadata();
    let heroes = metadata.map(|m| m.heroes).unwrap_or_default();
    let selected_hero_id = heroes.first().map(|h| h.id).unwrap_or(1);

    Html(
        IndexTemplate {
            heroes,
            selected_hero_id,
        }
        .render()
        .unwrap_or_default(),
    )
}

// =============================================================================
// Build Optimizer
// =============================================================================

pub async fn build_path(Path(hero_id): Path<i32>, State(state): State<AppState>) -> impl IntoResponse {
    let Some(model) = state.store.get_hero(hero_id) else {
        return Html("<p>Model not found</p>".to_string());
    };

    let metadata = state.store.get_metadata();
    let items = metadata.map(|m| m.items).unwrap_or_default();

    let build_path = find_optimal_path(&model.markov, 6);

    Html(
        BuildPathTemplate {
            hero_id,
            hero_name: model.hero_name.clone(),
            match_count: model.match_count,
            win_count: model.win_count,
            build_path,
            items,
        }
        .render()
        .unwrap_or_default(),
    )
}

pub async fn next_items(
    Path((hero_id, item_id)): Path<(i32, i32)>,
    State(state): State<AppState>,
) -> impl IntoResponse {
    let Some(model) = state.store.get_hero(hero_id) else {
        return Html("<p>Model not found</p>".to_string());
    };

    let next_items = get_next_probabilities(&model.markov, item_id, 5);

    Html(NextItemsTemplate { next_items }.render().unwrap_or_default())
}

pub async fn sankey_data(Path(hero_id): Path<i32>, State(state): State<AppState>) -> impl IntoResponse {
    let Some(model) = state.store.get_hero(hero_id) else {
        return Json(SankeyData {
            nodes: vec![],
            links: vec![],
        });
    };

    Json(build_sankey_data(&model.markov, 5))
}

/// Returns tree visualization data for build path.
pub async fn tree_data(Path(hero_id): Path<i32>, State(state): State<AppState>) -> impl IntoResponse {
    let Some(model) = state.store.get_hero(hero_id) else {
        return Json(TreeData {
            nodes: vec![],
            edges: vec![],
            max_y: 0.0,
        });
    };

    // Build item icon map from markov states
    // The model's markov states use internal IDs (100, 101, etc.)
    // We need to map these to icon URLs
    let item_icon_map: std::collections::HashMap<i32, String> = model
        .markov
        .states
        .iter()
        .filter(|s| s.item_id >= 0)
        .map(|s| {
            (
                s.item_id,
                format!("https://assets.deadlock-api.com/images/items/{}.png", s.item_id),
            )
        })
        .collect();

    // Build tree with 5 items per level, 3 levels deep
    Json(build_tree_data(&model.markov, 5, &item_icon_map))
}

// =============================================================================
// Synergies
// =============================================================================

#[derive(Deserialize)]
pub struct SynergyFilter {
    pub item_id: Option<i64>,
}

pub async fn synergies(
    Path(hero_id): Path<i32>,
    Query(filter): Query<SynergyFilter>,
    State(state): State<AppState>,
) -> impl IntoResponse {
    let Some(model) = state.store.get_hero(hero_id) else {
        return Html("<p>Model not found</p>".to_string());
    };

    // Build item name map from markov states (embedded in model) for correct ID matching
    let item_names: std::collections::HashMap<i64, String> = model
        .markov
        .states
        .iter()
        .filter(|s| s.item_id >= 0) // Skip START state (-1)
        .map(|s| (s.item_id as i64, s.name.clone()))
        .collect();

    // Filter rules if item_id specified
    let filtered_rules: Vec<&AssociationRule> = if let Some(filter_id) = filter.item_id {
        model
            .association_rules
            .iter()
            .filter(|r| r.antecedents.contains(&filter_id) || r.consequents.contains(&filter_id))
            .collect()
    } else {
        model.association_rules.iter().collect()
    };

    // Convert to display format
    let rules: Vec<DisplayRule> = filtered_rules
        .iter()
        .take(20)
        .map(|r| DisplayRule {
            antecedent_names: r
                .antecedents
                .iter()
                .map(|id| item_names.get(id).cloned().unwrap_or_else(|| format!("Item {id}")))
                .collect::<Vec<_>>()
                .join(", "),
            consequent_names: r
                .consequents
                .iter()
                .map(|id| item_names.get(id).cloned().unwrap_or_else(|| format!("Item {id}")))
                .collect::<Vec<_>>()
                .join(", "),
            support: r.support,
            confidence: r.confidence,
            lift: r.lift,
        })
        .collect();

    // Get items that appear in rules for filter dropdown
    // Build from markov states since those have the correct IDs that match association rules
    let mut filter_item_ids: std::collections::HashSet<i64> = std::collections::HashSet::new();
    for rule in &model.association_rules {
        filter_item_ids.extend(&rule.antecedents);
        filter_item_ids.extend(&rule.consequents);
    }
    let mut filter_items: Vec<ItemInfo> = model
        .markov
        .states
        .iter()
        .filter(|s| s.item_id >= 0 && filter_item_ids.contains(&(s.item_id as i64)))
        .map(|s| ItemInfo {
            id: s.item_id as i64,
            name: s.name.clone(),
            slot: String::new(),
            tier: 0,
            icon_url: String::new(),
        })
        .collect();
    filter_items.sort_by(|a, b| a.name.cmp(&b.name));

    let total_rules = filtered_rules.len();
    let avg_confidence = if total_rules > 0 {
        filtered_rules.iter().map(|r| r.confidence).sum::<f64>() / total_rules as f64
    } else {
        0.0
    };
    let avg_lift = if total_rules > 0 {
        filtered_rules.iter().map(|r| r.lift).sum::<f64>() / total_rules as f64
    } else {
        0.0
    };

    Html(
        SynergyTableTemplate {
            hero_id,
            rules,
            filter_items,
            total_rules,
            avg_confidence,
            avg_lift,
        }
        .render()
        .unwrap_or_default(),
    )
}

/// Returns JSON data for the synergy network graph.
#[derive(serde::Serialize)]
pub struct GraphData {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

#[derive(serde::Serialize)]
pub struct GraphNode {
    pub id: String,
    pub label: String,
}

#[derive(serde::Serialize)]
pub struct GraphEdge {
    pub from: String,
    pub to: String,
    pub width: f64,
}

pub async fn synergy_graph(Path(hero_id): Path<i32>, State(state): State<AppState>) -> impl IntoResponse {
    let Some(model) = state.store.get_hero(hero_id) else {
        return Json(GraphData {
            nodes: vec![],
            edges: vec![],
        });
    };

    // Build item name map from markov states (embedded in model) for correct ID matching
    let item_names: std::collections::HashMap<i64, String> = model
        .markov
        .states
        .iter()
        .filter(|s| s.item_id >= 0) // Skip START state (-1)
        .map(|s| (s.item_id as i64, s.name.clone()))
        .collect();

    // Sort rules by lift and take top 50
    let mut rules: Vec<_> = model.association_rules.iter().collect();
    rules.sort_by(|a, b| b.lift.partial_cmp(&a.lift).unwrap_or(std::cmp::Ordering::Equal));
    rules.truncate(50);

    let mut node_ids: std::collections::HashSet<i64> = std::collections::HashSet::new();
    let mut edges = Vec::new();

    for rule in &rules {
        for ant in &rule.antecedents {
            for cons in &rule.consequents {
                node_ids.insert(*ant);
                node_ids.insert(*cons);
                edges.push(GraphEdge {
                    from: ant.to_string(),
                    to: cons.to_string(),
                    width: (rule.confidence * 5.0).max(1.0),
                });
            }
        }
    }

    let nodes: Vec<GraphNode> = node_ids
        .into_iter()
        .map(|id| GraphNode {
            id: id.to_string(),
            label: item_names.get(&id).cloned().unwrap_or_else(|| format!("Item {id}")),
        })
        .collect();

    Json(GraphData { nodes, edges })
}

// =============================================================================
// Hero Stats
// =============================================================================

pub async fn hero_stats(
    Query(params): Query<HeroStatsQuery>,
    State(state): State<AppState>,
) -> impl IntoResponse {
    let metadata = state.store.get_metadata();
    let hero_info = metadata.map(|m| m.heroes).unwrap_or_default();

    // First pass: collect hero data and compute total matches
    let hero_data: Vec<_> = hero_info
        .iter()
        .filter_map(|info| {
            let model = state.store.get_hero(info.id)?;
            Some((info.id, info.name.clone(), model.match_count, model.win_count))
        })
        .collect();

    let total_matches: u64 = hero_data.iter().map(|(_, _, matches, _)| *matches).sum();

    // Second pass: build HeroStat with pick_rate
    let mut heroes: Vec<HeroStat> = hero_data
        .into_iter()
        .map(|(id, name, match_count, win_count)| {
            let win_rate = if match_count > 0 {
                win_count as f64 / match_count as f64
            } else {
                0.0
            };
            let pick_rate = if total_matches > 0 {
                match_count as f64 / total_matches as f64
            } else {
                0.0
            };
            HeroStat {
                id,
                name,
                match_count,
                win_count,
                win_rate,
                pick_rate,
            }
        })
        .collect();

    heroes.sort_by(|a, b| b.win_rate.partial_cmp(&a.win_rate).unwrap_or(std::cmp::Ordering::Equal));

    let selected_hero_id = params.hero_id.unwrap_or_else(|| heroes.first().map(|h| h.id).unwrap_or(1));

    Html(
        HeroStatsTemplate {
            heroes,
            selected_hero_id,
        }
        .render()
        .unwrap_or_default(),
    )
}

#[derive(Deserialize)]
pub struct HeroStatsQuery {
    pub hero_id: Option<i32>,
}

// =============================================================================
// All Items
// =============================================================================

#[derive(Template)]
#[template(path = "partials/all_items.html")]
pub struct AllItemsTemplate {
    pub items: Vec<DisplayItem>,
}

pub struct DisplayItem {
    pub id: i64,
    pub name: String,
    pub tier: i32,
    pub slot: String,
    pub icon_url: String,
    pub win_rate: f64,
    pub pick_rate: f64,
    pub match_count: u64,
}

pub async fn all_items(
    Path(hero_id): Path<i32>,
    State(state): State<AppState>,
) -> impl IntoResponse {
    let metadata = state.store.get_metadata();
    let raw_items = metadata.map(|m| m.items).unwrap_or_default();

    // Get hero model to access real item stats
    let hero_model = state.store.get_hero(hero_id);

    // Build a lookup map from item_id to ItemStats
    let item_stats_map: std::collections::HashMap<i64, &crate::models::ItemStats> = hero_model
        .as_ref()
        .map(|m| m.item_stats.iter().map(|s| (s.item_id, s)).collect())
        .unwrap_or_default();

    let mut items: Vec<DisplayItem> = raw_items
        .into_iter()
        .filter(|i| i.tier > 0) // Filter out ability items with no tier
        .map(|item| {
            // Look up real stats, fall back to zeros if not found
            let stats = item_stats_map.get(&item.id);
            // Use CDN fallback for empty icon URLs
            let icon_url = if item.icon_url.is_empty() {
                format!("https://assets.deadlock-api.com/images/items/{}.png", item.id)
            } else {
                item.icon_url
            };
            DisplayItem {
                id: item.id,
                name: item.name,
                tier: item.tier,
                slot: item.slot,
                icon_url,
                win_rate: stats.map(|s| s.win_rate).unwrap_or(0.0),
                pick_rate: stats.map(|s| s.pick_rate).unwrap_or(0.0),
                match_count: stats.map(|s| s.total_matches).unwrap_or(0),
            }
        })
        .collect();

    // Sort by win rate descending by default
    items.sort_by(|a, b| b.win_rate.partial_cmp(&a.win_rate).unwrap_or(std::cmp::Ordering::Equal));

    Html(AllItemsTemplate { items }.render().unwrap_or_default())
}

// =============================================================================
// Hero Matchups (replaces Match History)
// =============================================================================

/// Template for rendering hero matchups.
#[derive(Template)]
#[template(path = "partials/match_list.html")]
pub struct HeroMatchupsTemplate {
    pub matchups: Vec<DisplayMatchup>,
    pub heroes: Vec<HeroInfo>,
    pub selected_hero_id: i32,
    pub selected_hero_name: String,
    pub total_games: u32,
}

/// Display-ready matchup data for templates.
pub struct DisplayMatchup {
    pub opponent_id: i32,
    pub opponent_name: String,
    pub opponent_icon_url: String,
    pub games: u32,
    pub wins: u32,
    pub win_rate: f64,
    pub win_rate_class: String,  // "matchup-good", "matchup-bad", or ""
}

/// Query parameters for hero matchups.
#[derive(Deserialize)]
pub struct MatchupsQuery {
    pub hero_id: Option<i32>,
}

/// Handler for hero matchups showing win rates vs each opponent.
pub async fn match_list(
    Query(params): Query<MatchupsQuery>,
    State(state): State<AppState>,
) -> impl IntoResponse {
    // Get hero list
    let metadata = state.store.get_metadata();
    let heroes = metadata.as_ref().map(|m| m.heroes.clone()).unwrap_or_default();

    // Default to first hero if none selected
    let selected_hero_id = params.hero_id.unwrap_or_else(|| heroes.first().map(|h| h.id).unwrap_or(1));

    // Build hero name and icon lookup
    let hero_info: std::collections::HashMap<i32, &HeroInfo> =
        heroes.iter().map(|h| (h.id, h)).collect();

    let selected_hero_name = hero_info
        .get(&selected_hero_id)
        .map(|h| h.name.clone())
        .unwrap_or_else(|| format!("Hero {}", selected_hero_id));

    // Get counter matrix
    let counter_matrix = state.store.get_counter_matrix();

    let (matchups, total_games) = if let Some(ref matrix) = counter_matrix {
        let raw_matchups = matrix.get_all_matchups(selected_hero_id);
        let total: u32 = raw_matchups.iter().map(|(_, m)| m.games).sum();

        let display_matchups: Vec<DisplayMatchup> = raw_matchups
            .into_iter()
            .map(|(opponent_id, matchup)| {
                let (opponent_name, opponent_icon_url) = hero_info
                    .get(&opponent_id)
                    .map(|h| (h.name.clone(), h.icon_url()))
                    .unwrap_or_else(|| {
                        (
                            format!("Hero {}", opponent_id),
                            format!("https://assets.deadlock-api.com/images/heroes/{}.png", opponent_id),
                        )
                    });

                // Color code based on win rate: green >52%, red <48%
                let win_rate_class = if matchup.win_rate > 0.52 {
                    "matchup-good".to_string()
                } else if matchup.win_rate < 0.48 {
                    "matchup-bad".to_string()
                } else {
                    "".to_string()
                };

                DisplayMatchup {
                    opponent_id,
                    opponent_name,
                    opponent_icon_url,
                    games: matchup.games,
                    wins: matchup.wins,
                    win_rate: matchup.win_rate,
                    win_rate_class,
                }
            })
            .collect();

        (display_matchups, total)
    } else {
        (Vec::new(), 0)
    };

    Html(
        HeroMatchupsTemplate {
            matchups,
            heroes,
            selected_hero_id,
            selected_hero_name,
            total_games,
        }
        .render()
        .unwrap_or_else(|e| format!("<p>Error rendering template: {}</p>", e)),
    )
}

/// Handler for individual match details (placeholder).
pub async fn match_detail(
    Path(_match_id): Path<i64>,
    State(_state): State<AppState>,
) -> impl IntoResponse {
    Html("<div class=\"match-detail\"><p>Match details coming soon...</p></div>".to_string())
}

// =============================================================================
// Leaderboard Handlers
// =============================================================================

/// Query parameters for leaderboard requests.
#[derive(Debug, Deserialize, Default)]
pub struct LeaderboardParams {
    pub sort: Option<String>,
    pub min_matches: Option<u32>,
    pub page: Option<u32>,
}

/// Display struct for a ranked player.
#[derive(Debug)]
pub struct DisplayRanking {
    pub rank: u32,
    pub account_id: i64,
    pub matches: u32,
    pub wins: u32,
    pub win_rate: String,
    pub total_kills: u32,
    pub total_deaths: u32,
    pub total_assists: u32,
    pub kda: String,
}

/// Template for leaderboard display.
#[derive(Template)]
#[template(path = "partials/leaderboard.html")]
pub struct LeaderboardTemplate {
    pub rankings: Vec<DisplayRanking>,
    pub heroes: Vec<HeroInfo>,
    pub total_players: u64,
    pub current_page: u32,
    pub total_pages: u32,
    pub selected_hero_id: i32,
    pub selected_sort: String,
    pub min_matches: u32,
}

const PLAYERS_PER_PAGE: u32 = 50;

/// Handler for overall leaderboard.
pub async fn leaderboard_overall(
    Query(params): Query<LeaderboardParams>,
    State(state): State<AppState>,
) -> impl IntoResponse {
    let page = params.page.unwrap_or(1).max(1);
    let min_matches = params.min_matches.unwrap_or(10);
    let sort_str = params.sort.clone().unwrap_or_else(|| "win_rate".to_string());
    let sort = LeaderboardSort::from_str(&sort_str);

    let db_query = LeaderboardQuery {
        sort_by: sort,
        min_matches,
        limit: PLAYERS_PER_PAGE,
        offset: (page - 1) * PLAYERS_PER_PAGE,
    };

    // Get heroes for dropdown
    let heroes = state
        .store
        .get_metadata()
        .map(|m| m.heroes.clone())
        .unwrap_or_default();

    // Query database
    let (rankings, total_players) = match (
        state.db.get_overall_leaderboard(&db_query),
        state.db.get_leaderboard_count(None, min_matches),
    ) {
        (Ok(r), Ok(c)) => (r, c),
        _ => (Vec::new(), 0),
    };

    // Convert to display format
    let start_rank = (page - 1) * PLAYERS_PER_PAGE;
    let display_rankings: Vec<DisplayRanking> = rankings
        .into_iter()
        .enumerate()
        .map(|(i, r)| DisplayRanking {
            rank: start_rank + (i as u32) + 1,
            account_id: r.account_id,
            matches: r.matches,
            wins: r.wins,
            win_rate: format!("{:.1}%", r.win_rate * 100.0),
            total_kills: r.total_kills,
            total_deaths: r.total_deaths,
            total_assists: r.total_assists,
            kda: format!("{:.2}", r.kda),
        })
        .collect();

    let total_pages = ((total_players as f64) / (PLAYERS_PER_PAGE as f64)).ceil() as u32;

    Html(
        LeaderboardTemplate {
            rankings: display_rankings,
            heroes,
            total_players,
            current_page: page,
            total_pages,
            selected_hero_id: 0,
            selected_sort: sort_str,
            min_matches,
        }
        .render()
        .unwrap_or_else(|e| format!("<p>Error rendering template: {}</p>", e)),
    )
}

/// Handler for hero-specific leaderboard.
pub async fn leaderboard_hero(
    Path(hero_id): Path<i32>,
    Query(params): Query<LeaderboardParams>,
    State(state): State<AppState>,
) -> impl IntoResponse {
    let page = params.page.unwrap_or(1).max(1);
    let min_matches = params.min_matches.unwrap_or(5);
    let sort_str = params.sort.clone().unwrap_or_else(|| "win_rate".to_string());
    let sort = LeaderboardSort::from_str(&sort_str);

    let db_query = LeaderboardQuery {
        sort_by: sort,
        min_matches,
        limit: PLAYERS_PER_PAGE,
        offset: (page - 1) * PLAYERS_PER_PAGE,
    };

    // Get heroes for dropdown
    let heroes = state
        .store
        .get_metadata()
        .map(|m| m.heroes.clone())
        .unwrap_or_default();

    // Query database
    let (rankings, total_players) = match (
        state.db.get_hero_leaderboard(hero_id, &db_query),
        state.db.get_leaderboard_count(Some(hero_id), min_matches),
    ) {
        (Ok(r), Ok(c)) => (r, c),
        _ => (Vec::new(), 0),
    };

    // Convert to display format
    let start_rank = (page - 1) * PLAYERS_PER_PAGE;
    let display_rankings: Vec<DisplayRanking> = rankings
        .into_iter()
        .enumerate()
        .map(|(i, r)| DisplayRanking {
            rank: start_rank + (i as u32) + 1,
            account_id: r.account_id,
            matches: r.matches,
            wins: r.wins,
            win_rate: format!("{:.1}%", r.win_rate * 100.0),
            total_kills: r.total_kills,
            total_deaths: r.total_deaths,
            total_assists: r.total_assists,
            kda: format!("{:.2}", r.kda),
        })
        .collect();

    let total_pages = ((total_players as f64) / (PLAYERS_PER_PAGE as f64)).ceil() as u32;

    Html(
        LeaderboardTemplate {
            rankings: display_rankings,
            heroes,
            total_players,
            current_page: page,
            total_pages,
            selected_hero_id: hero_id,
            selected_sort: sort_str,
            min_matches,
        }
        .render()
        .unwrap_or_else(|e| format!("<p>Error rendering template: {}</p>", e)),
    )
}
