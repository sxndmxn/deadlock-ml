//! Route handlers for all endpoints.

use crate::{
    db::{LeaderboardQuery, LeaderboardSort, MatchListQuery as DbMatchListQuery},
    markov::{build_sankey_data, find_optimal_path, get_next_probabilities, ItemRecommendation, SankeyData},
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

    let metadata = state.store.get_metadata();
    let items = metadata.map(|m| m.items).unwrap_or_default();
    let item_names: std::collections::HashMap<i64, String> = items.iter().map(|i| (i.id, i.name.clone())).collect();

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
    let mut filter_item_ids: std::collections::HashSet<i64> = std::collections::HashSet::new();
    for rule in &model.association_rules {
        filter_item_ids.extend(&rule.antecedents);
        filter_item_ids.extend(&rule.consequents);
    }
    let mut filter_items: Vec<ItemInfo> = items
        .into_iter()
        .filter(|i| filter_item_ids.contains(&i.id))
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

    let metadata = state.store.get_metadata();
    let items = metadata.map(|m| m.items).unwrap_or_default();
    let item_names: std::collections::HashMap<i64, String> = items.iter().map(|i| (i.id, i.name.clone())).collect();

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

    let mut heroes: Vec<HeroStat> = hero_info
        .iter()
        .filter_map(|info| {
            let model = state.store.get_hero(info.id)?;
            let win_rate = if model.match_count > 0 {
                model.win_count as f64 / model.match_count as f64
            } else {
                0.0
            };
            Some(HeroStat {
                id: info.id,
                name: info.name.clone(),
                match_count: model.match_count,
                win_count: model.win_count,
                win_rate,
            })
        })
        .collect();

    heroes.sort_by(|a, b| a.win_rate.partial_cmp(&b.win_rate).unwrap_or(std::cmp::Ordering::Equal));

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
            DisplayItem {
                id: item.id,
                name: item.name,
                tier: item.tier,
                slot: item.slot,
                icon_url: item.icon_url,
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
// Match History
// =============================================================================

/// Template for rendering match list.
#[derive(Template)]
#[template(path = "partials/match_list.html")]
pub struct MatchListTemplate {
    pub matches: Vec<DisplayMatch>,
    pub heroes: Vec<HeroInfo>,
    pub total_matches: u64,
    pub current_page: u32,
    pub total_pages: u32,
    pub selected_hero_id: i32,  // 0 means no selection
    pub selected_outcome: String,  // "", "win", or "loss"
}

/// Display-ready match summary for templates.
pub struct DisplayMatch {
    pub match_id: i64,
    pub hero_id: i32,
    pub hero_name: String,
    pub account_id: i64,
    pub won: bool,
    pub kills: i32,
    pub deaths: i32,
    pub assists: i32,
    pub net_worth: i32,
    pub kda: String,
}

const MATCHES_PER_PAGE: u32 = 20;

/// Query parameters for match list filtering.
#[derive(Deserialize)]
pub struct MatchListQuery {
    pub hero_id: Option<i32>,
    pub account_id: Option<i64>,
    pub outcome: Option<String>,
    pub page: Option<u32>,
}

/// Handler for match list with filtering and pagination.
pub async fn match_list(
    Query(params): Query<MatchListQuery>,
    State(state): State<AppState>,
) -> impl IntoResponse {
    let page = params.page.unwrap_or(1).max(1);
    let offset = (page - 1) * MATCHES_PER_PAGE;

    // Convert outcome string to boolean
    let won = match params.outcome.as_deref() {
        Some("win") => Some(true),
        Some("loss") => Some(false),
        _ => None,
    };

    // Build database query
    let db_query = DbMatchListQuery {
        hero_id: params.hero_id,
        account_id: params.account_id,
        won,
        limit: MATCHES_PER_PAGE,
        offset,
    };

    // Query database
    let matches_result = state.db.get_match_list(&db_query);
    let count_result = state.db.get_match_count(&db_query);

    // Get hero info for name lookup
    let metadata = state.store.get_metadata();
    let heroes = metadata.as_ref().map(|m| m.heroes.clone()).unwrap_or_default();
    let hero_names: std::collections::HashMap<i32, String> =
        heroes.iter().map(|h| (h.id, h.name.clone())).collect();

    // Handle database errors gracefully
    let (matches, total_matches) = match (matches_result, count_result) {
        (Ok(m), Ok(c)) => (m, c),
        _ => (Vec::new(), 0),
    };

    // Convert to display format
    let display_matches: Vec<DisplayMatch> = matches
        .into_iter()
        .map(|m| {
            let kda = if m.deaths == 0 {
                format!("{}/{}/{} (Perfect)", m.kills, m.deaths, m.assists)
            } else {
                let kda_ratio = (m.kills as f64 + m.assists as f64) / m.deaths as f64;
                format!("{}/{}/{} ({:.2})", m.kills, m.deaths, m.assists, kda_ratio)
            };
            DisplayMatch {
                match_id: m.match_id,
                hero_id: m.hero_id,
                hero_name: hero_names.get(&m.hero_id).cloned().unwrap_or_else(|| format!("Hero {}", m.hero_id)),
                account_id: m.account_id,
                won: m.won,
                kills: m.kills,
                deaths: m.deaths,
                assists: m.assists,
                net_worth: m.net_worth,
                kda,
            }
        })
        .collect();

    let total_pages = ((total_matches as f64) / (MATCHES_PER_PAGE as f64)).ceil() as u32;

    Html(
        MatchListTemplate {
            matches: display_matches,
            heroes,
            total_matches,
            current_page: page,
            total_pages,
            selected_hero_id: params.hero_id.unwrap_or(0),
            selected_outcome: params.outcome.unwrap_or_default(),
        }
        .render()
        .unwrap_or_else(|e| format!("<p>Error rendering template: {}</p>", e)),
    )
}

/// Handler for individual match details.
pub async fn match_detail(
    Path(_match_id): Path<i64>,
    State(_state): State<AppState>,
) -> impl IntoResponse {
    // TODO: Implement full match detail view
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
