//! Model structs for JSON deserialization.

use serde::Deserialize;

/// Statistics for a single item's performance with a specific hero.
#[derive(Debug, Clone, Deserialize)]
pub struct ItemStats {
    pub item_id: i64,
    pub total_matches: u64,
    pub wins: u64,
    pub win_rate: f64,
    pub pick_rate: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HeroModel {
    pub hero_id: i32,
    pub hero_name: String,
    pub match_count: u64,
    pub win_count: u64,
    pub generated_at: String,
    pub markov: MarkovModel,
    pub association_rules: Vec<AssociationRule>,
    #[serde(default)]
    pub item_stats: Vec<ItemStats>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MarkovModel {
    pub states: Vec<MarkovState>,
    pub transitions: Vec<MarkovTransition>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MarkovState {
    pub idx: usize,
    pub item_id: i32,
    pub name: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MarkovTransition {
    pub from: usize,
    pub to: usize,
    pub prob: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AssociationRule {
    pub antecedents: Vec<i64>,
    pub consequents: Vec<i64>,
    pub support: f64,
    pub confidence: f64,
    pub lift: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Metadata {
    pub heroes: Vec<HeroInfo>,
    pub items: Vec<ItemInfo>,
    pub generated_at: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HeroInfo {
    pub id: i32,
    pub name: String,
    #[serde(default)]
    pub icon_url: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ItemInfo {
    pub id: i64,
    pub name: String,
    pub slot: String,
    pub tier: i32,
    #[serde(default)]
    pub icon_url: String,
}

impl HeroInfo {
    pub fn icon_url(&self) -> String {
        if self.icon_url.is_empty() {
            format!("https://assets.deadlock-api.com/images/heroes/{}.png", self.id)
        } else {
            self.icon_url.clone()
        }
    }
}

impl ItemInfo {
    pub fn icon_url(&self) -> String {
        if self.icon_url.is_empty() {
            format!("https://assets.deadlock-api.com/images/items/{}.png", self.id)
        } else {
            self.icon_url.clone()
        }
    }
}

// =============================================================================
// Counter Matrix (Hero Matchups)
// =============================================================================

use std::collections::HashMap;

/// Hero matchup statistics (wins, games, win_rate vs another hero)
#[derive(Debug, Clone, Deserialize)]
pub struct HeroMatchup {
    pub wins: u32,
    pub games: u32,
    pub win_rate: f64,
}

/// Metadata section of counter matrix
#[derive(Debug, Clone, Deserialize)]
pub struct CounterMatrixMetadata {
    pub num_heroes: u32,
    pub num_items: u32,
    pub hero_ids: Vec<i32>,
    pub item_ids: Vec<i64>,
}

/// Full counter matrix loaded from JSON
#[derive(Debug, Clone, Deserialize)]
pub struct CounterMatrix {
    pub model_type: String,
    pub generated_at: String,
    pub metadata: CounterMatrixMetadata,
    /// Maps hero_id (as string) -> opponent_hero_id (as string) -> HeroMatchup
    pub hero_matchups: HashMap<String, HashMap<String, HeroMatchup>>,
    /// Item counter scores (not used for hero matchups display)
    #[serde(default)]
    pub item_counter_scores: HashMap<String, serde_json::Value>,
}

impl CounterMatrix {
    /// Get matchup data for a specific hero vs opponent
    pub fn get_matchup(&self, hero_id: i32, vs_hero_id: i32) -> Option<&HeroMatchup> {
        self.hero_matchups
            .get(&hero_id.to_string())
            .and_then(|m| m.get(&vs_hero_id.to_string()))
    }

    /// Get all matchups for a hero, sorted by win rate descending
    pub fn get_all_matchups(&self, hero_id: i32) -> Vec<(i32, HeroMatchup)> {
        let Some(matchups) = self.hero_matchups.get(&hero_id.to_string()) else {
            return Vec::new();
        };

        let mut result: Vec<(i32, HeroMatchup)> = matchups
            .iter()
            .filter_map(|(vs_id_str, matchup)| {
                let vs_id = vs_id_str.parse::<i32>().ok()?;
                // Exclude self-matchup (should have 0 games anyway)
                if vs_id == hero_id || matchup.games == 0 {
                    return None;
                }
                Some((vs_id, matchup.clone()))
            })
            .collect();

        // Sort by win rate descending
        result.sort_by(|a, b| b.1.win_rate.partial_cmp(&a.1.win_rate).unwrap_or(std::cmp::Ordering::Equal));
        result
    }
}
