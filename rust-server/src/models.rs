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
