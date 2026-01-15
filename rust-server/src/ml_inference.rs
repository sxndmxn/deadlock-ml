//! ML inference module for build recommendations.
//!
//! This module provides inference capabilities for the ML models:
//! - Full build optimizer (synergy-based recommendations)
//! - Counter-picking (hero matchups and item counters)
//!
//! Falls back to Markov chains when ML inference is too slow or unavailable.

use crate::markov::{self, ItemRecommendation};
use crate::models::{ItemInfo, MarkovModel};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Maximum inference time before falling back to Markov chains.
const MAX_INFERENCE_TIME_MS: u64 = 50;

/// Errors that can occur during ML inference.
#[derive(Error, Debug)]
pub enum MlError {
    #[error("Model file not found: {0}")]
    ModelNotFound(String),
    #[error("Failed to parse model: {0}")]
    ParseError(String),
    #[error("Inference timeout")]
    Timeout,
    #[error("ONNX runtime error: {0}")]
    OnnxError(String),
}

/// A precomputed optimal build from the full build model.
#[derive(Debug, Clone, Deserialize)]
pub struct PrecomputedBuild {
    pub variant: u32,
    pub items: Vec<i64>,
    pub item_names: Vec<String>,
    pub score: f64,
    #[serde(default)]
    pub slot_counts: HashMap<String, u32>,
    #[serde(default)]
    pub tier_counts: HashMap<String, u32>,
    #[serde(default)]
    pub total_cost: u64,
}

/// Synergy score between two items.
#[derive(Debug, Clone, Deserialize)]
pub struct SynergyScore {
    pub item_a: i64,
    pub item_b: i64,
    pub score: f64,
}

/// Full build optimizer model.
#[derive(Debug, Clone, Deserialize)]
pub struct FullBuildModel {
    pub model_type: String,
    pub hero_id: i32,
    pub generated_at: String,
    pub config: FullBuildConfig,
    pub items: Vec<FullBuildItem>,
    #[serde(default)]
    pub synergy_scores: Vec<SynergyScore>,
    #[serde(default)]
    pub item_winrates: HashMap<String, f64>,
    #[serde(default)]
    pub precomputed_builds: Vec<PrecomputedBuild>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FullBuildConfig {
    pub max_items: u32,
    pub slots: Vec<String>,
    #[serde(default)]
    pub slot_soft_max: u32,
    #[serde(default)]
    pub slot_hard_max: u32,
    #[serde(default)]
    pub min_items_per_slot: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FullBuildItem {
    pub id: i64,
    pub name: String,
    pub tier: i32,
    pub slot: String,
    #[serde(default)]
    pub cost: u64,
}

/// Hero matchup data from counter matrix.
#[derive(Debug, Clone, Deserialize)]
pub struct HeroMatchup {
    pub wins: u32,
    pub games: u32,
    pub win_rate: f64,
}

/// Item counter effectiveness data.
#[derive(Debug, Clone, Deserialize)]
pub struct ItemCounterData {
    pub baseline_win_rate: f64,
    pub vs_heroes: HashMap<String, ItemVsHero>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ItemVsHero {
    pub wins: u32,
    pub games: u32,
    pub win_rate: f64,
    #[serde(default)]
    pub effectiveness: f64,
}

/// Counter-picking matrix model.
#[derive(Debug, Clone, Deserialize)]
pub struct CounterMatrix {
    pub model_type: String,
    pub generated_at: String,
    pub metadata: CounterMatrixMetadata,
    pub hero_matchups: HashMap<String, HashMap<String, HeroMatchup>>,
    pub item_counter_scores: HashMap<String, ItemCounterData>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CounterMatrixMetadata {
    pub num_heroes: u32,
    pub num_items: u32,
    pub hero_ids: Vec<i32>,
    pub item_ids: Vec<i64>,
    #[serde(default)]
    pub hero_names: HashMap<String, String>,
    #[serde(default)]
    pub item_names: HashMap<String, String>,
}

/// ML inference engine that provides item recommendations.
pub struct MlInference {
    /// Full build models per hero.
    full_build_models: HashMap<i32, FullBuildModel>,
    /// Counter-picking matrix (shared across heroes).
    counter_matrix: Option<CounterMatrix>,
    /// Synergy score lookup (item_a, item_b) -> score.
    synergy_lookup: HashMap<(i64, i64), f64>,
}

impl MlInference {
    /// Create a new ML inference engine by loading models from disk.
    pub fn new(models_dir: &Path) -> Self {
        let mut full_build_models = HashMap::new();
        let mut synergy_lookup = HashMap::new();

        // Load full build models
        if let Ok(entries) = fs::read_dir(models_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    // Load full_build_*.json files
                    if name.starts_with("full_build_") && name.ends_with(".json") {
                        if let Ok(content) = fs::read_to_string(&path) {
                            match serde_json::from_str::<FullBuildModel>(&content) {
                                Ok(model) => {
                                    // Build synergy lookup for this hero
                                    for synergy in &model.synergy_scores {
                                        synergy_lookup.insert(
                                            (synergy.item_a, synergy.item_b),
                                            synergy.score,
                                        );
                                        synergy_lookup.insert(
                                            (synergy.item_b, synergy.item_a),
                                            synergy.score,
                                        );
                                    }
                                    tracing::info!(
                                        "Loaded full build model for hero {}",
                                        model.hero_id
                                    );
                                    full_build_models.insert(model.hero_id, model);
                                }
                                Err(e) => {
                                    tracing::warn!("Failed to parse {}: {}", name, e);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Load counter matrix
        let counter_matrix_path = models_dir.join("counter_matrix.json");
        let counter_matrix = if counter_matrix_path.exists() {
            match fs::read_to_string(&counter_matrix_path) {
                Ok(content) => match serde_json::from_str::<CounterMatrix>(&content) {
                    Ok(matrix) => {
                        tracing::info!(
                            "Loaded counter matrix with {} heroes, {} items",
                            matrix.metadata.num_heroes,
                            matrix.metadata.num_items
                        );
                        Some(matrix)
                    }
                    Err(e) => {
                        tracing::warn!("Failed to parse counter matrix: {}", e);
                        None
                    }
                },
                Err(e) => {
                    tracing::warn!("Failed to read counter matrix: {}", e);
                    None
                }
            }
        } else {
            None
        };

        Self {
            full_build_models,
            counter_matrix,
            synergy_lookup,
        }
    }

    /// Get the best precomputed build for a hero.
    pub fn get_precomputed_build(&self, hero_id: i32) -> Option<&PrecomputedBuild> {
        self.full_build_models
            .get(&hero_id)
            .and_then(|model| model.precomputed_builds.first())
    }

    /// Get all precomputed build variants for a hero.
    pub fn get_all_precomputed_builds(&self, hero_id: i32) -> Vec<&PrecomputedBuild> {
        self.full_build_models
            .get(&hero_id)
            .map(|model| model.precomputed_builds.iter().collect())
            .unwrap_or_default()
    }

    /// Get synergy score between two items.
    pub fn get_synergy_score(&self, item_a: i64, item_b: i64) -> f64 {
        self.synergy_lookup
            .get(&(item_a, item_b))
            .copied()
            .unwrap_or(0.0)
    }

    /// Get hero matchup win rate (hero_a vs hero_b).
    pub fn get_hero_matchup(&self, hero_a: i32, hero_b: i32) -> Option<&HeroMatchup> {
        self.counter_matrix.as_ref().and_then(|matrix| {
            matrix
                .hero_matchups
                .get(&hero_a.to_string())
                .and_then(|matchups| matchups.get(&hero_b.to_string()))
        })
    }

    /// Get item effectiveness against a specific enemy hero.
    pub fn get_item_counter_effectiveness(&self, item_id: i64, enemy_hero_id: i32) -> Option<f64> {
        self.counter_matrix.as_ref().and_then(|matrix| {
            matrix
                .item_counter_scores
                .get(&item_id.to_string())
                .and_then(|data| data.vs_heroes.get(&enemy_hero_id.to_string()))
                .map(|vs| vs.effectiveness)
        })
    }

    /// Get recommended counter items against enemy heroes.
    ///
    /// Returns items sorted by their effectiveness against the enemy team.
    pub fn get_counter_items(
        &self,
        enemy_hero_ids: &[i32],
        available_items: &[&ItemInfo],
        top_k: usize,
    ) -> Vec<(i64, f64)> {
        let Some(matrix) = &self.counter_matrix else {
            return vec![];
        };

        let mut item_scores: Vec<(i64, f64)> = available_items
            .iter()
            .filter_map(|item| {
                let item_id_str = item.id.to_string();
                let counter_data = matrix.item_counter_scores.get(&item_id_str)?;

                // Sum effectiveness against all enemy heroes
                let total_effectiveness: f64 = enemy_hero_ids
                    .iter()
                    .filter_map(|hero_id| {
                        counter_data
                            .vs_heroes
                            .get(&hero_id.to_string())
                            .map(|vs| vs.effectiveness)
                    })
                    .sum();

                Some((item.id, total_effectiveness))
            })
            .collect();

        // Sort by effectiveness descending
        item_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        item_scores.truncate(top_k);
        item_scores
    }

    /// Recommend next items using ML model with Markov fallback.
    ///
    /// This method attempts ML inference first, but falls back to Markov chains
    /// if inference takes too long or encounters an error.
    pub fn recommend_next_items(
        &self,
        hero_id: i32,
        current_items: &[i64],
        markov_model: &MarkovModel,
        current_item_id: i32,
        top_k: usize,
    ) -> Vec<ItemRecommendation> {
        let start = Instant::now();

        // Try ML-based recommendation first
        if let Some(full_build) = self.full_build_models.get(&hero_id) {
            // Use synergy-based scoring
            let ml_recommendations = self.synergy_based_recommendations(
                full_build,
                current_items,
                top_k,
            );

            if start.elapsed() < Duration::from_millis(MAX_INFERENCE_TIME_MS) {
                if !ml_recommendations.is_empty() {
                    return ml_recommendations;
                }
            } else {
                tracing::debug!(
                    "ML inference timeout ({:?}), falling back to Markov",
                    start.elapsed()
                );
            }
        }

        // Fallback to Markov chains
        markov::get_next_probabilities(markov_model, current_item_id, top_k)
    }

    /// Generate recommendations based on item synergies.
    fn synergy_based_recommendations(
        &self,
        model: &FullBuildModel,
        current_items: &[i64],
        top_k: usize,
    ) -> Vec<ItemRecommendation> {
        let current_set: std::collections::HashSet<i64> = current_items.iter().copied().collect();

        let mut candidates: Vec<(i64, String, f64)> = model
            .items
            .iter()
            .filter(|item| !current_set.contains(&item.id))
            .map(|item| {
                // Calculate synergy score with current items
                let synergy_sum: f64 = current_items
                    .iter()
                    .map(|&owned| self.get_synergy_score(owned, item.id))
                    .sum();

                // Get item win rate
                let win_rate = model
                    .item_winrates
                    .get(&item.id.to_string())
                    .copied()
                    .unwrap_or(0.5);

                // Combined score: 60% synergy, 40% win rate
                let score = if current_items.is_empty() {
                    win_rate
                } else {
                    0.6 * (synergy_sum / current_items.len() as f64) + 0.4 * win_rate
                };

                (item.id, item.name.clone(), score)
            })
            .collect();

        // Sort by score descending
        candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(top_k);

        // Convert to ItemRecommendation format
        candidates
            .into_iter()
            .map(|(item_id, name, score)| ItemRecommendation {
                item_id: item_id as i32,
                prob: score,
                name,
            })
            .collect()
    }

    /// Check if ML models are available for a hero.
    pub fn has_model(&self, hero_id: i32) -> bool {
        self.full_build_models.contains_key(&hero_id)
    }

    /// Check if counter matrix is available.
    pub fn has_counter_matrix(&self) -> bool {
        self.counter_matrix.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_synergy_lookup() {
        let models_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("models");

        let inference = MlInference::new(&models_dir);

        // Test synergy lookup is symmetric
        let score_ab = inference.get_synergy_score(26002154, 465043967);
        let score_ba = inference.get_synergy_score(465043967, 26002154);
        assert_eq!(score_ab, score_ba);
    }

    #[test]
    fn test_precomputed_builds() {
        let models_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("models");

        let inference = MlInference::new(&models_dir);

        // Test getting precomputed builds
        if let Some(build) = inference.get_precomputed_build(6) {
            assert_eq!(build.items.len(), 12);
            assert!(build.score > 0.0);
        }
    }
}
