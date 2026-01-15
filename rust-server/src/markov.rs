//! Markov chain algorithms for build path recommendations.

use crate::models::MarkovModel;
use std::collections::HashSet;

pub const START_STATE: i32 = -1;

/// Item recommendation with probability and name.
#[derive(Debug, Clone)]
pub struct ItemRecommendation {
    pub item_id: i32,
    pub prob: f64,
    pub name: String,
}

/// Get next item probabilities from current state.
pub fn get_next_probabilities(model: &MarkovModel, current_item_id: i32, top_k: usize) -> Vec<ItemRecommendation> {
    // Find current state index
    let current_idx = model
        .states
        .iter()
        .find(|s| s.item_id == current_item_id)
        .map(|s| s.idx);

    let Some(from_idx) = current_idx else {
        return vec![];
    };

    // Collect outgoing transitions
    let mut next: Vec<ItemRecommendation> = model
        .transitions
        .iter()
        .filter(|t| t.from == from_idx && t.prob > 0.0)
        .filter_map(|t| {
            let state = model.states.iter().find(|s| s.idx == t.to)?;
            if state.item_id == START_STATE {
                return None;
            }
            Some(ItemRecommendation {
                item_id: state.item_id,
                prob: t.prob,
                name: state.name.clone(),
            })
        })
        .collect();

    next.sort_by(|a, b| b.prob.partial_cmp(&a.prob).unwrap_or(std::cmp::Ordering::Equal));
    next.truncate(top_k);
    next
}

/// Find optimal build path using greedy selection.
pub fn find_optimal_path(model: &MarkovModel, length: usize) -> Vec<ItemRecommendation> {
    let mut path = Vec::with_capacity(length);
    let mut current = START_STATE;
    let mut visited = HashSet::new();

    for _ in 0..length {
        let next = get_next_probabilities(model, current, 10);

        // Find best unvisited item
        let best = next.into_iter().find(|item| !visited.contains(&item.item_id));

        if let Some(item) = best {
            visited.insert(item.item_id);
            current = item.item_id;
            path.push(item);
        } else {
            break;
        }
    }

    path
}

/// Build Sankey diagram data for visualization.
#[derive(Debug, serde::Serialize)]
pub struct SankeyData {
    pub nodes: Vec<SankeyNode>,
    pub links: Vec<SankeyLink>,
}

#[derive(Debug, serde::Serialize)]
pub struct SankeyNode {
    pub id: String,
    pub label: String,
}

#[derive(Debug, serde::Serialize)]
pub struct SankeyLink {
    pub source: String,
    pub target: String,
    pub value: f64,
}

pub fn build_sankey_data(model: &MarkovModel, top_n: usize) -> SankeyData {
    let mut nodes = vec![SankeyNode {
        id: "start".to_string(),
        label: "START".to_string(),
    }];
    let mut links = Vec::new();
    let mut node_ids: HashSet<String> = HashSet::from(["start".to_string()]);

    // Get top first items from START
    let first_items = get_next_probabilities(model, START_STATE, top_n);

    for item in &first_items {
        let node_id = format!("1_{}", item.item_id);
        if !node_ids.contains(&node_id) {
            nodes.push(SankeyNode {
                id: node_id.clone(),
                label: item.name.clone(),
            });
            node_ids.insert(node_id.clone());
        }

        links.push(SankeyLink {
            source: "start".to_string(),
            target: node_id.clone(),
            value: item.prob * 100.0,
        });

        // Get second-level items
        let second_items = get_next_probabilities(model, item.item_id, 3);
        for next in second_items {
            if next.item_id == item.item_id {
                continue;
            }

            let next_node_id = format!("2_{}", next.item_id);
            if !node_ids.contains(&next_node_id) {
                nodes.push(SankeyNode {
                    id: next_node_id.clone(),
                    label: next.name.clone(),
                });
                node_ids.insert(next_node_id.clone());
            }

            links.push(SankeyLink {
                source: node_id.clone(),
                target: next_node_id,
                value: item.prob * next.prob * 100.0,
            });
        }
    }

    SankeyData { nodes, links }
}

// =============================================================================
// Tree Visualization (replaces Sankey)
// =============================================================================

/// A node in the build tree visualization.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TreeNode {
    /// Unique node ID (e.g., "0", "1_0", "2_0_1")
    pub id: String,
    /// Item ID for icon lookup
    pub item_id: i32,
    /// Item name
    pub name: String,
    /// Probability of reaching this node from parent
    pub prob: f64,
    /// X position in the tree (horizontal: depth level)
    pub x: f64,
    /// Y position in the tree (vertical: spread within level)
    pub y: f64,
    /// Icon URL for the item
    pub icon_url: String,
}

/// An edge connecting two nodes in the tree.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TreeEdge {
    /// Source node ID
    pub source: String,
    /// Target node ID
    pub target: String,
    /// Edge weight (probability)
    pub weight: f64,
    /// Source X position
    pub x0: f64,
    /// Source Y position
    pub y0: f64,
    /// Target X position
    pub x1: f64,
    /// Target Y position
    pub y1: f64,
}

/// Full tree data for visualization.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TreeData {
    pub nodes: Vec<TreeNode>,
    pub edges: Vec<TreeEdge>,
    /// Maximum Y value for layout calculations
    pub max_y: f64,
}

/// Build tree data for visualization (3 levels deep, top N items per branch).
///
/// Layout: horizontal left-to-right tree
/// - Level 0: START (x=0)
/// - Level 1: First items (x=1)
/// - Level 2: Second items (x=2)
/// - Level 3: Third items (x=3)
pub fn build_tree_data(model: &MarkovModel, items_per_level: usize, item_icon_map: &std::collections::HashMap<i32, String>) -> TreeData {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();

    // Spacing constants
    let x_spacing = 1.0;
    let y_base_spacing = 1.0;

    // Track Y positions used at each level to avoid overlap
    let mut current_y = 0.0;

    // Level 0: START node
    let start_node = TreeNode {
        id: "start".to_string(),
        item_id: START_STATE,
        name: "START".to_string(),
        prob: 1.0,
        x: 0.0,
        y: 0.0,
        icon_url: String::new(),
    };
    nodes.push(start_node);

    // Level 1: First items from START
    let first_items = get_next_probabilities(model, START_STATE, items_per_level);
    let level1_count = first_items.len();
    let level1_start_y = -((level1_count as f64 - 1.0) / 2.0) * y_base_spacing;

    for (i, item) in first_items.iter().enumerate() {
        let node_id = format!("1_{}", i);
        let y = level1_start_y + (i as f64) * y_base_spacing;

        let icon_url = item_icon_map
            .get(&item.item_id)
            .cloned()
            .unwrap_or_else(|| format!("https://assets.deadlock-api.com/images/items/{}.png", item.item_id));

        let node = TreeNode {
            id: node_id.clone(),
            item_id: item.item_id,
            name: item.name.clone(),
            prob: item.prob,
            x: x_spacing,
            y,
            icon_url,
        };

        edges.push(TreeEdge {
            source: "start".to_string(),
            target: node_id.clone(),
            weight: item.prob,
            x0: 0.0,
            y0: 0.0,
            x1: x_spacing,
            y1: y,
        });

        nodes.push(node);

        // Level 2: Second items from each first item
        let second_items = get_next_probabilities(model, item.item_id, items_per_level);
        let level2_filtered: Vec<_> = second_items
            .iter()
            .filter(|it| it.item_id != item.item_id)
            .take(items_per_level)
            .collect();

        let level2_count = level2_filtered.len();
        if level2_count == 0 {
            continue;
        }

        let level2_spread = (level2_count as f64 - 1.0) * (y_base_spacing * 0.6);
        let level2_start_y = y - level2_spread / 2.0;

        for (j, second_item) in level2_filtered.iter().enumerate() {
            let node2_id = format!("2_{}_{}", i, j);
            let y2 = level2_start_y + (j as f64) * (y_base_spacing * 0.6);

            let icon_url2 = item_icon_map
                .get(&second_item.item_id)
                .cloned()
                .unwrap_or_else(|| format!("https://assets.deadlock-api.com/images/items/{}.png", second_item.item_id));

            let node2 = TreeNode {
                id: node2_id.clone(),
                item_id: second_item.item_id,
                name: second_item.name.clone(),
                prob: second_item.prob,
                x: x_spacing * 2.0,
                y: y2,
                icon_url: icon_url2,
            };

            edges.push(TreeEdge {
                source: node_id.clone(),
                target: node2_id.clone(),
                weight: second_item.prob,
                x0: x_spacing,
                y0: y,
                x1: x_spacing * 2.0,
                y1: y2,
            });

            nodes.push(node2);

            // Level 3: Third items from each second item
            let third_items = get_next_probabilities(model, second_item.item_id, items_per_level);
            let level3_filtered: Vec<_> = third_items
                .iter()
                .filter(|it| it.item_id != second_item.item_id && it.item_id != item.item_id)
                .take(items_per_level)
                .collect();

            let level3_count = level3_filtered.len();
            if level3_count == 0 {
                continue;
            }

            let level3_spread = (level3_count as f64 - 1.0) * (y_base_spacing * 0.4);
            let level3_start_y = y2 - level3_spread / 2.0;

            for (k, third_item) in level3_filtered.iter().enumerate() {
                let node3_id = format!("3_{}_{}_{}", i, j, k);
                let y3 = level3_start_y + (k as f64) * (y_base_spacing * 0.4);

                let icon_url3 = item_icon_map
                    .get(&third_item.item_id)
                    .cloned()
                    .unwrap_or_else(|| format!("https://assets.deadlock-api.com/images/items/{}.png", third_item.item_id));

                let node3 = TreeNode {
                    id: node3_id.clone(),
                    item_id: third_item.item_id,
                    name: third_item.name.clone(),
                    prob: third_item.prob,
                    x: x_spacing * 3.0,
                    y: y3,
                    icon_url: icon_url3,
                };

                edges.push(TreeEdge {
                    source: node2_id.clone(),
                    target: node3_id.clone(),
                    weight: third_item.prob,
                    x0: x_spacing * 2.0,
                    y0: y2,
                    x1: x_spacing * 3.0,
                    y1: y3,
                });

                nodes.push(node3);

                // Track max Y for layout
                if y3.abs() > current_y {
                    current_y = y3.abs();
                }
            }

            if y2.abs() > current_y {
                current_y = y2.abs();
            }
        }

        if y.abs() > current_y {
            current_y = y.abs();
        }
    }

    TreeData {
        nodes,
        edges,
        max_y: current_y + 1.0,
    }
}
