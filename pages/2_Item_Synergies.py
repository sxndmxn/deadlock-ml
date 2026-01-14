"""
Item Synergies Page - Association rules visualization using streamlit-agraph.
"""

import streamlit as st
import pandas as pd
from streamlit_agraph import agraph, Node, Edge, Config

from ml.precompute import load_model
from lib.items import get_item_names_map

st.set_page_config(
    page_title="Item Synergies - Deadlock",
    page_icon="🔗",
    layout="wide",
)

st.title("Item Synergies")

# Check for selected hero
if "selected_hero_id" not in st.session_state or st.session_state.selected_hero_id is None:
    st.warning("Please select a hero from the sidebar on the main page first.")
    st.stop()

hero_id = st.session_state.selected_hero_id
hero_name = st.session_state.selected_hero_name

st.subheader(f"Item Synergies for {hero_name}")

# Load precomputed model
model = load_model(hero_id)

if model is None:
    st.error(f"Insufficient data for {hero_name}. No model available.")
    st.stop()

rules_df = model["association_rules"]

if rules_df.empty:
    st.warning("No association rules found for this hero.")
    st.info("This may indicate insufficient match data or no strong item synergies detected.")
    st.stop()

# Get item name mapping
item_names = get_item_names_map()


def frozenset_to_names(fs: frozenset) -> str:
    """Convert frozenset of item IDs to comma-separated names."""
    names = [item_names.get(item_id, f"Item {item_id}") for item_id in fs]
    return ", ".join(sorted(names))


# Prepare rules display dataframe
display_df = rules_df.copy()
display_df["Antecedent"] = display_df["antecedents"].apply(frozenset_to_names)
display_df["Consequent"] = display_df["consequents"].apply(frozenset_to_names)
display_df["Support"] = display_df["support"].apply(lambda x: f"{x*100:.1f}%")
display_df["Confidence"] = display_df["confidence"].apply(lambda x: f"{x*100:.1f}%")
display_df["Lift"] = display_df["lift"].apply(lambda x: f"{x:.2f}")

# Filter controls
st.sidebar.header("Filters")

# Get all unique items in rules
all_items_in_rules = set()
for _, row in rules_df.iterrows():
    all_items_in_rules.update(row["antecedents"])
    all_items_in_rules.update(row["consequents"])

item_filter_options = ["Show all"] + sorted(
    [item_names.get(item_id, f"Item {item_id}") for item_id in all_items_in_rules]
)

selected_filter = st.sidebar.selectbox(
    "Filter by item:",
    options=item_filter_options,
)

# Apply filter
if selected_filter != "Show all":
    # Find item ID from name
    filter_item_id = None
    for item_id, name in item_names.items():
        if name == selected_filter:
            filter_item_id = item_id
            break

    if filter_item_id is not None:
        mask = rules_df.apply(
            lambda row: filter_item_id in row["antecedents"] or filter_item_id in row["consequents"],
            axis=1,
        )
        filtered_rules = rules_df[mask]
        filtered_display = display_df[mask]
    else:
        filtered_rules = rules_df
        filtered_display = display_df
else:
    filtered_rules = rules_df
    filtered_display = display_df

# Network graph
st.markdown("### Item Relationship Graph")
st.caption("Nodes are items. Edges show association strength (thicker = higher confidence).")


def build_network_graph(rules: pd.DataFrame, max_edges: int = 50):
    """Build network graph from association rules."""
    if rules.empty:
        return [], []

    # Collect nodes and edges
    node_ids = set()
    edges_data = []

    # Sort by lift to get most interesting relationships
    sorted_rules = rules.sort_values("lift", ascending=False).head(max_edges)

    for _, row in sorted_rules.iterrows():
        antecedents = row["antecedents"]
        consequents = row["consequents"]
        confidence = row["confidence"]
        lift = row["lift"]

        # Add edges between all antecedent/consequent pairs
        for ant_id in antecedents:
            for cons_id in consequents:
                node_ids.add(ant_id)
                node_ids.add(cons_id)
                edges_data.append({
                    "source": ant_id,
                    "target": cons_id,
                    "confidence": confidence,
                    "lift": lift,
                })

    # Create nodes
    nodes = []
    for item_id in node_ids:
        item_name = item_names.get(item_id, f"Item {item_id}")
        nodes.append(
            Node(
                id=str(item_id),
                label=item_name,
                size=20,
                color="#97C2FC",
            )
        )

    # Create edges with thickness based on confidence
    edges = []
    for edge in edges_data:
        width = max(1, edge["confidence"] * 5)  # Scale confidence to edge width
        edges.append(
            Edge(
                source=str(edge["source"]),
                target=str(edge["target"]),
                width=width,
                title=f"Confidence: {edge['confidence']:.2f}, Lift: {edge['lift']:.2f}",
            )
        )

    return nodes, edges


nodes, edges = build_network_graph(filtered_rules)

if nodes and edges:
    config = Config(
        width=800,
        height=500,
        directed=True,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=False,
    )

    agraph(nodes=nodes, edges=edges, config=config)
else:
    st.info("No graph data available with current filters.")

# Rules table
st.markdown("---")
st.markdown("### Association Rules Table")
st.caption("Top rules sorted by lift. Higher lift indicates stronger association.")

# Display top 20 rules
table_display = filtered_display[["Antecedent", "Consequent", "Support", "Confidence", "Lift"]].head(20)

st.dataframe(
    table_display,
    use_container_width=True,
    hide_index=True,
)

# Summary stats
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Rules", len(filtered_rules))

with col2:
    avg_confidence = filtered_rules["confidence"].mean() * 100 if not filtered_rules.empty else 0
    st.metric("Avg Confidence", f"{avg_confidence:.1f}%")

with col3:
    avg_lift = filtered_rules["lift"].mean() if not filtered_rules.empty else 0
    st.metric("Avg Lift", f"{avg_lift:.2f}")
