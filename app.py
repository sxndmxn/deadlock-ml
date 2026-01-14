"""
Deadlock Build Optimizer - Main Streamlit App
Top navigation with tabs, no sidebar.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import polars as pl
from streamlit_agraph import agraph, Node, Edge, Config

from lib.db import get_heroes, get_hero_stats, get_hero_item_stats
from lib.items import get_item_names_map
from ml.precompute import load_model
from ml.markov import get_next_item_probabilities, find_optimal_path, START_STATE

st.set_page_config(
    page_title="Deadlock Build Optimizer",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load heroes
@st.cache_data
def load_heroes():
    try:
        return get_heroes()
    except FileNotFoundError:
        return None

heroes_df = load_heroes()

if heroes_df is None:
    st.error("Data files not found. Run `python scripts/fetch_data.py` first.")
    st.stop()

# Get item names
item_names = get_item_names_map()

# Top bar: Title + Hero selector
col_title, col_hero = st.columns([3, 1])

with col_title:
    st.title("Deadlock Build Optimizer")

with col_hero:
    sorted_heroes = heroes_df.sort("name")
    hero_options = [(row["id"], row["name"]) for row in sorted_heroes.iter_rows(named=True)]
    hero_names = [name for _, name in hero_options]
    hero_id_map = {name: hero_id for hero_id, name in hero_options}

    selected_hero = st.selectbox("Hero", options=hero_names, label_visibility="collapsed")
    hero_id = hero_id_map.get(selected_hero)
    hero_name = selected_hero

# Navigation tabs
tab_build, tab_synergies, tab_stats = st.tabs(["Build Optimizer", "Item Synergies", "Hero Stats"])

# ============================================================================
# TAB 1: Build Optimizer
# ============================================================================
with tab_build:
    model = load_model(hero_id)

    if model is None:
        st.warning(f"No model available for {hero_name}. Run precompute first.")
    else:
        st.caption(f"Based on {model['match_count']:,} matches ({model['win_count']:,} wins)")

        def display_build_path(markov_model: dict, length: int = 6):
            if not markov_model or markov_model["matrix"].size == 0:
                st.warning("No data available.")
                return

            matrix = markov_model["matrix"]
            item_to_idx = markov_model["item_to_idx"]
            idx_to_item = markov_model["idx_to_item"]

            path = find_optimal_path(matrix, item_to_idx, idx_to_item, START_STATE, length)

            if not path:
                st.warning("Could not generate build path.")
                return

            for i, (item_id, prob) in enumerate(path, 1):
                name = item_names.get(item_id, f"Item {item_id}")
                st.markdown(f"{i}. **{name}** ({prob*100:.1f}%)")

        def create_sankey(markov_model: dict, top_n: int = 5):
            if not markov_model or markov_model["matrix"].size == 0:
                return None

            matrix = markov_model["matrix"]
            item_to_idx = markov_model["item_to_idx"]
            idx_to_item = markov_model["idx_to_item"]

            start_probs = get_next_item_probabilities(START_STATE, matrix, item_to_idx, idx_to_item, top_k=top_n)
            if not start_probs:
                return None

            labels = ["START"]
            label_to_idx = {"START": 0}
            sources, targets, values = [], [], []

            for item_id, prob in start_probs:
                name = item_names.get(item_id, f"Item {item_id}")
                key = f"1_{name}"
                if key not in label_to_idx:
                    label_to_idx[key] = len(labels)
                    labels.append(name)

                sources.append(0)
                targets.append(label_to_idx[key])
                values.append(prob * 100)

                next_probs = get_next_item_probabilities(item_id, matrix, item_to_idx, idx_to_item, top_k=3)
                for next_id, next_prob in next_probs:
                    if next_id == item_id:
                        continue
                    next_name = item_names.get(next_id, f"Item {next_id}")
                    next_key = f"2_{next_name}"
                    if next_key not in label_to_idx:
                        label_to_idx[next_key] = len(labels)
                        labels.append(next_name)
                    sources.append(label_to_idx[key])
                    targets.append(label_to_idx[next_key])
                    values.append(prob * next_prob * 100)

            if not sources:
                return None

            fig = go.Figure(go.Sankey(
                node=dict(pad=15, thickness=20, label=labels, color="lightblue"),
                link=dict(source=sources, target=targets, value=values),
            ))
            fig.update_layout(height=400, margin=dict(t=20, b=20))
            return fig

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Recommended Build")
            display_build_path(model["markov_model"])

        with col2:
            st.subheader("Build Flow")
            fig = create_sankey(model["markov_model"])
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("What Should I Buy Next?")

        all_item_ids = [v for v in model["markov_model"]["idx_to_item"].values() if v != START_STATE]
        item_options = ["(Select item)"] + sorted([item_names.get(i, f"Item {i}") for i in all_item_ids])

        selected_item = st.selectbox("I just bought:", options=item_options)

        if selected_item != "(Select item)":
            sel_id = next((k for k, v in item_names.items() if v == selected_item), None)
            if sel_id:
                matrix = model["markov_model"]["matrix"]
                item_to_idx = model["markov_model"]["item_to_idx"]
                idx_to_item = model["markov_model"]["idx_to_item"]

                next_items = get_next_item_probabilities(sel_id, matrix, item_to_idx, idx_to_item, top_k=5)
                if next_items:
                    for item_id, prob in next_items:
                        st.markdown(f"- **{item_names.get(item_id, item_id)}** ({prob*100:.1f}%)")

# ============================================================================
# TAB 2: Item Synergies
# ============================================================================
with tab_synergies:
    model = load_model(hero_id)

    if model is None:
        st.warning(f"No model available for {hero_name}.")
    else:
        rules_df = model["association_rules"]

        if rules_df.empty:
            st.warning("No association rules found.")
        else:
            def frozenset_to_names(fs):
                return ", ".join(sorted([item_names.get(i, f"Item {i}") for i in fs]))

            # Filter control
            all_items_in_rules = set()
            for _, row in rules_df.iterrows():
                all_items_in_rules.update(row["antecedents"])
                all_items_in_rules.update(row["consequents"])

            filter_options = ["Show all"] + sorted([item_names.get(i, f"Item {i}") for i in all_items_in_rules])
            selected_filter = st.selectbox("Filter by item:", options=filter_options)

            if selected_filter != "Show all":
                filter_id = next((k for k, v in item_names.items() if v == selected_filter), None)
                if filter_id:
                    mask = rules_df.apply(lambda r: filter_id in r["antecedents"] or filter_id in r["consequents"], axis=1)
                    filtered_rules = rules_df[mask]
                else:
                    filtered_rules = rules_df
            else:
                filtered_rules = rules_df

            # Network graph
            st.subheader("Item Relationship Graph")

            def build_graph(rules, max_edges=50):
                if rules.empty:
                    return [], []

                sorted_rules = rules.sort_values("lift", ascending=False).head(max_edges)
                node_ids = set()
                edges_data = []

                for _, row in sorted_rules.iterrows():
                    for ant in row["antecedents"]:
                        for cons in row["consequents"]:
                            node_ids.add(ant)
                            node_ids.add(cons)
                            edges_data.append({"source": ant, "target": cons, "confidence": row["confidence"], "lift": row["lift"]})

                nodes = [Node(id=str(i), label=item_names.get(i, f"Item {i}"), size=20, color="#97C2FC") for i in node_ids]
                edges = [Edge(source=str(e["source"]), target=str(e["target"]), width=max(1, e["confidence"]*5)) for e in edges_data]
                return nodes, edges

            nodes, edges = build_graph(filtered_rules)
            if nodes:
                config = Config(width=800, height=500, directed=True, physics=True)
                agraph(nodes=nodes, edges=edges, config=config)

            # Rules table
            st.divider()
            st.subheader("Association Rules")

            display_df = filtered_rules.copy()
            display_df["Antecedent"] = display_df["antecedents"].apply(frozenset_to_names)
            display_df["Consequent"] = display_df["consequents"].apply(frozenset_to_names)
            display_df["Support"] = display_df["support"].apply(lambda x: f"{x*100:.1f}%")
            display_df["Confidence"] = display_df["confidence"].apply(lambda x: f"{x*100:.1f}%")
            display_df["Lift"] = display_df["lift"].apply(lambda x: f"{x:.2f}")

            st.dataframe(
                display_df[["Antecedent", "Consequent", "Support", "Confidence", "Lift"]].head(20),
                use_container_width=True,
                hide_index=True,
            )

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Rules", len(filtered_rules))
            col2.metric("Avg Confidence", f"{filtered_rules['confidence'].mean()*100:.1f}%")
            col3.metric("Avg Lift", f"{filtered_rules['lift'].mean():.2f}")

# ============================================================================
# TAB 3: Hero Stats
# ============================================================================
with tab_stats:
    @st.cache_data
    def load_hero_stats():
        try:
            stats = get_hero_stats()
            heroes = get_heroes()
            return stats.join(heroes, left_on="hero_id", right_on="id", how="left")
        except FileNotFoundError:
            return None

    hero_stats = load_hero_stats()

    if hero_stats is None:
        st.warning("No stats available.")
    else:
        st.subheader("Hero Win Rates")

        chart_df = hero_stats.with_columns([
            (pl.col("win_rate") * 100).alias("win_rate_pct"),
            (pl.col("hero_id") == hero_id).alias("is_selected"),
        ]).sort("win_rate_pct")

        fig = px.bar(
            chart_df.to_pandas(),
            x="win_rate_pct", y="name", orientation="h",
            color="is_selected",
            color_discrete_map={True: "#ff6b6b", False: "#4dabf7"},
            labels={"win_rate_pct": "Win Rate (%)", "name": "Hero"},
        )
        fig.update_layout(showlegend=False, height=max(400, len(chart_df)*25))
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader(f"Item Stats for {hero_name}")

        hero_row = hero_stats.filter(pl.col("hero_id") == hero_id)
        if hero_row.height > 0:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Matches", f"{hero_row['total_matches'][0]:,}")
            col2.metric("Win Rate", f"{hero_row['win_rate'][0]*100:.1f}%")
            total_all = hero_stats["total_matches"].sum()
            col3.metric("Pick Rate", f"{hero_row['total_matches'][0]/total_all*100:.1f}%")

        item_stats = get_hero_item_stats(hero_id)
        if item_stats is not None and not item_stats.is_empty():
            display_df = item_stats.with_columns([
                pl.col("item_id").map_elements(lambda x: item_names.get(x, f"Item {x}"), return_dtype=pl.Utf8).alias("Item"),
                (pl.col("win_rate") * 100).alias("Win %"),
            ]).sort("win_rate", descending=True)

            st.dataframe(
                display_df.select(["Item", "times_bought", "wins", "Win %"]).to_pandas(),
                use_container_width=True,
                hide_index=True,
            )
