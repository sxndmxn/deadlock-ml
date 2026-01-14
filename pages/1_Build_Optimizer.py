"""
Build Optimizer Page - Recommended build paths for selected hero.
"""

import streamlit as st
import plotly.graph_objects as go

from ml.precompute import load_model
from ml.markov import get_next_item_probabilities, find_optimal_path, START_STATE
from lib.items import get_item_name, get_item_names_map

st.set_page_config(
    page_title="Build Optimizer - Deadlock",
    page_icon="🎯",
    layout="wide",
)

st.title("Build Optimizer")

# Check for selected hero
if "selected_hero_id" not in st.session_state or st.session_state.selected_hero_id is None:
    st.warning("Please select a hero from the sidebar on the main page first.")
    st.stop()

hero_id = st.session_state.selected_hero_id
hero_name = st.session_state.selected_hero_name

st.subheader(f"Recommended Builds for {hero_name}")

# Load precomputed model
model = load_model(hero_id)

if model is None:
    st.error(f"Insufficient data for {hero_name}. No model available.")
    st.info("This hero may not have enough match data to generate reliable recommendations.")
    st.stop()

# Display match count
st.caption(f"Based on {model['match_count']:,} matches ({model['win_count']:,} wins)")

# Get item name mapping
item_names = get_item_names_map()

# Tabs for different slots
tab_all, tab_weapon, tab_vitality, tab_spirit = st.tabs(
    ["All Items", "Weapon", "Vitality", "Spirit"]
)


def display_build_path(markov_model: dict, title: str, length: int = 6):
    """Display optimal build path from Markov model."""
    if not markov_model or markov_model["matrix"].size == 0:
        st.warning("No data available for this slot.")
        return

    matrix = markov_model["matrix"]
    item_to_idx = markov_model["item_to_idx"]
    idx_to_item = markov_model["idx_to_item"]

    # Get optimal path
    path = find_optimal_path(matrix, item_to_idx, idx_to_item, START_STATE, length)

    if not path:
        st.warning("Could not generate build path.")
        return

    # Display as ordered list
    st.markdown(f"**{title}**")

    for i, (item_id, prob) in enumerate(path, 1):
        item_name = item_names.get(item_id, f"Item {item_id}")
        prob_pct = prob * 100
        st.markdown(f"{i}. **{item_name}** ({prob_pct:.1f}%)")


def create_sankey_diagram(markov_model: dict, top_n: int = 5) -> go.Figure | None:
    """Create Sankey diagram showing top build paths."""
    if not markov_model or markov_model["matrix"].size == 0:
        return None

    matrix = markov_model["matrix"]
    item_to_idx = markov_model["item_to_idx"]
    idx_to_item = markov_model["idx_to_item"]

    # Get multiple paths by varying starting approach
    # We'll show transitions from START and then top transitions
    paths_data = []

    # Get top first items from START
    start_probs = get_next_item_probabilities(
        START_STATE, matrix, item_to_idx, idx_to_item, top_k=top_n
    )

    if not start_probs:
        return None

    # Build Sankey data
    labels = ["START"]
    label_to_idx = {"START": 0}
    sources = []
    targets = []
    values = []

    # Add first level (START -> items)
    for item_id, prob in start_probs:
        item_name = item_names.get(item_id, f"Item {item_id}")
        label_key = f"1_{item_name}"

        if label_key not in label_to_idx:
            label_to_idx[label_key] = len(labels)
            labels.append(item_name)

        sources.append(0)  # START
        targets.append(label_to_idx[label_key])
        values.append(prob * 100)

        # Add second level transitions
        next_probs = get_next_item_probabilities(
            item_id, matrix, item_to_idx, idx_to_item, top_k=3
        )

        for next_item_id, next_prob in next_probs:
            if next_item_id == item_id:
                continue
            next_name = item_names.get(next_item_id, f"Item {next_item_id}")
            next_label_key = f"2_{next_name}"

            if next_label_key not in label_to_idx:
                label_to_idx[next_label_key] = len(labels)
                labels.append(next_name)

            sources.append(label_to_idx[label_key])
            targets.append(label_to_idx[next_label_key])
            values.append(prob * next_prob * 100)

    if not sources:
        return None

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color="lightblue",
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
        ),
    ))

    fig.update_layout(
        title_text="Build Path Flow",
        font_size=12,
        height=400,
    )

    return fig


# All Items tab
with tab_all:
    col1, col2 = st.columns([1, 2])

    with col1:
        display_build_path(model["markov_model"], "Recommended Build Order")

    with col2:
        fig = create_sankey_diagram(model["markov_model"])
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data for flow diagram.")

# Slot-specific tabs
for tab, slot_name in [(tab_weapon, "weapon"), (tab_vitality, "vitality"), (tab_spirit, "spirit")]:
    with tab:
        slot_model = model["slot_models"].get(slot_name)

        col1, col2 = st.columns([1, 2])

        with col1:
            display_build_path(
                slot_model,
                f"Recommended {slot_name.title()} Build",
                length=4,
            )

        with col2:
            if slot_model:
                fig = create_sankey_diagram(slot_model, top_n=4)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Insufficient data for flow diagram.")
            else:
                st.info(f"No {slot_name} item data available.")

# What Next section
st.markdown("---")
st.subheader("What Should I Buy Next?")

# Get all items from the model
all_item_ids = [
    idx_to_item
    for idx_to_item in model["markov_model"]["idx_to_item"].values()
    if idx_to_item != START_STATE
]

item_options = ["(Select starting item)"] + sorted(
    [item_names.get(item_id, f"Item {item_id}") for item_id in all_item_ids]
)

selected_item_name = st.selectbox("I just bought:", options=item_options)

if selected_item_name != "(Select starting item)":
    # Find item ID from name
    selected_item_id = None
    for item_id, name in item_names.items():
        if name == selected_item_name:
            selected_item_id = item_id
            break

    if selected_item_id:
        matrix = model["markov_model"]["matrix"]
        item_to_idx = model["markov_model"]["item_to_idx"]
        idx_to_item = model["markov_model"]["idx_to_item"]

        next_items = get_next_item_probabilities(
            selected_item_id, matrix, item_to_idx, idx_to_item, top_k=5
        )

        if next_items:
            st.markdown("**Recommended next items:**")
            for item_id, prob in next_items:
                name = item_names.get(item_id, f"Item {item_id}")
                st.markdown(f"- {name} ({prob*100:.1f}%)")
        else:
            st.info("No recommendations available for this item.")
