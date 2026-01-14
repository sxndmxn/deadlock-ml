"""
Deadlock Build Optimizer - Main Streamlit App

A ML-powered tool for finding optimal item build paths per hero in Deadlock.
"""

import streamlit as st

from lib.db import get_heroes

st.set_page_config(
    page_title="Deadlock Build Optimizer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Deadlock Build Optimizer")

# Load heroes for sidebar dropdown
@st.cache_data
def load_heroes():
    try:
        return get_heroes()
    except FileNotFoundError:
        return None


heroes_df = load_heroes()

if heroes_df is None:
    st.error("Data files not found. Please run `python scripts/fetch_data.py` first.")
    st.stop()

# Sidebar: Hero selection
st.sidebar.header("Hero Selection")

sorted_heroes = heroes_df.sort("name")
hero_options = [(row["id"], row["name"]) for row in sorted_heroes.iter_rows(named=True)]
hero_names = [name for _, name in hero_options]
hero_id_map = {name: hero_id for hero_id, name in hero_options}

# Initialize session state for selected hero
if "selected_hero_name" not in st.session_state:
    st.session_state.selected_hero_name = hero_names[0] if hero_names else None

selected_hero = st.sidebar.selectbox(
    "Select Hero",
    options=hero_names,
    index=hero_names.index(st.session_state.selected_hero_name) if st.session_state.selected_hero_name in hero_names else 0,
    key="hero_selector",
)

# Update session state
st.session_state.selected_hero_name = selected_hero
st.session_state.selected_hero_id = hero_id_map.get(selected_hero)

# Main page content
st.markdown("""
Welcome to the **Deadlock Build Optimizer**! This tool uses machine learning to analyze
winning match data and recommend optimal item builds for each hero.

### Features

- **Build Optimizer**: View recommended item build paths with probability scores
- **Item Synergies**: Discover which items work well together using association rules
- **Hero Stats**: Explore hero win rates and item popularity

### How It Works

1. **Data Collection**: Match data is fetched from the Deadlock API
2. **Association Rules**: FP-Growth algorithm finds items frequently bought together in winning games
3. **Markov Chains**: Transition probabilities model optimal item purchase sequences
4. **Win Weighting**: Recommendations are weighted toward builds that lead to victories

### Getting Started

Select a hero from the sidebar, then navigate to one of the analysis pages to explore
build recommendations and item synergies.
""")

# Show selected hero info
if st.session_state.selected_hero_id:
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Selected:** {selected_hero}")
    st.sidebar.markdown(f"Hero ID: `{st.session_state.selected_hero_id}`")
