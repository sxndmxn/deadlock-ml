"""
Hero Stats Page - Hero win rates and item statistics.
"""

import streamlit as st
import plotly.express as px
import polars as pl

from lib.db import get_heroes, get_hero_stats, get_hero_item_stats
from lib.items import get_item_names_map

st.set_page_config(
    page_title="Hero Stats - Deadlock",
    page_icon="📊",
    layout="wide",
)

st.title("Hero Stats")


@st.cache_data
def load_hero_stats():
    """Load hero statistics."""
    try:
        stats = get_hero_stats()
        heroes = get_heroes()
        # Merge to get hero names
        merged = stats.join(heroes, left_on="hero_id", right_on="id", how="left")
        return merged
    except FileNotFoundError:
        return None


@st.cache_data
def load_item_stats(hero_id: int):
    """Load item statistics for a hero."""
    try:
        return get_hero_item_stats(hero_id)
    except FileNotFoundError:
        return None


# Load stats
hero_stats = load_hero_stats()

if hero_stats is None:
    st.error("Data files not found. Please run `python scripts/fetch_data.py` first.")
    st.stop()

# Get item name mapping
item_names = get_item_names_map()

# Hero win rate chart
st.subheader("Hero Win Rates")

# Prepare data for chart
chart_df = hero_stats.with_columns(
    (pl.col("win_rate") * 100).alias("win_rate_pct")
).sort("win_rate_pct")

# Highlight selected hero
selected_hero_id = st.session_state.get("selected_hero_id")
selected_hero_name = st.session_state.get("selected_hero_name")

if selected_hero_id:
    chart_df = chart_df.with_columns(
        (pl.col("hero_id") == selected_hero_id).alias("is_selected")
    )
else:
    chart_df = chart_df.with_columns(
        pl.lit(False).alias("is_selected")
    )

# Convert to pandas for plotly
chart_pd = chart_df.to_pandas()

# Create bar chart
fig = px.bar(
    chart_pd,
    x="win_rate_pct",
    y="name",
    orientation="h",
    color="is_selected",
    color_discrete_map={True: "#ff6b6b", False: "#4dabf7"},
    labels={"win_rate_pct": "Win Rate (%)", "name": "Hero"},
    title="Win Rate by Hero",
)

fig.update_layout(
    showlegend=False,
    height=max(400, len(chart_pd) * 25),
    yaxis=dict(tickfont=dict(size=10)),
)

fig.update_traces(
    hovertemplate="<b>%{y}</b><br>Win Rate: %{x:.1f}%<extra></extra>"
)

st.plotly_chart(fig, use_container_width=True)

# Selected hero stats
st.markdown("---")

if selected_hero_id is None:
    st.info("Select a hero from the sidebar on the main page to see detailed item statistics.")
    st.stop()

st.subheader(f"Item Stats for {selected_hero_name}")

# Get hero-specific data
hero_row = hero_stats.filter(pl.col("hero_id") == selected_hero_id)

if hero_row.height > 0:
    col1, col2, col3 = st.columns(3)

    with col1:
        total_matches = hero_row["total_matches"][0]
        st.metric("Total Matches", f"{total_matches:,}")

    with col2:
        wins = hero_row["wins"][0]
        win_rate = hero_row["win_rate"][0] * 100
        st.metric("Win Rate", f"{win_rate:.1f}%", delta=f"{wins:,} wins")

    with col3:
        # Calculate pick rate (matches / total across all heroes)
        total_all = hero_stats["total_matches"].sum()
        pick_rate = (total_matches / total_all) * 100 if total_all > 0 else 0
        st.metric("Pick Rate", f"{pick_rate:.1f}%")

# Item win rates table
st.markdown("---")
st.subheader("Item Win Rates")

item_stats = load_item_stats(selected_hero_id)

if item_stats is None or item_stats.is_empty():
    st.warning("No item statistics available for this hero.")
    st.stop()

# Prepare display dataframe
display_df = item_stats.with_columns([
    pl.col("item_id").map_elements(
        lambda x: item_names.get(x, f"Item {x}"), return_dtype=pl.Utf8
    ).alias("Item Name"),
    pl.col("times_bought").alias("Times Bought"),
    pl.col("wins").alias("Wins"),
    (pl.col("win_rate") * 100).alias("Win Rate Value"),
]).with_columns(
    pl.col("Win Rate Value").map_elements(
        lambda x: f"{x:.1f}%", return_dtype=pl.Utf8
    ).alias("Win Rate")
)

# Sort controls
sort_options = {"Win Rate (High to Low)": ("Win Rate Value", True),
                "Win Rate (Low to High)": ("Win Rate Value", False),
                "Popularity (High to Low)": ("Times Bought", True),
                "Popularity (Low to High)": ("Times Bought", False)}

sort_by = st.selectbox("Sort by:", options=list(sort_options.keys()))
sort_col, descending = sort_options[sort_by]

display_df = display_df.sort(sort_col, descending=descending)

# Display table
st.dataframe(
    display_df.select(["Item Name", "Times Bought", "Wins", "Win Rate"]).to_pandas(),
    use_container_width=True,
    hide_index=True,
)

# Top items visualization
st.markdown("---")
st.subheader("Top Items by Win Rate")

# Filter to items with sufficient sample size
min_purchases = st.slider("Minimum purchases to include:", 10, 500, 50)
filtered_items = item_stats.filter(pl.col("times_bought") >= min_purchases)

if filtered_items.is_empty():
    st.warning(f"No items with at least {min_purchases} purchases.")
else:
    filtered_items = filtered_items.with_columns([
        pl.col("item_id").map_elements(
            lambda x: item_names.get(x, f"Item {x}"), return_dtype=pl.Utf8
        ).alias("Item Name"),
        (pl.col("win_rate") * 100).alias("Win Rate %"),
    ])

    # Top 15 by win rate
    top_items = filtered_items.sort("win_rate", descending=True).head(15)

    fig2 = px.bar(
        top_items.to_pandas(),
        x="Win Rate %",
        y="Item Name",
        orientation="h",
        color="Win Rate %",
        color_continuous_scale="RdYlGn",
        labels={"Win Rate %": "Win Rate (%)"},
        title=f"Top Items by Win Rate (min {min_purchases} purchases)",
    )

    fig2.update_layout(
        height=400,
        yaxis=dict(categoryorder="total ascending"),
    )

    st.plotly_chart(fig2, use_container_width=True)
