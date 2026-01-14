mod db;
mod handlers;
mod markov;
mod models;
mod store;

use axum::{routing::get, Router};
use std::{net::SocketAddr, path::PathBuf};
use tower_http::services::ServeDir;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::store::ModelStore;

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Determine models directory (relative to project root)
    let models_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("models");

    tracing::info!("Loading models from {:?}", models_dir);

    // Initialize model store
    let store = ModelStore::new(models_dir);

    // Start file watcher for hot-reload
    store.clone().start_watcher();

    // Build router
    let app = Router::new()
        // Main page
        .route("/", get(handlers::index))
        // HTMX partials
        .route("/htmx/build-path/{hero_id}", get(handlers::build_path))
        .route("/htmx/next-items/{hero_id}/{item_id}", get(handlers::next_items))
        .route("/htmx/sankey/{hero_id}", get(handlers::sankey_data))
        .route("/htmx/synergies/{hero_id}", get(handlers::synergies))
        .route("/htmx/synergy-graph/{hero_id}", get(handlers::synergy_graph))
        .route("/htmx/hero-stats", get(handlers::hero_stats))
        .route("/htmx/all-items/{hero_id}", get(handlers::all_items))
        // Static files
        .nest_service("/static", ServeDir::new("static"))
        .with_state(store);

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    tracing::info!("Server running at http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
