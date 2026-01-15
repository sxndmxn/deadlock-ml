//! Model store with loading and hot-reload.

use crate::models::{CounterMatrix, HeroModel, Metadata};
use notify_debouncer_mini::{new_debouncer, DebouncedEventKind};
use parking_lot::RwLock;
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

pub struct ModelStore {
    models: RwLock<HashMap<i32, Arc<HeroModel>>>,
    metadata: RwLock<Option<Metadata>>,
    counter_matrix: RwLock<Option<Arc<CounterMatrix>>>,
    models_dir: PathBuf,
}

impl ModelStore {
    pub fn new(models_dir: PathBuf) -> Arc<Self> {
        let store = Arc::new(Self {
            models: RwLock::new(HashMap::new()),
            metadata: RwLock::new(None),
            counter_matrix: RwLock::new(None),
            models_dir,
        });
        store.load_all();
        store
    }

    pub fn load_all(&self) {
        self.reload_metadata();
        self.reload_counter_matrix();

        let Ok(entries) = fs::read_dir(&self.models_dir) else {
            tracing::warn!("Could not read models directory: {:?}", self.models_dir);
            return;
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "json") {
                if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                    if name == "metadata" || name == "counter_matrix" {
                        continue;
                    }
                    if let Ok(hero_id) = name.parse::<i32>() {
                        self.reload_hero(hero_id);
                    }
                }
            }
        }

        let models = self.models.read();
        tracing::info!("Loaded {} hero models", models.len());
    }

    pub fn reload_metadata(&self) {
        let path = self.models_dir.join("metadata.json");
        match fs::read_to_string(&path) {
            Ok(content) => match serde_json::from_str::<Metadata>(&content) {
                Ok(meta) => {
                    tracing::info!("Loaded metadata: {} heroes, {} items", meta.heroes.len(), meta.items.len());
                    *self.metadata.write() = Some(meta);
                }
                Err(e) => tracing::error!("Failed to parse metadata.json: {e}"),
            },
            Err(e) => tracing::warn!("Could not read metadata.json: {e}"),
        }
    }

    pub fn reload_counter_matrix(&self) {
        let path = self.models_dir.join("counter_matrix.json");
        match fs::read_to_string(&path) {
            Ok(content) => match serde_json::from_str::<CounterMatrix>(&content) {
                Ok(matrix) => {
                    tracing::info!(
                        "Loaded counter matrix: {} heroes, {} matchup entries",
                        matrix.metadata.num_heroes,
                        matrix.hero_matchups.len()
                    );
                    *self.counter_matrix.write() = Some(Arc::new(matrix));
                }
                Err(e) => tracing::error!("Failed to parse counter_matrix.json: {e}"),
            },
            Err(e) => tracing::warn!("Could not read counter_matrix.json: {e}"),
        }
    }

    pub fn reload_hero(&self, hero_id: i32) {
        let path = self.models_dir.join(format!("{hero_id}.json"));
        match fs::read_to_string(&path) {
            Ok(content) => match serde_json::from_str::<HeroModel>(&content) {
                Ok(model) => {
                    tracing::info!("Loaded model for {} ({})", model.hero_name, hero_id);
                    self.models.write().insert(hero_id, Arc::new(model));
                }
                Err(e) => tracing::error!("Failed to parse {hero_id}.json: {e}"),
            },
            Err(e) => tracing::warn!("Could not read {hero_id}.json: {e}"),
        }
    }

    pub fn get_hero(&self, hero_id: i32) -> Option<Arc<HeroModel>> {
        self.models.read().get(&hero_id).cloned()
    }

    pub fn get_metadata(&self) -> Option<Metadata> {
        self.metadata.read().clone()
    }

    pub fn get_counter_matrix(&self) -> Option<Arc<CounterMatrix>> {
        self.counter_matrix.read().clone()
    }

    pub fn hero_ids(&self) -> Vec<i32> {
        self.models.read().keys().copied().collect()
    }

    pub fn start_watcher(self: Arc<Self>) {
        let store = self.clone();
        let models_dir = self.models_dir.clone();

        std::thread::spawn(move || {
            let (tx, rx) = std::sync::mpsc::channel();
            let mut debouncer = new_debouncer(Duration::from_millis(500), tx).unwrap();

            debouncer
                .watcher()
                .watch(&models_dir, notify::RecursiveMode::NonRecursive)
                .unwrap();

            tracing::info!("Watching {:?} for model changes", models_dir);

            for result in rx {
                match result {
                    Ok(events) => {
                        for event in events {
                            if event.kind != DebouncedEventKind::Any {
                                continue;
                            }
                            store.handle_file_change(&event.path);
                        }
                    }
                    Err(e) => tracing::error!("Watch error: {e:?}"),
                }
            }
        });
    }

    fn handle_file_change(&self, path: &Path) {
        let Some(filename) = path.file_name().and_then(|s| s.to_str()) else {
            return;
        };

        if filename == "metadata.json" {
            tracing::info!("Reloading metadata");
            self.reload_metadata();
        } else if filename == "counter_matrix.json" {
            tracing::info!("Reloading counter matrix");
            self.reload_counter_matrix();
        } else if filename.ends_with(".json") {
            if let Ok(hero_id) = filename.trim_end_matches(".json").parse::<i32>() {
                tracing::info!("Reloading hero {hero_id}");
                self.reload_hero(hero_id);
            }
        }
    }
}
