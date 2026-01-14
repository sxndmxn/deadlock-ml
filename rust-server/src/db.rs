//! Database module for match history and leaderboards using DuckDB.

use duckdb::{params, Connection, Result as DuckResult};
use parking_lot::Mutex;
use serde::Serialize;
use std::path::PathBuf;

/// Summary of a match for display in match list.
#[derive(Debug, Clone, Serialize)]
pub struct MatchSummary {
    pub match_id: i64,
    pub hero_id: i32,
    pub account_id: i64,
    pub won: bool,
    pub kills: i32,
    pub deaths: i32,
    pub assists: i32,
    pub net_worth: i32,
}

/// Query parameters for filtering match list.
#[derive(Debug, Default)]
pub struct MatchListQuery {
    pub hero_id: Option<i32>,
    pub account_id: Option<i64>,
    pub won: Option<bool>,
    pub limit: u32,
    pub offset: u32,
}

/// Database wrapper providing access to match history via DuckDB.
pub struct Database {
    conn: Mutex<Connection>,
    data_dir: PathBuf,
}

impl Database {
    /// Create a new database connection.
    /// Uses in-memory DuckDB with parquet file queries.
    pub fn new(data_dir: PathBuf) -> DuckResult<Self> {
        let conn = Connection::open_in_memory()?;

        // Enable parquet extension (bundled by default)
        conn.execute_batch("INSTALL parquet; LOAD parquet;")?;

        Ok(Self {
            conn: Mutex::new(conn),
            data_dir,
        })
    }

    /// Get list of matches with optional filters.
    pub fn get_match_list(&self, query: &MatchListQuery) -> DuckResult<Vec<MatchSummary>> {
        let conn = self.conn.lock();
        let parquet_path = self.data_dir.join("match_metadata/*.parquet");

        // Build query with dynamic filters
        let mut sql = format!(
            r#"
            SELECT
                match_id,
                hero_id,
                account_id,
                won,
                kills,
                deaths,
                assists,
                net_worth
            FROM read_parquet('{}')
            WHERE 1=1
            "#,
            parquet_path.display()
        );

        // Add filters
        if query.hero_id.is_some() {
            sql.push_str(" AND hero_id = ?");
        }
        if query.account_id.is_some() {
            sql.push_str(" AND account_id = ?");
        }
        if query.won.is_some() {
            sql.push_str(" AND won = ?");
        }

        sql.push_str(" ORDER BY match_id DESC");
        sql.push_str(&format!(" LIMIT {} OFFSET {}", query.limit, query.offset));

        let mut stmt = conn.prepare(&sql)?;

        // Collect results
        let mut results = Vec::new();

        // Build params dynamically based on which filters are present
        if let (Some(hero_id), Some(account_id), Some(won)) =
            (query.hero_id, query.account_id, query.won)
        {
            let rows = stmt.query_map(params![hero_id, account_id, won], |row| {
                Ok(MatchSummary {
                    match_id: row.get(0)?,
                    hero_id: row.get(1)?,
                    account_id: row.get(2)?,
                    won: row.get(3)?,
                    kills: row.get(4)?,
                    deaths: row.get(5)?,
                    assists: row.get(6)?,
                    net_worth: row.get(7)?,
                })
            })?;
            for row in rows {
                results.push(row?);
            }
        } else if let (Some(hero_id), Some(account_id)) = (query.hero_id, query.account_id) {
            let rows = stmt.query_map(params![hero_id, account_id], |row| {
                Ok(MatchSummary {
                    match_id: row.get(0)?,
                    hero_id: row.get(1)?,
                    account_id: row.get(2)?,
                    won: row.get(3)?,
                    kills: row.get(4)?,
                    deaths: row.get(5)?,
                    assists: row.get(6)?,
                    net_worth: row.get(7)?,
                })
            })?;
            for row in rows {
                results.push(row?);
            }
        } else if let (Some(hero_id), Some(won)) = (query.hero_id, query.won) {
            let rows = stmt.query_map(params![hero_id, won], |row| {
                Ok(MatchSummary {
                    match_id: row.get(0)?,
                    hero_id: row.get(1)?,
                    account_id: row.get(2)?,
                    won: row.get(3)?,
                    kills: row.get(4)?,
                    deaths: row.get(5)?,
                    assists: row.get(6)?,
                    net_worth: row.get(7)?,
                })
            })?;
            for row in rows {
                results.push(row?);
            }
        } else if let (Some(account_id), Some(won)) = (query.account_id, query.won) {
            let rows = stmt.query_map(params![account_id, won], |row| {
                Ok(MatchSummary {
                    match_id: row.get(0)?,
                    hero_id: row.get(1)?,
                    account_id: row.get(2)?,
                    won: row.get(3)?,
                    kills: row.get(4)?,
                    deaths: row.get(5)?,
                    assists: row.get(6)?,
                    net_worth: row.get(7)?,
                })
            })?;
            for row in rows {
                results.push(row?);
            }
        } else if let Some(hero_id) = query.hero_id {
            let rows = stmt.query_map(params![hero_id], |row| {
                Ok(MatchSummary {
                    match_id: row.get(0)?,
                    hero_id: row.get(1)?,
                    account_id: row.get(2)?,
                    won: row.get(3)?,
                    kills: row.get(4)?,
                    deaths: row.get(5)?,
                    assists: row.get(6)?,
                    net_worth: row.get(7)?,
                })
            })?;
            for row in rows {
                results.push(row?);
            }
        } else if let Some(account_id) = query.account_id {
            let rows = stmt.query_map(params![account_id], |row| {
                Ok(MatchSummary {
                    match_id: row.get(0)?,
                    hero_id: row.get(1)?,
                    account_id: row.get(2)?,
                    won: row.get(3)?,
                    kills: row.get(4)?,
                    deaths: row.get(5)?,
                    assists: row.get(6)?,
                    net_worth: row.get(7)?,
                })
            })?;
            for row in rows {
                results.push(row?);
            }
        } else if let Some(won) = query.won {
            let rows = stmt.query_map(params![won], |row| {
                Ok(MatchSummary {
                    match_id: row.get(0)?,
                    hero_id: row.get(1)?,
                    account_id: row.get(2)?,
                    won: row.get(3)?,
                    kills: row.get(4)?,
                    deaths: row.get(5)?,
                    assists: row.get(6)?,
                    net_worth: row.get(7)?,
                })
            })?;
            for row in rows {
                results.push(row?);
            }
        } else {
            let rows = stmt.query_map([], |row| {
                Ok(MatchSummary {
                    match_id: row.get(0)?,
                    hero_id: row.get(1)?,
                    account_id: row.get(2)?,
                    won: row.get(3)?,
                    kills: row.get(4)?,
                    deaths: row.get(5)?,
                    assists: row.get(6)?,
                    net_worth: row.get(7)?,
                })
            })?;
            for row in rows {
                results.push(row?);
            }
        }

        Ok(results)
    }

    /// Get total count of matches (for pagination).
    pub fn get_match_count(&self, query: &MatchListQuery) -> DuckResult<u64> {
        let conn = self.conn.lock();
        let parquet_path = self.data_dir.join("match_metadata/*.parquet");

        let mut sql = format!(
            r#"
            SELECT COUNT(*)
            FROM read_parquet('{}')
            WHERE 1=1
            "#,
            parquet_path.display()
        );

        if query.hero_id.is_some() {
            sql.push_str(" AND hero_id = ?");
        }
        if query.account_id.is_some() {
            sql.push_str(" AND account_id = ?");
        }
        if query.won.is_some() {
            sql.push_str(" AND won = ?");
        }

        let mut stmt = conn.prepare(&sql)?;

        // Bind parameters based on which filters are present
        let count: i64 = if let (Some(hero_id), Some(account_id), Some(won)) =
            (query.hero_id, query.account_id, query.won)
        {
            stmt.query_row(params![hero_id, account_id, won], |row| row.get(0))?
        } else if let (Some(hero_id), Some(account_id)) = (query.hero_id, query.account_id) {
            stmt.query_row(params![hero_id, account_id], |row| row.get(0))?
        } else if let (Some(hero_id), Some(won)) = (query.hero_id, query.won) {
            stmt.query_row(params![hero_id, won], |row| row.get(0))?
        } else if let (Some(account_id), Some(won)) = (query.account_id, query.won) {
            stmt.query_row(params![account_id, won], |row| row.get(0))?
        } else if let Some(hero_id) = query.hero_id {
            stmt.query_row(params![hero_id], |row| row.get(0))?
        } else if let Some(account_id) = query.account_id {
            stmt.query_row(params![account_id], |row| row.get(0))?
        } else if let Some(won) = query.won {
            stmt.query_row(params![won], |row| row.get(0))?
        } else {
            stmt.query_row([], |row| row.get(0))?
        };

        Ok(count as u64)
    }
}
