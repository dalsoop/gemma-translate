//! 번역 커넥터 — 다른 백엔드 간 공통 인터페이스.
//!
//! 런타임에 이름으로 선택해서 쓸 수 있음 (Nickel config + TUI 에서 지정).

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub mod claude;
pub mod deepl;
pub mod gemma;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslateResult {
    pub translation: String,
    #[serde(default)]
    pub elapsed_s: f32,
    pub backend: String,
}

#[async_trait]
pub trait Connector: Send + Sync {
    fn name(&self) -> &'static str;

    async fn translate(
        &self,
        text: &str,
        source: &str,
        target: &str,
        context: Option<&str>,
    ) -> Result<TranslateResult>;

    async fn health(&self) -> Result<String>;
}

pub type BoxConnector = Arc<dyn Connector>;

/// 이름 → 커넥터 팩토리.
#[derive(Default)]
pub struct Registry {
    pub connectors: Vec<(String, BoxConnector)>,
}

impl Registry {
    pub fn register(&mut self, name: impl Into<String>, c: BoxConnector) {
        self.connectors.push((name.into(), c));
    }

    pub fn get(&self, name: &str) -> Option<&BoxConnector> {
        self.connectors.iter().find(|(n, _)| n == name).map(|(_, c)| c)
    }

    pub fn names(&self) -> Vec<String> {
        self.connectors.iter().map(|(n, _)| n.clone()).collect()
    }
}
