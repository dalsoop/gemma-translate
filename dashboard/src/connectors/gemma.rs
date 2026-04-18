//! Gemma connector — 기존 TranslateClient 래핑 (자체 TranslateGemma 서버).

use super::{Connector, TranslateResult};
use crate::backend::translate::TranslateClient;
use anyhow::Result;
use async_trait::async_trait;

pub struct GemmaConnector {
    client: TranslateClient,
}

impl GemmaConnector {
    pub fn new(endpoints: Vec<String>) -> Self {
        Self { client: TranslateClient::new(endpoints) }
    }
}

#[async_trait]
impl Connector for GemmaConnector {
    fn name(&self) -> &'static str { "gemma" }

    async fn translate(
        &self,
        text: &str,
        source: &str,
        target: &str,
        context: Option<&str>,
    ) -> Result<TranslateResult> {
        let r = self.client.translate(text, source, target, context).await?;
        Ok(TranslateResult {
            translation: r.translation,
            elapsed_s: r.elapsed_s,
            backend: "gemma".into(),
        })
    }

    async fn health(&self) -> Result<String> {
        use std::time::Duration;
        let eps = self.client.endpoints().to_vec();
        let total = eps.len();
        // 병렬 spawn + timeout 3s
        let mut handles = Vec::with_capacity(total);
        for ep in eps {
            let c = self.client.clone();
            handles.push(tokio::spawn(async move {
                tokio::time::timeout(Duration::from_secs(3), c.health(&ep)).await
            }));
        }
        let mut ok = 0usize;
        let mut busy = 0usize;
        let mut down = 0usize;
        for h in handles {
            match h.await {
                Ok(Ok(Ok(hh))) if hh.ok => ok += 1,
                Ok(Ok(_)) => down += 1,
                Ok(Err(_)) => busy += 1,  // timeout
                Err(_) => down += 1,       // join error
            }
        }
        Ok(format!("{ok}/{total} ok · {busy} busy · {down} down"))
    }
}
