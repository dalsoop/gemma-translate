//! TranslateGemma HTTP 클라이언트 — round-robin 라우팅 + 재시도.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug, Clone, Serialize)]
struct TranslateReq<'a> {
    text: &'a str,
    source_lang_code: &'a str,
    target_lang_code: &'a str,
    max_new_tokens: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TranslateResp {
    pub translation: String,
    #[serde(default)]
    pub elapsed_s: f32,
}

#[derive(Debug, Deserialize)]
pub struct Health {
    pub ok: bool,
    #[serde(default)]
    pub vram_gb: f32,
}

#[derive(Clone)]
pub struct TranslateClient {
    endpoints: Vec<String>,
    counter: Arc<AtomicUsize>,
    http: reqwest::Client,
}

impl TranslateClient {
    pub fn new(endpoints: Vec<String>) -> Self {
        Self {
            endpoints,
            counter: Arc::new(AtomicUsize::new(0)),
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(90))
                .build()
                .expect("reqwest client"),
        }
    }

    pub fn endpoints(&self) -> &[String] { &self.endpoints }

    fn pick(&self) -> &str {
        let n = self.endpoints.len();
        if n == 1 { return &self.endpoints[0]; }
        let i = self.counter.fetch_add(1, Ordering::Relaxed) % n;
        &self.endpoints[i]
    }

    pub async fn health(&self, endpoint: &str) -> Result<Health> {
        let url = format!("{endpoint}/health");
        Ok(self.http.get(&url).send().await?.error_for_status()?.json().await?)
    }

    pub async fn translate(
        &self,
        text: &str,
        src: &str,
        tgt: &str,
        context: Option<&str>,
    ) -> Result<TranslateResp> {
        let payload_text = match context {
            Some(ctx) if !ctx.is_empty() => format!("[context: {ctx}] {text}"),
            _ => text.to_string(),
        };
        let body = TranslateReq {
            text: &payload_text,
            source_lang_code: src,
            target_lang_code: tgt,
            max_new_tokens: 512,
        };
        let mut last_err: Option<anyhow::Error> = None;
        for attempt in 0..3 {
            let ep = self.pick().to_string();
            let url = format!("{ep}/translate");
            match self.http.post(&url).json(&body).send().await {
                Ok(r) if r.status().is_success() => {
                    let parsed: TranslateResp = r.json().await.context("parse resp")?;
                    return Ok(parsed);
                }
                Ok(r) => {
                    last_err = Some(anyhow::anyhow!("http {} from {url}", r.status()));
                }
                Err(e) => {
                    last_err = Some(e.into());
                }
            }
            tokio::time::sleep(Duration::from_millis(500 + 1000 * attempt as u64)).await;
        }
        Err(last_err.unwrap_or_else(|| anyhow::anyhow!("unknown")))
    }
}
