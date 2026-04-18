//! DeepL connector (Free or Pro API).
//! API: https://api-free.deepl.com/v2/translate (free) or https://api.deepl.com/v2/translate (pro)

use super::{Connector, TranslateResult};
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::Deserialize;
use std::time::{Duration, Instant};

pub struct DeeplConnector {
    api_key: String,
    base_url: String,
    http: reqwest::Client,
}

impl DeeplConnector {
    pub fn new(api_key: String, pro: bool) -> Self {
        let base_url = if pro { "https://api.deepl.com" } else { "https://api-free.deepl.com" };
        Self {
            api_key,
            base_url: base_url.into(),
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .build().unwrap(),
        }
    }
}

#[derive(Deserialize)]
struct DeeplResp {
    translations: Vec<DeeplItem>,
}
#[derive(Deserialize)]
struct DeeplItem {
    text: String,
}

#[async_trait]
impl Connector for DeeplConnector {
    fn name(&self) -> &'static str { "deepl" }

    async fn translate(
        &self,
        text: &str,
        source: &str,
        target: &str,
        _context: Option<&str>,
    ) -> Result<TranslateResult> {
        let t0 = Instant::now();
        let url = format!("{}/v2/translate", self.base_url);
        let resp: DeeplResp = self.http.post(&url)
            .header("Authorization", format!("DeepL-Auth-Key {}", self.api_key))
            .form(&[
                ("text", text),
                ("source_lang", &source.to_uppercase()),
                ("target_lang", &target.to_uppercase()),
            ])
            .send().await?.error_for_status()?.json().await.context("deepl json")?;
        let translation = resp.translations.into_iter().next()
            .map(|t| t.text).unwrap_or_default();
        Ok(TranslateResult {
            translation,
            elapsed_s: t0.elapsed().as_secs_f32(),
            backend: "deepl".into(),
        })
    }

    async fn health(&self) -> Result<String> {
        let url = format!("{}/v2/usage", self.base_url);
        let r = self.http.get(&url)
            .header("Authorization", format!("DeepL-Auth-Key {}", self.api_key))
            .send().await?;
        if r.status().is_success() {
            Ok("ok".into())
        } else {
            Ok(format!("status {}", r.status()))
        }
    }
}
