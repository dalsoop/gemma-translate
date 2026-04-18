//! Claude (Anthropic Messages API) connector — 긴 문장/맥락 필요 시 유용.

use super::{Connector, TranslateResult};
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

pub struct ClaudeConnector {
    api_key: String,
    model: String,
    http: reqwest::Client,
}

impl ClaudeConnector {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            api_key,
            model,
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(60))
                .build().unwrap(),
        }
    }
}

#[derive(Serialize)]
struct MsgReq<'a> {
    model: &'a str,
    max_tokens: u32,
    messages: Vec<MsgItem<'a>>,
}
#[derive(Serialize)]
struct MsgItem<'a> {
    role: &'a str,
    content: String,
}
#[derive(Deserialize)]
struct MsgResp {
    content: Vec<RespContent>,
}
#[derive(Deserialize)]
struct RespContent {
    #[serde(default)]
    text: Option<String>,
}

#[async_trait]
impl Connector for ClaudeConnector {
    fn name(&self) -> &'static str { "claude" }

    async fn translate(
        &self,
        text: &str,
        source: &str,
        target: &str,
        context: Option<&str>,
    ) -> Result<TranslateResult> {
        let t0 = Instant::now();
        let ctx_line = context.map(|c| format!("Context: {c}\n")).unwrap_or_default();
        let prompt = format!(
            "{ctx_line}Translate from {source} to {target}. Output only the translation, no quotes or commentary.\n\n{text}"
        );
        let body = MsgReq {
            model: &self.model,
            max_tokens: 1024,
            messages: vec![MsgItem { role: "user", content: prompt }],
        };
        let resp: MsgResp = self.http.post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body).send().await?.error_for_status()?.json().await?;
        let translation = resp.content.into_iter()
            .filter_map(|c| c.text)
            .collect::<Vec<_>>()
            .join("\n")
            .trim()
            .to_string();
        Ok(TranslateResult {
            translation,
            elapsed_s: t0.elapsed().as_secs_f32(),
            backend: "claude".into(),
        })
    }

    async fn health(&self) -> Result<String> {
        // model info via dummy message — 작은 요청
        let body = MsgReq {
            model: &self.model,
            max_tokens: 4,
            messages: vec![MsgItem { role: "user", content: "hi".into() }],
        };
        let r = self.http.post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body).send().await?;
        Ok(if r.status().is_success() { "ok".into() } else { format!("status {}", r.status()) })
    }
}
