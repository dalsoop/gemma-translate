use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;
use std::process::Command;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub api_endpoints: Vec<String>,
    pub gpu: GpuConfig,
    pub defaults: Defaults,
    pub jobs: JobsConfig,
    pub ui: UiConfig,
    #[serde(default)]
    pub connectors: ConnectorsConfig,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct ConnectorsConfig {
    /// 현재 기본 커넥터 이름 ("gemma" | "deepl" | "claude" | ...)
    #[serde(default = "default_connector")]
    pub default: String,
    #[serde(default)]
    pub deepl: Option<DeeplConfig>,
    #[serde(default)]
    pub claude: Option<ClaudeConfig>,
}

fn default_connector() -> String { "gemma".into() }

#[derive(Debug, Clone, Deserialize)]
pub struct DeeplConfig {
    pub api_key: String,
    #[serde(default)]
    pub pro: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ClaudeConfig {
    pub api_key: String,
    #[serde(default = "default_claude_model")]
    pub model: String,
}

fn default_claude_model() -> String { "claude-opus-4-6".into() }

#[derive(Debug, Clone, Deserialize)]
pub struct GpuConfig {
    pub host: String,
    pub gpu_ids: Vec<u32>,
    pub poll_interval_s: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Defaults {
    pub source_lang: String,
    pub target_lang: String,
    pub workers: u32,
    pub context: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct JobsConfig {
    pub translate: TranslateJobConfig,
    pub sentry_i18n: SentryI18nConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TranslateJobConfig {
    pub cli: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SentryI18nConfig {
    pub cli: String,
    pub state_path: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UiConfig {
    pub refresh_ms: u64,
    pub history_max: usize,
}

impl Config {
    /// Nickel 파일을 `nickel export --format json` 으로 JSON 화해서 파싱.
    /// Nickel 바이너리가 없으면 동일 경로의 .json fallback 사용.
    pub fn load(path: &Path) -> Result<Self> {
        // 1) nickel 바이너리 시도
        if which("nickel") {
            let out = Command::new("nickel")
                .args(["export", "--format", "json"])
                .arg(path)
                .output()
                .context("failed to run nickel")?;
            if out.status.success() {
                return Ok(serde_json::from_slice(&out.stdout)
                    .context("parse nickel json output")?);
            }
        }
        // 2) JSON fallback (config.ncl.json 형태)
        let json_alt = path.with_extension("ncl.json");
        if json_alt.exists() {
            let data = std::fs::read(&json_alt)?;
            return Ok(serde_json::from_slice(&data)?);
        }
        // 3) 같은 이름 .json
        let json_same = path.with_extension("json");
        if json_same.exists() {
            let data = std::fs::read(&json_same)?;
            return Ok(serde_json::from_slice(&data)?);
        }
        anyhow::bail!(
            "nickel 바이너리와 fallback JSON 모두 없음. \
             `cargo install nickel-lang-cli` 하거나 {} 옆에 .json 두세요",
            path.display()
        )
    }
}

fn which(cmd: &str) -> bool {
    Command::new("sh").arg("-c").arg(format!("command -v {cmd}")).status()
        .map(|s| s.success()).unwrap_or(false)
}
