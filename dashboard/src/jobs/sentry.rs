use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SentryStep {
    Extract,
    Scan,
    Translate,
    Build,
    Deploy,
    /// 전체 파이프라인 (extract → translate → build → deploy)
    Sync,
}

impl SentryStep {
    pub fn as_str(&self) -> &'static str {
        match self {
            SentryStep::Extract => "extract",
            SentryStep::Scan => "scan",
            SentryStep::Translate => "translate",
            SentryStep::Build => "build",
            SentryStep::Deploy => "deploy",
            SentryStep::Sync => "sync",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentryJob {
    pub step: SentryStep,
    pub cache_bust: bool,
    pub workers: u32,
    /// extract 시 source locales (기본 "ja,ru,ro,cs,fr,de,es")
    pub sources: Option<String>,
    pub limit: Option<u32>,
}
