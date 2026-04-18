use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub mod sentry;
pub mod translate;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobStatus {
    Queued,
    Running,
    Done,
    Failed,
    Cancelled,
}

impl JobStatus {
    pub fn symbol(&self) -> &'static str {
        match self {
            JobStatus::Queued => "…",
            JobStatus::Running => "●",
            JobStatus::Done => "✓",
            JobStatus::Failed => "✗",
            JobStatus::Cancelled => "◌",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JobKind {
    /// 단일/리스트/파일 번역
    Translate(translate::TranslateJob),
    /// Sentry i18n 파이프라인 (extract → translate → build → deploy)
    SentryI18n(sentry::SentryJob),
}

impl JobKind {
    pub fn title(&self) -> String {
        match self {
            JobKind::Translate(t) => format!(
                "Translate [{}→{}] {}",
                t.source_lang,
                t.target_lang,
                t.display_label()
            ),
            JobKind::SentryI18n(s) => format!("Sentry i18n: {}", s.step.as_str()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    pub id: Uuid,
    pub kind: JobKind,
    pub status: JobStatus,
    pub progress: f32, // 0.0..=1.0
    pub message: String,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub finished_at: Option<DateTime<Utc>>,
}

impl Job {
    pub fn new(kind: JobKind) -> Self {
        Self {
            id: Uuid::new_v4(),
            kind,
            status: JobStatus::Queued,
            progress: 0.0,
            message: String::new(),
            created_at: Utc::now(),
            started_at: None,
            finished_at: None,
        }
    }
}
