//! Job 실행자. 큐에서 job 꺼내 status 갱신하며 실행.
//!
//! - 진행률 파싱: "N/M (P%)" 패턴을 stderr/stdout 에서 감지해 job.progress 업데이트
//! - 취소: CancelToken 으로 외부에서 종료 요청 가능 (실행 중이면 child kill, 대기 중이면 drop)
//! - 로그: 각 job 의 모든 stdout/stderr 라인 → log 채널

use crate::config::Config;
use crate::connectors::{claude::ClaudeConnector, deepl::DeeplConnector, gemma::GemmaConnector, BoxConnector, Registry};
use crate::jobs::{sentry::SentryStep, translate::TranslateInput, Job, JobKind, JobStatus};
use anyhow::{anyhow, Result};
use chrono::Utc;
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::{mpsc, Mutex, Notify, RwLock};
use tokio::task::JoinHandle;
use uuid::Uuid;

pub struct WorkerHandle {
    pub log_rx: mpsc::UnboundedReceiver<String>,
    pub job_tx: mpsc::UnboundedSender<Job>,
    pub state: Arc<Mutex<Vec<Job>>>,
    pub history: Arc<Mutex<Vec<Job>>>,
    pub cancel: Arc<Mutex<HashMap<Uuid, Arc<Notify>>>>,
    /// 등록된 커넥터 목록 + 현재 활성화된 커넥터 이름 (스위치 가능)
    pub registry: Arc<Registry>,
    pub active_connector: Arc<RwLock<String>>,
    pub _task: JoinHandle<()>,
}

impl WorkerHandle {
    pub async fn cancel_job(&self, id: Uuid) -> bool {
        let cancels = self.cancel.lock().await;
        if let Some(n) = cancels.get(&id) {
            n.notify_one();
            true
        } else {
            false
        }
    }

    /// 활성 커넥터를 런타임에 교체. 이미 실행중인 job 은 현 커넥터 유지.
    pub async fn set_connector(&self, name: &str) -> bool {
        if self.registry.get(name).is_some() {
            *self.active_connector.write().await = name.to_string();
            true
        } else {
            false
        }
    }

    pub async fn active_connector_name(&self) -> String {
        self.active_connector.read().await.clone()
    }
}

pub fn spawn_worker(cfg: Arc<Config>) -> WorkerHandle {
    let (log_tx, log_rx) = mpsc::unbounded_channel::<String>();
    let (job_tx, mut job_rx) = mpsc::unbounded_channel::<Job>();
    let state = Arc::new(Mutex::new(Vec::<Job>::new()));
    let history = Arc::new(Mutex::new(Vec::<Job>::new()));
    let cancel: Arc<Mutex<HashMap<Uuid, Arc<Notify>>>> = Arc::new(Mutex::new(HashMap::new()));

    // 커넥터 등록
    let mut registry = Registry::default();
    registry.register("gemma",
        Arc::new(GemmaConnector::new(cfg.api_endpoints.clone())) as BoxConnector);
    if let Some(d) = cfg.connectors.deepl.as_ref() {
        registry.register("deepl",
            Arc::new(DeeplConnector::new(d.api_key.clone(), d.pro)) as BoxConnector);
    }
    if let Some(c) = cfg.connectors.claude.as_ref() {
        registry.register("claude",
            Arc::new(ClaudeConnector::new(c.api_key.clone(), c.model.clone())) as BoxConnector);
    }
    let registry = Arc::new(registry);
    let active = cfg.connectors.default.clone();
    // 존재하지 않는 기본값이면 gemma 로 fallback
    let active = if registry.get(&active).is_some() { active } else { "gemma".into() };
    let active_connector = Arc::new(RwLock::new(active));

    // 초기 로드
    {
        let h = history.clone();
        let cfg = cfg.clone();
        tokio::spawn(async move {
            if let Some(items) = load_history(&cfg) {
                *h.lock().await = items;
            }
        });
    }

    let state_c = state.clone();
    let history_c = history.clone();
    let cancel_c = cancel.clone();
    let registry_c = registry.clone();
    let active_c = active_connector.clone();
    let task = tokio::spawn(async move {
        while let Some(mut job) = job_rx.recv().await {
            job.status = JobStatus::Running;
            job.started_at = Some(Utc::now());
            let notify = Arc::new(Notify::new());
            cancel_c.lock().await.insert(job.id, notify.clone());

            state_c.lock().await.push(job.clone());
            let _ = log_tx.send(format!("[{}] start: {}", short(job.id), job.kind.title()));

            let conn_name = active_c.read().await.clone();
            let conn = registry_c.get(&conn_name).cloned().ok_or_else(|| anyhow!("no connector: {conn_name}"));
            let result = match conn {
                Ok(c) => run_job(&mut job, &cfg, &c, &log_tx, notify.clone(), &state_c).await,
                Err(e) => Err(e),
            };
            job.finished_at = Some(Utc::now());
            cancel_c.lock().await.remove(&job.id);

            match result {
                Ok(true) => {
                    job.status = JobStatus::Done;
                    job.progress = 1.0;
                    let _ = log_tx.send(format!("[{}] done", short(job.id)));
                }
                Ok(false) => {
                    job.status = JobStatus::Cancelled;
                    let _ = log_tx.send(format!("[{}] cancelled", short(job.id)));
                }
                Err(e) => {
                    job.status = JobStatus::Failed;
                    job.message = e.to_string();
                    let _ = log_tx.send(format!("[{}] FAILED: {e}", short(job.id)));
                }
            }

            {
                let mut s = state_c.lock().await;
                s.retain(|j| j.id != job.id);
            }
            {
                let mut h = history_c.lock().await;
                h.insert(0, job);
                if h.len() > cfg.ui.history_max { h.truncate(cfg.ui.history_max); }
                let _ = save_history(&cfg, &h);
            }
        }
    });

    WorkerHandle {
        log_rx, job_tx, state, history, cancel,
        registry, active_connector,
        _task: task,
    }
}

fn short(id: Uuid) -> String {
    id.to_string().chars().take(8).collect()
}

/// `Ok(true)` = done, `Ok(false)` = cancelled, `Err(_)` = failed
async fn run_job(
    job: &mut Job,
    cfg: &Config,
    connector: &BoxConnector,
    log: &mpsc::UnboundedSender<String>,
    cancel: Arc<Notify>,
    shared: &Arc<Mutex<Vec<Job>>>,
) -> Result<bool> {
    match &job.kind.clone() {
        JobKind::Translate(t) => match &t.input {
            TranslateInput::Text(text) => {
                tokio::select! {
                    _ = cancel.notified() => Ok(false),
                    r = connector.translate(text, &t.source_lang, &t.target_lang, t.context.as_deref()) => {
                        let r = r?;
                        job.message = r.translation.clone();
                        update_progress(shared, job.id, 1.0).await;
                        let _ = log.send(format!("  [{}] → {}", connector.name(), r.translation));
                        Ok(true)
                    }
                }
            }
            TranslateInput::List(items) => {
                for (i, item) in items.iter().enumerate() {
                    tokio::select! {
                        _ = cancel.notified() => return Ok(false),
                        r = connector.translate(item, &t.source_lang, &t.target_lang, t.context.as_deref()) => {
                            let r = r?;
                            let p = (i as f32 + 1.0) / items.len() as f32;
                            job.progress = p;
                            update_progress(shared, job.id, p).await;
                            let _ = log.send(format!("  [{}/{} {}] {} → {}", i + 1, items.len(), connector.name(), item, r.translation));
                        }
                    }
                }
                Ok(true)
            }
            TranslateInput::File { path, out } => {
                let mut cmd = Command::new(&cfg.jobs.translate.cli);
                cmd.arg("-i").arg(path)
                    .arg("-s").arg(&t.source_lang)
                    .arg("-t").arg(&t.target_lang)
                    .arg("-w").arg(cfg.defaults.workers.to_string());
                if let Some(o) = out { cmd.arg("-o").arg(o); }
                if let Some(c) = &t.context { cmd.arg("-c").arg(c); }
                cmd.env("TRANSLATE_API", cfg.api_endpoints.join(","));
                stream_cmd(cmd, job.id, log, cancel, shared).await
            }
        },
        JobKind::SentryI18n(s) => {
            let mut cmd = Command::new(&cfg.jobs.sentry_i18n.cli);
            cmd.arg("--workers").arg(s.workers.to_string());
            if s.cache_bust { cmd.arg("--bust"); }
            if let Some(src) = &s.sources { cmd.arg("--sources").arg(src); }
            if let Some(l) = s.limit { cmd.arg("--limit").arg(l.to_string()); }
            match s.step {
                SentryStep::Extract => { cmd.arg("extract"); }
                SentryStep::Scan => { cmd.arg("scan"); }
                SentryStep::Translate => { cmd.arg("translate"); }
                SentryStep::Build => { cmd.arg("build"); }
                SentryStep::Deploy => { cmd.arg("deploy"); }
                SentryStep::Sync => { cmd.arg("sync"); }
            }
            cmd.env("TRANSLATE_API", cfg.api_endpoints.join(","));
            stream_cmd(cmd, job.id, log, cancel, shared).await
        }
    }
}

async fn update_progress(shared: &Arc<Mutex<Vec<Job>>>, id: Uuid, p: f32) {
    let mut s = shared.lock().await;
    if let Some(j) = s.iter_mut().find(|j| j.id == id) {
        j.progress = p;
    }
}

fn parse_progress(line: &str) -> Option<f32> {
    // 매칭 대상: "  352/1127 (31%)" or "31%"
    if let Some(pct_pos) = line.find('%') {
        let prefix = &line[..pct_pos];
        let digits: String = prefix.chars().rev().take_while(|c| c.is_ascii_digit()).collect();
        if !digits.is_empty() {
            if let Ok(n) = digits.chars().rev().collect::<String>().parse::<u32>() {
                if n <= 100 { return Some(n as f32 / 100.0); }
            }
        }
    }
    // "N/M" fallback
    if let Some((n, rest)) = line.split_once('/') {
        let n_clean: String = n.chars().rev().take_while(|c| c.is_ascii_digit()).collect();
        let m_clean: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if let (Ok(a), Ok(b)) = (
            n_clean.chars().rev().collect::<String>().parse::<u32>(),
            m_clean.parse::<u32>(),
        ) {
            if b > 0 { return Some((a as f32 / b as f32).min(1.0)); }
        }
    }
    None
}

async fn stream_cmd(
    mut cmd: Command,
    id: Uuid,
    log: &mpsc::UnboundedSender<String>,
    cancel: Arc<Notify>,
    shared: &Arc<Mutex<Vec<Job>>>,
) -> Result<bool> {
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn()?;
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    let log1 = log.clone();
    let log2 = log.clone();
    let shared1 = shared.clone();
    let shared2 = shared.clone();

    let t1 = tokio::spawn(async move {
        let mut r = BufReader::new(stdout).lines();
        while let Ok(Some(line)) = r.next_line().await {
            if let Some(p) = parse_progress(&line) { update_progress(&shared1, id, p).await; }
            let _ = log1.send(format!("[{}] {}", short(id), line));
        }
    });
    let t2 = tokio::spawn(async move {
        // stderr 는 \r 로 구분된 tqdm 진행바가 옴 → 라인 여러 개가 한 줄에 섞여있을 수 있음
        let mut r = BufReader::new(stderr);
        loop {
            let mut buf = Vec::new();
            let mut one = [0u8; 1];
            use tokio::io::AsyncReadExt;
            // read until \n or \r
            let n = match r.read(&mut one).await { Ok(0) => break, Ok(n) => n, Err(_) => break };
            if n == 0 { break; }
            if one[0] == b'\n' || one[0] == b'\r' {
                if buf.is_empty() { continue; }
                let line = String::from_utf8_lossy(&buf).to_string();
                if let Some(p) = parse_progress(&line) { update_progress(&shared2, id, p).await; }
                let _ = log2.send(format!("[{}] {}", short(id), line));
                continue;
            }
            buf.push(one[0]);
            // accumulate up to reasonable length
            loop {
                let n = match r.read(&mut one).await { Ok(0) => break, Ok(n) => n, Err(_) => break };
                if n == 0 { break; }
                if one[0] == b'\n' || one[0] == b'\r' {
                    let line = String::from_utf8_lossy(&buf).to_string();
                    if let Some(p) = parse_progress(&line) { update_progress(&shared2, id, p).await; }
                    let _ = log2.send(format!("[{}] {}", short(id), line));
                    buf.clear();
                    break;
                }
                buf.push(one[0]);
                if buf.len() > 2048 {
                    let line = String::from_utf8_lossy(&buf).to_string();
                    let _ = log2.send(format!("[{}] {}", short(id), line));
                    buf.clear();
                    break;
                }
            }
        }
    });

    let cancelled = tokio::select! {
        _ = cancel.notified() => {
            let _ = child.kill().await;
            true
        }
        status = child.wait() => {
            let _ = tokio::join!(t1, t2);
            let status = status?;
            if !status.success() {
                anyhow::bail!("exit code: {}", status.code().unwrap_or(-1));
            }
            false
        }
    };
    Ok(!cancelled)
}

// ─── 히스토리 영속화 ───

fn history_path(_cfg: &Config) -> std::path::PathBuf {
    let base = std::env::var("XDG_DATA_HOME")
        .ok()
        .or_else(|| std::env::var("HOME").ok().map(|h| format!("{h}/.local/share")))
        .unwrap_or_else(|| "/tmp".into());
    std::path::PathBuf::from(base).join("translate-dashboard").join("history.json")
}

fn save_history(cfg: &Config, items: &[Job]) -> std::io::Result<()> {
    let p = history_path(cfg);
    if let Some(parent) = p.parent() { std::fs::create_dir_all(parent)?; }
    let json = serde_json::to_string_pretty(items).unwrap_or_else(|_| "[]".into());
    std::fs::write(p, json)
}

fn load_history(cfg: &Config) -> Option<Vec<Job>> {
    let p = history_path(cfg);
    let data = std::fs::read(p).ok()?;
    serde_json::from_slice(&data).ok()
}
