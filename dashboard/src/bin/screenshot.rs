//! 스크린샷 생성기: TestBackend 로 UI 프레임을 렌더링 → SVG 저장.
//!
//! 사용:
//!   cargo run --release --bin screenshot -- main   --out docs/screenshot-main.svg
//!   cargo run --release --bin screenshot -- newjob --out docs/screenshot-newjob.svg

use anyhow::Result;
use chrono::Utc;
use ratatui::backend::TestBackend;
use ratatui::buffer::Buffer;
use ratatui::style::Color;
use ratatui::Terminal;
use std::fmt::Write;
use std::path::PathBuf;
use std::sync::Arc;
use translate_dashboard::{
    app::{App, Mode, NewJobField, NewJobType},
    backend::gpu::{GpuSnapshot, GpuStat},
    config::{Config, ConnectorsConfig, Defaults, GpuConfig, JobsConfig, SentryI18nConfig, TranslateJobConfig, UiConfig},
    jobs::{
        sentry::{SentryJob, SentryStep},
        translate::{TranslateInput, TranslateJob},
        Job, JobKind, JobStatus,
    },
    ui,
};

const W: u16 = 120;
const H: u16 = 36;
const CW: u32 = 9; // char width px
const CH: u32 = 18; // char height px

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "main".into());
    let mut out = PathBuf::from(format!("docs/screenshot-{mode}.svg"));
    while let Some(a) = args.next() {
        if a == "--out" {
            if let Some(v) = args.next() { out = PathBuf::from(v); }
        }
    }

    let cfg = Arc::new(mock_cfg());
    let mut app = App::new(cfg);
    populate(&mut app);

    if mode == "newjob" {
        app.mode = Mode::NewJob;
        app.new_job.focus = NewJobField::Main;
        app.new_job.job_type = NewJobType::Sentry;
        app.new_job.sentry_step = SentryStep::Sync;
    }

    let backend = TestBackend::new(W, H);
    let mut terminal = Terminal::new(backend)?;
    terminal.draw(|f| ui::draw(f, &app))?;
    let buf = terminal.backend().buffer().clone();

    if let Some(parent) = out.parent() { std::fs::create_dir_all(parent)?; }
    std::fs::write(&out, render_svg(&buf))?;
    println!("wrote {}", out.display());
    Ok(())
}

fn mock_cfg() -> Config {
    Config {
        api_endpoints: (8080..=8083).map(|p| format!("http://10.0.60.108:{p}")).collect(),
        gpu: GpuConfig { host: "192.168.2.60".into(), gpu_ids: vec![0,1,2,3], poll_interval_s: 5 },
        defaults: Defaults {
            source_lang: "en".into(), target_lang: "ko".into(),
            workers: 32, context: "concise UI text in Korean".into(),
        },
        jobs: JobsConfig {
            translate: TranslateJobConfig { cli: "/usr/local/bin/phs-translate".into() },
            sentry_i18n: SentryI18nConfig {
                cli: "/usr/local/bin/phs-sentry-i18n".into(),
                state_path: "/opt/sentry-i18n/state.json".into(),
            },
        },
        ui: UiConfig { refresh_ms: 250, history_max: 100 },
        connectors: ConnectorsConfig::default(),
    }
}

fn populate(app: &mut App) {
    app.gpu = GpuSnapshot {
        gpus: vec![
            GpuStat { index: 0, name: "RTX 3090".into(), util_pct: 42, mem_used_mib: 17410, mem_total_mib: 24576, temp_c: 68 },
            GpuStat { index: 1, name: "RTX 3090".into(), util_pct: 67, mem_used_mib: 17320, mem_total_mib: 24576, temp_c: 72 },
            GpuStat { index: 2, name: "RTX 3090".into(), util_pct: 95, mem_used_mib: 20840, mem_total_mib: 24576, temp_c: 79 },
            GpuStat { index: 3, name: "RTX 3090".into(), util_pct: 34, mem_used_mib: 18150, mem_total_mib: 24576, temp_c: 65 },
        ],
        error: None,
    };

    let now = Utc::now();
    let mut j1 = Job::new(JobKind::Translate(TranslateJob {
        source_lang: "en".into(), target_lang: "ko".into(),
        context: Some("UI".into()),
        input: TranslateInput::Text("Set Up User Feedback".into()),
    }));
    j1.status = JobStatus::Running; j1.progress = 0.74; j1.started_at = Some(now);

    let mut j2 = Job::new(JobKind::SentryI18n(SentryJob {
        step: SentryStep::Translate, cache_bust: true, workers: 32,
        sources: None, limit: None,
    }));
    j2.status = JobStatus::Running; j2.progress = 0.31; j2.started_at = Some(now);

    let mut jq = Job::new(JobKind::Translate(TranslateJob {
        source_lang: "en".into(), target_lang: "ja".into(),
        context: None,
        input: TranslateInput::File { path: "strings.json".into(), out: None },
    }));
    jq.status = JobStatus::Queued;
    app.active = vec![j1.clone(), j2.clone()];
    app.queue.push_back(jq);

    let mut h1 = Job::new(JobKind::SentryI18n(SentryJob {
        step: SentryStep::Deploy, cache_bust: true, workers: 32, sources: None, limit: None,
    }));
    h1.status = JobStatus::Done; h1.progress = 1.0;
    h1.started_at = Some(now); h1.finished_at = Some(now);

    let mut h2 = Job::new(JobKind::Translate(TranslateJob {
        source_lang: "en".into(), target_lang: "ko".into(),
        context: None,
        input: TranslateInput::List(vec!["Save".into(), "Cancel".into(), "Delete".into()]),
    }));
    h2.status = JobStatus::Done; h2.progress = 1.0;
    h2.started_at = Some(now); h2.finished_at = Some(now);

    let mut h3 = Job::new(JobKind::Translate(TranslateJob {
        source_lang: "en".into(), target_lang: "ko".into(),
        context: None, input: TranslateInput::Text("retry".into()),
    }));
    h3.status = JobStatus::Failed;
    h3.started_at = Some(now); h3.finished_at = Some(now);

    app.history = vec![h1, h2, h3];

    app.push_log("[a1b2c3d4] start: Sentry i18n: translate".into());
    app.push_log("[a1b2c3d4] translating 1127 entries  workers=16".into());
    app.push_log("[a1b2c3d4]   352/1127 (31%)  0.9/s  eta=861s".into());
    app.push_log("[f0e1d2c3]   → 사용자 피드백 설정".into());
    app.push_log("[1234abcd] deploy: ko.c80f77eb.. → ko.9ab31f02.. (rehash ok)".into());
    app.push_log("[9ab31f02] done".into());
}

fn color_hex(c: Color, default: &str) -> String {
    match c {
        Color::Reset => default.into(),
        Color::Black => "#1e1e1e".into(),
        Color::Red => "#f44747".into(),
        Color::Green => "#6cc24a".into(),
        Color::Yellow => "#e5c07b".into(),
        Color::Blue => "#569cd6".into(),
        Color::Magenta => "#c678dd".into(),
        Color::Cyan => "#56b6c2".into(),
        Color::Gray => "#abb2bf".into(),
        Color::DarkGray => "#5c6370".into(),
        Color::LightRed => "#e06c75".into(),
        Color::LightGreen => "#98c379".into(),
        Color::LightYellow => "#e5c07b".into(),
        Color::LightBlue => "#61afef".into(),
        Color::LightMagenta => "#c678dd".into(),
        Color::LightCyan => "#56b6c2".into(),
        Color::White => "#d4d4d4".into(),
        Color::Rgb(r,g,b) => format!("#{r:02x}{g:02x}{b:02x}"),
        Color::Indexed(_) => default.into(),
    }
}

fn render_svg(buf: &Buffer) -> String {
    let width = W as u32 * CW;
    let height = H as u32 * CH;
    let mut svg = String::new();
    writeln!(svg, r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" font-family="ui-monospace, Menlo, Consolas, monospace" font-size="14">"##).unwrap();
    writeln!(svg, r##"<rect width="100%" height="100%" fill="#1e1e1e"/>"##).unwrap();

    // Backgrounds
    for y in 0..H {
        for x in 0..W {
            if let Some(cell) = buf.cell((x, y)) {
                let bg = color_hex(cell.bg, "#1e1e1e");
                if bg != "#1e1e1e" {
                    let px = x as u32 * CW;
                    let py = y as u32 * CH;
                    writeln!(svg, r##"<rect x="{px}" y="{py}" width="{CW}" height="{CH}" fill="{bg}"/>"##).unwrap();
                }
            }
        }
    }
    // Foreground text (per-row, grouping by color for brevity)
    for y in 0..H {
        let py = (y as u32 + 1) * CH - 4;
        let mut run_x: Option<u16> = None;
        let mut run_fg = String::new();
        let mut run_text = String::new();
        let flush = |svg: &mut String, run_x: &mut Option<u16>, run_fg: &mut String, run_text: &mut String| {
            if let Some(sx) = *run_x {
                if !run_text.trim().is_empty() || run_text.contains(|c: char| !c.is_whitespace()) {
                    let px = sx as u32 * CW;
                    let esc = run_text.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;");
                    writeln!(svg, r##"<text x="{px}" y="{py}" fill="{run_fg}" xml:space="preserve">{esc}</text>"##).unwrap();
                }
                *run_x = None;
                run_text.clear();
            }
        };
        for x in 0..W {
            let cell = buf.cell((x, y));
            let (sym, fg) = match cell {
                Some(c) => (c.symbol().to_string(), color_hex(c.fg, "#d4d4d4")),
                None => (" ".into(), "#d4d4d4".into()),
            };
            if run_x.is_none() {
                run_x = Some(x);
                run_fg = fg.clone();
            } else if run_fg != fg {
                flush(&mut svg, &mut run_x, &mut run_fg, &mut run_text);
                run_x = Some(x);
                run_fg = fg;
            }
            run_text.push_str(&sym);
        }
        flush(&mut svg, &mut run_x, &mut run_fg, &mut run_text);
    }

    svg.push_str("</svg>\n");
    svg
}
