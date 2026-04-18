use translate_dashboard::app::{App, Focus, Mode, NewJobField, NewJobType};
use translate_dashboard::backend::{self, gpu, worker};
use translate_dashboard::config::Config;
use translate_dashboard::jobs::sentry::SentryStep;
use translate_dashboard::ui;
use anyhow::{Context, Result};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::watch;
use tui_input::backend::crossterm::EventHandler;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    let cfg_path = std::env::args().nth(1).map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("config.ncl"));
    let cfg: Arc<Config> = Arc::new(
        Config::load(&cfg_path).with_context(|| format!("load {}", cfg_path.display()))?,
    );

    let gpu_rx = gpu::spawn_poller(
        cfg.gpu.host.clone(),
        Duration::from_secs(cfg.gpu.poll_interval_s),
    );
    let mut worker = worker::spawn_worker(cfg.clone());

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let res = run_app(&mut terminal, cfg, gpu_rx, &mut worker).await;

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
    terminal.show_cursor()?;

    if let Err(e) = res {
        eprintln!("error: {e:?}");
        std::process::exit(1);
    }
    Ok(())
}

async fn run_app<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    cfg: Arc<Config>,
    mut gpu_rx: watch::Receiver<backend::gpu::GpuSnapshot>,
    worker: &mut worker::WorkerHandle,
) -> Result<()> {
    let mut app = App::new(cfg.clone());
    let tick = Duration::from_millis(cfg.ui.refresh_ms);
    loop {
        if gpu_rx.has_changed().unwrap_or(false) {
            app.gpu = gpu_rx.borrow_and_update().clone();
        }
        while let Ok(line) = worker.log_rx.try_recv() {
            app.push_log(line);
        }
        {
            let active = worker.state.lock().await;
            app.active = active.clone();
        }
        {
            let hist = worker.history.lock().await;
            app.history = hist.clone();
        }
        app.active_connector = worker.active_connector_name().await;
        app.available_connectors = worker.registry.names();
        if app.selected_active >= app.active.len() && !app.active.is_empty() {
            app.selected_active = app.active.len() - 1;
        }

        terminal.draw(|f| ui::draw(f, &app))?;

        if event::poll(tick)? {
            if let Event::Key(key) = event::read()? {
                handle_key(&mut app, key, worker).await;
                if app.should_quit { break; }
            }
        }
    }
    Ok(())
}

async fn handle_key(app: &mut App, key: KeyEvent, worker: &worker::WorkerHandle) {
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        app.should_quit = true;
        return;
    }
    match app.mode {
        Mode::Normal => handle_normal(app, key, worker).await,
        Mode::NewJob => handle_new_job_key(app, key, worker),
        Mode::Help => {
            if matches!(key.code, KeyCode::Esc | KeyCode::Char('?') | KeyCode::Char('q')) {
                app.mode = Mode::Normal;
            }
        }
    }
}

async fn handle_normal(app: &mut App, key: KeyEvent, worker: &worker::WorkerHandle) {
    match key.code {
        KeyCode::Char('q') => app.should_quit = true,
        KeyCode::Char('n') => app.mode = Mode::NewJob,
        KeyCode::Char('?') => app.mode = Mode::Help,
        KeyCode::Up => {
            if app.selected_active > 0 { app.selected_active -= 1; }
        }
        KeyCode::Down => {
            if !app.active.is_empty() && app.selected_active + 1 < app.active.len() {
                app.selected_active += 1;
            }
        }
        KeyCode::Char('x') => {
            if let Some(j) = app.active.get(app.selected_active) {
                let id = j.id;
                let _ = worker.cancel_job(id).await;
                app.push_log(format!("cancel signal → {}", &id.to_string()[..8]));
            }
        }
        KeyCode::Char('c') => {
            // 다음 커넥터로 순환
            let list = &app.available_connectors;
            if !list.is_empty() {
                let cur = list.iter().position(|n| *n == app.active_connector).unwrap_or(0);
                let next = (cur + 1) % list.len();
                let name = list[next].clone();
                if worker.set_connector(&name).await {
                    app.push_log(format!("connector switched → {name}"));
                }
            }
        }
        KeyCode::Char('b') => {
            // 간이 벤치마크: 10건 전송 → 소요 시간 로그
            app.push_log("benchmark: 10건 전송 중…".into());
            let connector = {
                let name = worker.active_connector_name().await;
                worker.registry.get(&name).cloned()
            };
            if let Some(c) = connector {
                let c = c.clone();
                let t0 = std::time::Instant::now();
                let mut ok = 0u32;
                let mut fail = 0u32;
                for i in 0..10 {
                    match c.translate(
                        &format!("Benchmark test {i}"), "en", "ko", None
                    ).await {
                        Ok(_) => ok += 1,
                        Err(_) => fail += 1,
                    }
                }
                let elapsed = t0.elapsed().as_secs_f32();
                let rps = 10.0 / elapsed;
                app.push_log(format!(
                    "benchmark: {ok}/10 ok, {fail} fail, {elapsed:.1}s ({rps:.1} req/s)"
                ));
            }
        }
        KeyCode::Tab => {
            app.focus = match app.focus {
                Focus::Gpu => Focus::Jobs,
                Focus::Jobs => Focus::History,
                Focus::History => Focus::Log,
                Focus::Log => Focus::Gpu,
            };
        }
        _ => {}
    }
}

fn handle_new_job_key(app: &mut App, key: KeyEvent, worker: &worker::WorkerHandle) {
    // 글로벌 키 (편집 가능 필드여도 우선 처리)
    match key.code {
        KeyCode::Esc => { app.mode = Mode::Normal; return; }
        KeyCode::Tab => {
            if key.modifiers.contains(KeyModifiers::SHIFT) {
                app.new_job.prev_field();
            } else {
                app.new_job.next_field();
            }
            return;
        }
        KeyCode::BackTab => { app.new_job.prev_field(); return; }
        KeyCode::Enter => {
            if let Some(job) = app.new_job.to_job() {
                let _ = worker.job_tx.send(job);
                app.mode = Mode::Normal;
            }
            return;
        }
        _ => {}
    }

    // 선택 필드 (편집 불가)
    match (app.new_job.focus, app.new_job.job_type, key.code) {
        (NewJobField::Type, _, KeyCode::Char('<') | KeyCode::Left) |
        (NewJobField::Type, _, KeyCode::Char('>') | KeyCode::Right) => {
            app.new_job.job_type = match app.new_job.job_type {
                NewJobType::Translate => NewJobType::Sentry,
                NewJobType::Sentry => NewJobType::Translate,
            };
            return;
        }
        (NewJobField::Main, NewJobType::Sentry, KeyCode::Char(c)) if matches!(c, '1'..='6') => {
            app.new_job.sentry_step = match c {
                '1' => SentryStep::Extract,
                '2' => SentryStep::Scan,
                '3' => SentryStep::Translate,
                '4' => SentryStep::Build,
                '5' => SentryStep::Deploy,
                _   => SentryStep::Sync,
            };
            return;
        }
        (NewJobField::Extra, NewJobType::Sentry, KeyCode::Char('b')) => {
            app.new_job.cache_bust = !app.new_job.cache_bust;
            return;
        }
        _ => {}
    }

    // 나머지는 편집 가능 필드로 라우팅
    if let Some(input) = app.new_job.editable_input() {
        input.handle_event(&Event::Key(key));
    }
}
