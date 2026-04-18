//! Smoke test — config 로드 → 커넥터 등록 → translate 1회 + GPU 1회 폴 확인.
//!
//! 사용:
//!   cargo run --release --bin smoke -- config.ncl

use anyhow::Result;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use translate_dashboard::{
    backend::gpu,
    config::Config,
    connectors::{claude::ClaudeConnector, deepl::DeeplConnector, gemma::GemmaConnector, BoxConnector, Registry},
};

#[tokio::main]
async fn main() -> Result<()> {
    let cfg_path = std::env::args().nth(1).map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("config.ncl"));
    let cfg = Arc::new(Config::load(&cfg_path)?);
    println!("[config] endpoints={}  default={}",
        cfg.api_endpoints.len(), cfg.connectors.default);

    // registry
    let mut reg = Registry::default();
    reg.register("gemma", Arc::new(GemmaConnector::new(cfg.api_endpoints.clone())) as BoxConnector);
    if let Some(d) = cfg.connectors.deepl.as_ref() {
        reg.register("deepl", Arc::new(DeeplConnector::new(d.api_key.clone(), d.pro)) as BoxConnector);
    }
    if let Some(c) = cfg.connectors.claude.as_ref() {
        reg.register("claude", Arc::new(ClaudeConnector::new(c.api_key.clone(), c.model.clone())) as BoxConnector);
    }
    println!("[connectors] registered: {:?}", reg.names());

    // health
    for name in reg.names() {
        let c = reg.get(&name).unwrap().clone();
        match c.health().await {
            Ok(s) => println!("  health({name}): {s}"),
            Err(e) => println!("  health({name}): ERR {e}"),
        }
    }

    // translate round-trip
    let c = reg.get(&cfg.connectors.default).unwrap().clone();
    let sample = "Hello, how are you today?";
    println!("[translate] en→ko  sample={sample:?}  via={}", c.name());
    match c.translate(sample, "en", "ko", None).await {
        Ok(r) => println!("  → {:?}  ({:.2}s)", r.translation, r.elapsed_s),
        Err(e) => println!("  FAILED: {e}"),
    }

    // gpu
    println!("[gpu] polling {host}…", host = cfg.gpu.host);
    let mut rx = gpu::spawn_poller(cfg.gpu.host.clone(), Duration::from_secs(cfg.gpu.poll_interval_s));
    // wait first snapshot
    rx.changed().await.ok();
    let snap = rx.borrow().clone();
    if let Some(err) = snap.error {
        println!("  ERR: {err}");
    } else {
        for g in snap.gpus {
            println!("  GPU{} util={}% mem={}/{}MiB temp={}°C",
                g.index, g.util_pct, g.mem_used_mib, g.mem_total_mib, g.temp_c);
        }
    }

    println!("[smoke] done");
    Ok(())
}
