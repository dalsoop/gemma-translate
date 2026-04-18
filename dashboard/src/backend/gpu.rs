//! GPU 통계 폴러. ssh <host> nvidia-smi --query-gpu ... 를 주기적으로 실행.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::process::Command;
use std::time::Duration;
use tokio::sync::watch;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GpuStat {
    pub index: u32,
    pub name: String,
    pub util_pct: u32,
    pub mem_used_mib: u32,
    pub mem_total_mib: u32,
    pub temp_c: u32,
}

#[derive(Debug, Clone, Default)]
pub struct GpuSnapshot {
    pub gpus: Vec<GpuStat>,
    pub error: Option<String>,
}

pub fn spawn_poller(
    host: String,
    interval: Duration,
) -> watch::Receiver<GpuSnapshot> {
    let (tx, rx) = watch::channel(GpuSnapshot::default());
    tokio::spawn(async move {
        loop {
            let snap = match fetch(&host).await {
                Ok(gpus) => GpuSnapshot { gpus, error: None },
                Err(e) => GpuSnapshot { gpus: vec![], error: Some(e.to_string()) },
            };
            let _ = tx.send(snap);
            tokio::time::sleep(interval).await;
        }
    });
    rx
}

async fn fetch(host: &str) -> Result<Vec<GpuStat>> {
    let host = host.to_string();
    let out = tokio::task::spawn_blocking(move || {
        Command::new("ssh")
            .arg("-o").arg("ConnectTimeout=3")
            .arg("-o").arg("BatchMode=yes")
            .arg(&host)
            .arg("nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits")
            .output()
    })
    .await??;
    if !out.status.success() {
        anyhow::bail!(
            "ssh nvidia-smi failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut gpus = Vec::new();
    for line in text.lines() {
        let parts: Vec<&str> = line.split(',').map(str::trim).collect();
        if parts.len() < 6 { continue; }
        gpus.push(GpuStat {
            index: parts[0].parse().unwrap_or(0),
            name: parts[1].to_string(),
            util_pct: parts[2].parse().unwrap_or(0),
            mem_used_mib: parts[3].parse().unwrap_or(0),
            mem_total_mib: parts[4].parse().unwrap_or(0),
            temp_c: parts[5].parse().unwrap_or(0),
        });
    }
    Ok(gpus)
}
