//! gemma-translate — TranslateGemma 27B-IT 설치/관리 CLI (Rust)
//!
//! 서브커맨드:
//!   install                     venv + 모델(27b-it) 다운로드 + server.py 배치
//!   up <gpu> <port>             해당 GPU 에 인스턴스 기동 (systemd unit 생성)
//!   down <port>                 인스턴스 중지 + unit 제거
//!   list                        설치 상태/실행중 인스턴스
//!   info <port>                 /info 엔드포인트 조회 + 정돈 출력
//!
//! 환경변수:
//!   HF_TOKEN   install 시 필요 (Gemma 라이선스 통과된 토큰)
//!   ROOT       설치 루트 (기본 /opt/translate-gemma)
//!   QUANT      nf4 (default) | int8 | none

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const SERVER_PY: &str = include_str!("../../server/server.py");
const REQUIREMENTS_TXT: &str = include_str!("../../server/requirements.txt");
const MODEL_REPO: &str = "google/translategemma-27b-it";
const DEFAULT_ROOT: &str = "/opt/translate-gemma";

#[derive(Parser)]
#[command(version, about = "gemma-translate installer/manager")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// venv + 모델 다운로드 + server.py 배치
    Install,
    /// 인스턴스 기동: up <gpu> <port>
    Up { gpu: u32, port: u16 },
    /// 인스턴스 중지: down <port>
    Down { port: u16 },
    /// 설치/실행 상태
    List,
    /// /info 조회
    Info { #[arg(default_value_t = 8080)] port: u16 },
}

fn root() -> PathBuf {
    std::env::var("ROOT")
        .unwrap_or_else(|_| DEFAULT_ROOT.to_string())
        .into()
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Install => install(),
        Cmd::Up { gpu, port } => up(gpu, port),
        Cmd::Down { port } => down(port),
        Cmd::List => list(),
        Cmd::Info { port } => info(port),
    }
}

// ─── subcommands ───

fn install() -> Result<()> {
    let root = root();
    let venv = root.join("venv");
    let model_dir = root.join("model");
    let server_py = root.join("server.py");
    let req_txt = root.join("requirements.txt");

    let hf_token = std::env::var("HF_TOKEN").context("HF_TOKEN 환경변수 필요")?;

    ensure_root()?;
    println!("[1/4] apt 패키지 설치");
    sh_env(&[("DEBIAN_FRONTEND", "noninteractive")],
        &["apt-get", "update", "-qq"])?;
    sh_env(&[("DEBIAN_FRONTEND", "noninteractive")],
        &["apt-get", "install", "-y", "-qq",
          "python3", "python3-venv", "python3-pip", "git", "curl", "jq"])?;

    println!("[2/4] venv + Python 패키지 ({}~3분)", 2);
    fs::create_dir_all(&root)?;
    fs::write(&req_txt, REQUIREMENTS_TXT)?;
    if !venv.join("bin/python").exists() {
        sh(&["python3", "-m", "venv", venv.to_str().unwrap()])?;
    }
    let pip = venv.join("bin/pip");
    sh(&[pip.to_str().unwrap(), "install", "--upgrade", "pip", "wheel", "--quiet"])?;
    sh(&[pip.to_str().unwrap(), "install",
         "torch==2.5.1",
         "--index-url", "https://download.pytorch.org/whl/cu124",
         "--quiet"])?;
    sh(&[pip.to_str().unwrap(), "install", "-r", req_txt.to_str().unwrap(), "--quiet"])?;

    println!("[3/4] 모델 다운로드: {MODEL_REPO}");
    let hf = venv.join("bin/hf");
    sh_env(
        &[("HF_HUB_ENABLE_HF_TRANSFER", "1")],
        &[hf.to_str().unwrap(), "auth", "login",
          "--token", &hf_token, "--add-to-git-credential"],
    ).ok(); // 이미 로그인된 경우 무시
    sh_env(
        &[("HF_HUB_ENABLE_HF_TRANSFER", "1")],
        &[hf.to_str().unwrap(), "download", MODEL_REPO,
          "--local-dir", model_dir.to_str().unwrap()],
    )?;

    println!("[4/4] server.py 배치");
    fs::write(&server_py, SERVER_PY)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&server_py, fs::Permissions::from_mode(0o755))?;
    }

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(" 설치 완료.");
    println!("   gemma-translate up 0 8080");
    println!("   gemma-translate up 1 8081");
    println!("   gemma-translate info 8080");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Ok(())
}

fn up(gpu: u32, port: u16) -> Result<()> {
    ensure_root()?;
    let root = root();
    let model_dir = root.join("model");
    let venv_py = root.join("venv/bin/python");
    let server_py = root.join("server.py");
    if !model_dir.exists() { bail!("모델 없음: {} — 먼저 `gemma-translate install`", model_dir.display()); }
    if !server_py.exists() { bail!("server.py 없음: {} — 먼저 install", server_py.display()); }

    let quant = std::env::var("QUANT").unwrap_or_else(|_| "nf4".into());
    let unit_path = format!("/etc/systemd/system/translate-gemma@{port}.service");
    let unit = format!(r#"[Unit]
Description=TranslateGemma 27B (GPU {gpu}, port {port})
After=network.target

[Service]
Type=simple
WorkingDirectory={root}
Environment="CUDA_VISIBLE_DEVICES={gpu}"
Environment="TRANSLATE_PORT={port}"
Environment="MODEL_DIR={model}"
Environment="MODEL_NAME=27b-it"
Environment="QUANT={quant}"
ExecStart={py} {srv}
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
"#,
        root = root.display(),
        model = model_dir.display(),
        py = venv_py.display(),
        srv = server_py.display(),
    );
    fs::write(&unit_path, unit)?;
    sh(&["systemctl", "daemon-reload"])?;
    sh(&["systemctl", "enable", "--now", &format!("translate-gemma@{port}.service")])?;
    println!("기동: translate-gemma@{port} (GPU {gpu})");
    println!("확인: gemma-translate info {port}");
    Ok(())
}

fn down(port: u16) -> Result<()> {
    ensure_root()?;
    let unit = format!("translate-gemma@{port}.service");
    let _ = Command::new("systemctl").args(["disable", "--now", &unit]).status();
    let _ = fs::remove_file(format!("/etc/systemd/system/{unit}"));
    sh(&["systemctl", "daemon-reload"])?;
    println!("중지: {unit}");
    Ok(())
}

fn list() -> Result<()> {
    let root = root();
    println!("루트: {}", root.display());
    println!("모델 설치:   {}", root.join("model").exists());
    println!("server.py:   {}", root.join("server.py").exists());
    println!("venv:        {}", root.join("venv/bin/python").exists());
    println!();
    println!("── 실행중 인스턴스 ──");
    let out = Command::new("systemctl")
        .args(["list-units", "translate-gemma@*", "--no-legend"])
        .output()?;
    let text = String::from_utf8_lossy(&out.stdout);
    if text.trim().is_empty() {
        println!("  (없음)");
    } else {
        for line in text.lines() {
            println!("  {line}");
        }
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
struct Info {
    model: String,
    quant: String,
    vram_gb: f32,
    load_time_s: f32,
    cuda_visible_devices: String,
    port: u16,
}

fn info(port: u16) -> Result<()> {
    let url = format!("http://localhost:{port}/info");
    let resp: Info = reqwest::blocking::get(&url)
        .with_context(|| format!("connect {url}"))?
        .json()?;
    println!("port:     {}", resp.port);
    println!("model:    {}", resp.model);
    println!("quant:    {}", resp.quant);
    println!("VRAM:     {:.2} GB", resp.vram_gb);
    println!("load:     {:.1} s", resp.load_time_s);
    println!("CUDA:     {}", resp.cuda_visible_devices);
    Ok(())
}

// ─── helpers ───

fn ensure_root() -> Result<()> {
    #[cfg(unix)]
    if unsafe { libc::getuid() } != 0 {
        bail!("root 권한 필요 (sudo)");
    }
    Ok(())
}

fn sh(args: &[&str]) -> Result<()> {
    sh_env(&[], args)
}

fn sh_env(envs: &[(&str, &str)], args: &[&str]) -> Result<()> {
    let mut cmd = Command::new(args[0]);
    cmd.args(&args[1..]).stdout(Stdio::inherit()).stderr(Stdio::inherit());
    for (k, v) in envs { cmd.env(k, v); }
    let status = cmd.status().with_context(|| format!("spawn {}", args[0]))?;
    if !status.success() {
        bail!("{} failed: {}", args[0], status);
    }
    Ok(())
}

fn _unused(_p: &Path) {}
