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
use serde::{Deserialize, Serialize};
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
    /// venv + 모델 다운로드 + server.py 배치 (transformers 백엔드)
    Install,
    /// 인스턴스 기동 (transformers 백엔드): up <gpu> <port>
    Up { gpu: u32, port: u16 },
    /// 인스턴스 중지: down <port>
    Down { port: u16 },
    /// 설치/실행 상태
    List,
    /// /info 조회 (transformers 서버만)
    Info { #[arg(default_value_t = 8080)] port: u16 },

    /// llama.cpp 백엔드: GGUF 준비 (HF 다운 OR 로컬 safetensors 변환) + shim 배치
    LlamaInstall {
        /// HuggingFace 저장소 (기본: bullerwins/translategemma-27b-it-GGUF)
        #[arg(long, default_value = "bullerwins/translategemma-27b-it-GGUF")]
        repo: String,
        /// 양자화 필터 (기본 Q4_K_M, 사전 변환본 다운로드 시)
        #[arg(long, default_value = "*Q4_K_M*")]
        quant: String,
        /// 로컬 HuggingFace 디렉토리(safetensors)에서 BF16 GGUF 로 변환 (HF 다운 스킵)
        /// 예: --from-local /root/models/translategemma-27b-it
        #[arg(long)]
        from_local: Option<String>,
    },
    /// llama.cpp 인스턴스 기동: llama-up <gpu_list> <port>  (예: "0,1,2,3")
    LlamaUp { gpus: String, port: u16 },
    /// llama.cpp 인스턴스 중지
    LlamaDown { port: u16 },

    /// vLLM 백엔드: vllm 설치 + 모델 다운로드 (기본 Infomaniak vLLM-호환 버전)
    VllmInstall {
        /// HuggingFace 저장소 (vLLM 호환 모델)
        #[arg(long, default_value = "Infomaniak-AI/vllm-translategemma-27b-it")]
        repo: String,
        /// 양자화 옵션 (none|awq|bitsandbytes|fp8)
        #[arg(long, default_value = "none")]
        quantization: String,
    },
    /// vLLM 인스턴스 기동: vllm-up <gpu_list> <port>  (예: "0" 또는 "0,1,2,3")
    VllmUp { gpus: String, port: u16 },
    /// vLLM 인스턴스 중지
    VllmDown { port: u16 },
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
        Cmd::LlamaInstall { repo, quant, from_local } => llama_install(&repo, &quant, from_local.as_deref()),
        Cmd::LlamaUp { gpus, port } => llama_up(&gpus, port),
        Cmd::LlamaDown { port } => llama_down(port),
        Cmd::VllmInstall { repo, quantization } => vllm_install(&repo, &quantization),
        Cmd::VllmUp { gpus, port } => vllm_up(&gpus, port),
        Cmd::VllmDown { port } => vllm_down(port),
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

// ─── llama.cpp 백엔드 ───

const LLAMA_ROOT: &str = "/opt/llama.cpp";
const LLAMA_MODEL_DIR: &str = "/opt/llama.cpp/models/local/translategemma-27b";
const LLAMA_SERVER_BIN: &str = "/usr/local/bin/llama-server";
const LLAMA_SHIM_PY: &str = "/opt/llama.cpp/translate-shim.py";
const LLAMA_SHIM_UNIT_PREFIX: &str = "translate-llama";

// shim: phs-translate 의 /translate POST → llama-server /completion (raw prompt) 변환
const SHIM_PY_CONTENTS: &str = r#"#!/usr/bin/env python3
"""llama-server (TranslateGemma) → /translate 호환 shim.

llama-server 의 chat template 이 까다로워 --no-jinja 로 띄우고,
shim 이 raw prompt 를 직접 빌드해 /completion 으로 호출한다."""
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

LLAMA_URL = os.environ.get("LLAMA_URL", "http://127.0.0.1:18080")
MODEL_NAME = os.environ.get("MODEL_NAME", "translategemma-27b")
SHIM_PORT = int(os.environ.get("SHIM_PORT", "8080"))

app = FastAPI()
client = httpx.AsyncClient(timeout=120)


def build_prompt(text: str, src: str, tgt: str) -> str:
    return (
        "<start_of_turn>user\n"
        f"Translate the following text from {src} to {tgt}.\n\n{text}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


class Req(BaseModel):
    text: str
    source_lang_code: str = "en"
    target_lang_code: str = "ko"
    max_new_tokens: int = 512


@app.get("/health")
async def health():
    try:
        r = await client.get(f"{LLAMA_URL}/health")
        return {"ok": r.status_code == 200, "backend": "llama.cpp"}
    except Exception:
        return {"ok": False}


@app.get("/info")
async def info():
    return {"model": MODEL_NAME, "backend": "llama.cpp", "upstream": LLAMA_URL, "port": SHIM_PORT}


@app.post("/translate")
async def translate(r: Req):
    if not r.text.strip():
        raise HTTPException(400, "empty text")
    prompt = build_prompt(r.text, r.source_lang_code, r.target_lang_code)
    payload = {
        "prompt": prompt,
        "n_predict": r.max_new_tokens,
        "temperature": 0,
        "stop": ["<end_of_turn>", "<start_of_turn>", "</s>"],
        "cache_prompt": False,
    }
    try:
        resp = await client.post(f"{LLAMA_URL}/completion", json=payload)
        data = resp.json()
        if resp.status_code >= 400:
            raise HTTPException(500, f"llama: {data}")
        return {"translation": data.get("content", "").strip()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"shim error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SHIM_PORT)
"#;

const LLAMA_GGUF_PATH: &str = "/opt/llama.cpp/models/translategemma-27b-bf16.gguf";

// CUDA 런타임 라이브러리 경로 (PyTorch venv 가 있으면 그 안의 nvidia 패키지에서 가져옴)
fn cuda_ld_path() -> String {
    let candidates = [
        "/root/venv/lib/python3.11/site-packages/nvidia",
        "/opt/translate-gemma/venv/lib/python3.11/site-packages/nvidia",
    ];
    let nvidia_root = candidates.iter().find(|p| Path::new(p).exists())
        .map(|s| s.to_string()).unwrap_or_default();
    let mut paths = vec![format!("{LLAMA_ROOT}/build/bin")];
    for sub in ["cuda_runtime", "cublas", "cuda_nvrtc", "nccl",
                "cufft", "curand", "cusolver", "cusparse", "cudnn"] {
        paths.push(format!("{nvidia_root}/{sub}/lib"));
    }
    paths.join(":")
}

fn llama_install(repo: &str, quant_pattern: &str, from_local: Option<&str>) -> Result<()> {
    ensure_root()?;
    if !Path::new(LLAMA_SERVER_BIN).exists() {
        bail!("{} 없음. llama.cpp 가 설치되어 있어야 함 (`/opt/llama.cpp/build/bin/llama-server`)",
              LLAMA_SERVER_BIN);
    }

    println!("[1/3] shim 의존성 설치 (fastapi, httpx, uvicorn)");
    sh(&["pip", "install", "--quiet", "--break-system-packages",
         "fastapi", "httpx", "uvicorn",
         "huggingface_hub[cli]", "hf_transfer",
         "gguf", "sentencepiece", "protobuf"])
        .or_else(|_| sh(&["pip", "install", "--quiet",
                          "fastapi", "httpx", "uvicorn",
                          "huggingface_hub[cli]", "hf_transfer",
                          "gguf", "sentencepiece", "protobuf"]))?;

    fs::create_dir_all(LLAMA_MODEL_DIR)?;

    if let Some(src) = from_local {
        // 로컬 safetensors → BF16 GGUF 변환
        let convert_py = format!("{LLAMA_ROOT}/convert_hf_to_gguf.py");
        if !Path::new(&convert_py).exists() {
            bail!("convert_hf_to_gguf.py 없음: {}.\n  llama.cpp 소스 트리도 같은 경로에 있어야 함.", convert_py);
        }
        if !Path::new(src).exists() {
            bail!("로컬 모델 디렉토리 없음: {src}");
        }
        println!("[2/3] 로컬 safetensors → BF16 GGUF 변환");
        println!("  src: {src}");
        println!("  out: {LLAMA_GGUF_PATH}");
        sh(&["python3", &convert_py, src,
             "--outtype", "bf16",
             "--outfile", LLAMA_GGUF_PATH])?;
    } else {
        let hf_token = std::env::var("HF_TOKEN")
            .context("HF_TOKEN 필요 (Gemma 게이트, 또는 --from-local 사용)")?;
        let hf_bin = ["/usr/local/bin/hf", "/usr/bin/hf",
                      "/usr/local/bin/huggingface-cli", "/usr/bin/huggingface-cli"]
            .iter().find(|p| Path::new(p).exists()).map(|s| s.to_string())
            .unwrap_or_else(|| "hf".into());
        println!("[2/3] GGUF 다운로드: {repo}  filter={quant_pattern}");
        sh_env(
            &[("HF_HUB_ENABLE_HF_TRANSFER", "1"), ("HF_TOKEN", &hf_token)],
            &[&hf_bin, "download", repo,
              "--include", quant_pattern,
              "--local-dir", LLAMA_MODEL_DIR],
        )?;
    }

    println!("[3/3] /translate shim 배치: {LLAMA_SHIM_PY}");
    fs::write(LLAMA_SHIM_PY, SHIM_PY_CONTENTS)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(LLAMA_SHIM_PY, fs::Permissions::from_mode(0o755))?;
    }

    println!("\n설치 완료. 다음:");
    println!("  gemma-translate llama-up 0,1,2,3 8080   # 4 GPU 분산 + shim 자동 기동");
    Ok(())
}

fn pick_gguf() -> Result<String> {
    // 우선순위: 변환 산출물 → repo 다운로드 dir 안의 가장 큰 .gguf
    if Path::new(LLAMA_GGUF_PATH).exists() {
        return Ok(LLAMA_GGUF_PATH.to_string());
    }
    let dir = Path::new(LLAMA_MODEL_DIR);
    if !dir.exists() { bail!("모델 없음 — `gemma-translate llama-install` 먼저"); }
    let mut best: Option<(std::path::PathBuf, u64)> = None;
    for e in fs::read_dir(dir)? {
        let e = e?;
        let p = e.path();
        if p.extension().and_then(|s| s.to_str()) == Some("gguf") {
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if name.contains("-of-") && !name.contains("00001-of-") { continue; }
            let sz = e.metadata()?.len();
            if best.as_ref().map(|(_, s)| sz > *s).unwrap_or(true) {
                best = Some((p, sz));
            }
        }
    }
    best.ok_or_else(|| anyhow::anyhow!("GGUF 파일 없음: {LLAMA_MODEL_DIR}"))
        .map(|(p, _)| p.to_string_lossy().to_string())
}

fn llama_up(gpus: &str, port: u16) -> Result<()> {
    ensure_root()?;
    if !Path::new(LLAMA_SHIM_PY).exists() {
        bail!("shim 없음 — 먼저 `gemma-translate llama-install`");
    }
    let gguf = pick_gguf()?;
    let llama_port = 18000 + port as u32; // shim 이 바라보는 내부 포트
    let tp_size = gpus.split(',').count();
    let tensor_split = vec!["1"; tp_size].join(",");
    let ld_path = cuda_ld_path();

    let llama_unit = format!("/etc/systemd/system/llama-server-gemma@{port}.service");
    let llama = format!(r#"[Unit]
Description=llama-server (TranslateGemma, GPUs={gpus}, upstream :{llama_port})
After=network.target

[Service]
Type=simple
Environment="CUDA_VISIBLE_DEVICES={gpus}"
Environment="LD_LIBRARY_PATH={ld_path}"
ExecStart={server} --model {gguf} --host 127.0.0.1 --port {llama_port} \
  --n-gpu-layers 999 --tensor-split {tensor_split} \
  --parallel 16 --cont-batching --flash-attn on \
  --no-jinja --chat-template chatml --ctx-size 4096
Restart=on-failure
RestartSec=10
LimitMEMLOCK=infinity

[Install]
WantedBy=multi-user.target
"#, server = LLAMA_SERVER_BIN);
    fs::write(&llama_unit, llama)?;

    // shim 유닛
    let shim_unit = format!("/etc/systemd/system/{LLAMA_SHIM_UNIT_PREFIX}@{port}.service");
    let shim = format!(r#"[Unit]
Description=TranslateGemma shim (/translate → llama-server :{llama_port})
After=llama-server-gemma@{port}.service
Requires=llama-server-gemma@{port}.service

[Service]
Type=simple
Environment="LLAMA_URL=http://127.0.0.1:{llama_port}"
Environment="SHIM_PORT={port}"
Environment="MODEL_NAME=translategemma-27b"
ExecStart=/usr/bin/python3 {LLAMA_SHIM_PY}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
"#);
    fs::write(&shim_unit, shim)?;

    sh(&["systemctl", "daemon-reload"])?;
    sh(&["systemctl", "enable", "--now", &format!("llama-server-gemma@{port}.service")])?;
    sh(&["systemctl", "enable", "--now", &format!("{LLAMA_SHIM_UNIT_PREFIX}@{port}.service")])?;
    println!("기동: llama-server-gemma@{port} (GPUs={gpus}, upstream :{llama_port}) + shim :{port}");
    println!("확인: curl http://localhost:{port}/health");
    Ok(())
}

fn llama_down(port: u16) -> Result<()> {
    ensure_root()?;
    for name in [
        format!("{LLAMA_SHIM_UNIT_PREFIX}@{port}.service"),
        format!("llama-server-gemma@{port}.service"),
    ] {
        let _ = Command::new("systemctl").args(["disable", "--now", &name]).status();
        let _ = fs::remove_file(format!("/etc/systemd/system/{name}"));
    }
    sh(&["systemctl", "daemon-reload"])?;
    println!("중지: port {port} (llama + shim)");
    Ok(())
}

// ─── vLLM 백엔드 ───

const VLLM_MODEL_ROOT: &str = "/opt/vllm/models";
const VLLM_META_PATH: &str = "/etc/gemma-translate/vllm.json";
const VLLM_VENV: &str = "/opt/translate-gemma/venv"; // transformers 와 공유 (이미 설치된 경우)

#[derive(Serialize, Deserialize)]
#[allow(dead_code)]
struct VllmMeta {
    model_dir: String,
    repo: String,
    quantization: String,
}

fn vllm_install(repo: &str, quantization: &str) -> Result<()> {
    ensure_root()?;
    let hf_token = std::env::var("HF_TOKEN").context("HF_TOKEN 필요 (Gemma 게이트)")?;

    println!("[1/3] vLLM + 의존성 설치");
    // 기존 venv 있으면 재사용, 없으면 만들기
    if !Path::new(VLLM_VENV).exists() {
        fs::create_dir_all("/opt/translate-gemma")?;
        sh(&["python3", "-m", "venv", VLLM_VENV])?;
    }
    let pip = format!("{VLLM_VENV}/bin/pip");
    sh(&[&pip, "install", "--upgrade", "pip", "--quiet"])?;
    sh(&[&pip, "install", "--quiet",
         "vllm", "huggingface_hub[cli]", "hf_transfer"])?;

    println!("[2/3] 모델 다운로드: {repo}");
    let safe_name = repo.replace('/', "-");
    let model_dir = format!("{VLLM_MODEL_ROOT}/{safe_name}");
    fs::create_dir_all(&model_dir)?;
    let hf = format!("{VLLM_VENV}/bin/hf");
    sh_env(
        &[("HF_HUB_ENABLE_HF_TRANSFER", "1"), ("HF_TOKEN", &hf_token)],
        &[&hf, "auth", "login", "--token", &hf_token, "--add-to-git-credential"],
    ).ok();
    sh_env(
        &[("HF_HUB_ENABLE_HF_TRANSFER", "1"), ("HF_TOKEN", &hf_token)],
        &[&hf, "download", repo, "--local-dir", &model_dir],
    )?;

    println!("[3/3] 메타데이터 저장: {VLLM_META_PATH}");
    fs::create_dir_all("/etc/gemma-translate")?;
    let meta = serde_json::json!({
        "model_dir": &model_dir,
        "repo": repo,
        "quantization": quantization,
    });
    fs::write(VLLM_META_PATH, serde_json::to_string_pretty(&meta)?)?;

    println!("\n설치 완료. 다음:");
    println!("  gemma-translate vllm-up 0 8080          # GPU 0 단독");
    println!("  gemma-translate vllm-up 0,1,2,3 8080    # TP=4 전체 분산");
    Ok(())
}

fn vllm_meta() -> Result<(String, String)> {
    let data = fs::read_to_string(VLLM_META_PATH)
        .context("vllm meta 없음 — 먼저 `vllm-install`")?;
    let v: serde_json::Value = serde_json::from_str(&data)?;
    let model_dir = v["model_dir"].as_str().unwrap_or("").to_string();
    let quant = v["quantization"].as_str().unwrap_or("none").to_string();
    Ok((model_dir, quant))
}

fn vllm_up(gpus: &str, port: u16) -> Result<()> {
    ensure_root()?;
    let (model_dir, quantization) = vllm_meta()?;
    let tp_size = gpus.split(',').count();
    let unit_path = format!("/etc/systemd/system/translate-vllm@{port}.service");

    let mut exec = format!(
        "{venv}/bin/vllm serve {model} --host 0.0.0.0 --port {port} \
         --tensor-parallel-size {tp}",
        venv = VLLM_VENV, model = model_dir, tp = tp_size,
    );
    if quantization != "none" && !quantization.is_empty() {
        exec.push_str(&format!(" --quantization {quantization}"));
    }

    let unit = format!(r#"[Unit]
Description=vLLM TranslateGemma (GPUs={gpus}, TP={tp_size}, port={port})
After=network.target

[Service]
Type=simple
Environment="CUDA_VISIBLE_DEVICES={gpus}"
ExecStart={exec}
Restart=on-failure
RestartSec=10
LimitMEMLOCK=infinity

[Install]
WantedBy=multi-user.target
"#);
    fs::write(&unit_path, unit)?;
    sh(&["systemctl", "daemon-reload"])?;
    sh(&["systemctl", "enable", "--now", &format!("translate-vllm@{port}.service")])?;
    println!("기동: translate-vllm@{port}  GPUs={gpus}  TP={tp_size}");
    println!("확인: curl http://localhost:{port}/v1/models");
    Ok(())
}

fn vllm_down(port: u16) -> Result<()> {
    ensure_root()?;
    let unit = format!("translate-vllm@{port}.service");
    let _ = Command::new("systemctl").args(["disable", "--now", &unit]).status();
    let _ = fs::remove_file(format!("/etc/systemd/system/{unit}"));
    sh(&["systemctl", "daemon-reload"])?;
    println!("중지: {unit}");
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
