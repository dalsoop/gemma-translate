# Changelog

## [0.2.0] — 2026-04-18

### Added
- **`status`** command — GPU VRAM, 포트별 backend/shim health, glossary 현황 일괄 체크
- **`restart`** command — 좀비 shim 프로세스 정리 + llama-server/shim 일괄 재기동
- **`translate`** command — 내장 병렬 JSON 번역. resume 지원, 실패 키 보고, atomic write
- **`llama-down --keep-units`** — 유닛 파일 보존 (restart로 재기동 가능)
- `llama-up --replicas` 병렬 기동 — 유닛 일괄 생성 → daemon-reload 1회 → 동시 start

### Fixed
- batch save 비원자적 쓰기 → `.tmp` + `rename()` atomic write
- corrupted JSON resume 시 패닉 → `.corrupted` 백업 후 처음부터 재시작
- `status`/`translate` 포트 8080-8083 하드코딩 → systemd 유닛 자동 탐지
- `pkill -f translate-shim.py` 전체 kill → 포트별 `SHIM_PORT` 환경변수 매칭
- `cuda_ld_path()` python3.11 하드코딩 → `python3.*` glob 자동 탐지
- ExecStart `--model` 경로 공백 시 깨짐 → 인용 처리
- `VllmMeta` dead code + unused `Serialize` import 제거

## [0.1.0] — 2026-04-16

First public release.

### Added
- **Rust installer CLI** (`gemma-translate`) — clap-based, three backends:
  - `install` / `up` / `down` / `info` / `list` — transformers + NF4
  - `llama-install` / `llama-up` / `llama-down` — llama.cpp + BF16 GGUF (4-GPU tensor split)
  - `vllm-install` / `vllm-up` / `vllm-down` — vLLM
- **Glossary** (`glossary add/remove/list/import/export`) — standardized translations bypass the model. Case/whitespace-normalized lookup.
- **FastAPI shim** exposing `/translate`, `/health`, `/info` with:
  - Optional `TRANSLATE_API_KEY` auth (`X-API-Key` / `Authorization: Bearer`)
  - BCP-47 language code validation
  - `max_new_tokens` hard cap (2048)
  - Placeholder preservation rules in prompt (`%s`, `{name}`, etc.)
- **systemd units** auto-generated per port:
  - `llama-server-gemma@<port>.service` (backend)
  - `translate-llama@<port>.service` (shim, gated on upstream health)
  - `StartLimitBurst=5 / 60s` prevents crash loops
- **Python universal CLI** (`cli/translate`) — single text / list / JSON / gettext `.po` / file modes; round-robin over multiple endpoints.

### Fixed / Hardened (during development)
- llama.cpp `--ctx-size 4096` with `--parallel 16` → only 256 tokens per slot.
  Raised to `65536` (4096/slot).
- `--from-local` conversion failed because system `python3` lacks `transformers`.
  Now auto-detects a venv and installs deps into it.
- Shim's `upstream ready` race: `ExecStartPre` loops `/health` for up to 4 minutes before the shim binds.
- Glossary mtime-watched reload — no restart on edit.
- Placeholder reordering — instruction added to prompt.
- Long-text truncation at 500 chars — `max_new_tokens` default raised to 1024.
- Response with leading/trailing `---` — stripped.

### Ships with
- `Infomaniak-AI/vllm-translategemma-27b-it` support (vLLM-compatible variant).
- GGUF built locally from `/root/models/translategemma-27b-it` via `convert_hf_to_gguf.py --outtype bf16`.

### Tried and abandoned
- **AWQ quantization**: AutoAWQ deprecated (incompatible with transformers 5.x).
  llm-compressor `oneshot()` fails on TranslateGemma's vision_tower —
  scans layers but never runs actual quantization. No working AWQ path
  for Gemma3-based models as of 2026-04.
- **vLLM TP>1 in LXC**: NCCL IPC blocked by LXC GPU passthrough.
  Use bare-metal or VM for multi-GPU tensor-parallel vLLM.
