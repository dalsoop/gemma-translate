# Changelog

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
