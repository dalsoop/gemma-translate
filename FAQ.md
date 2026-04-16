# FAQ / Troubleshooting

## Build / Install

### `cargo build` fails with `libc` linker errors
Use the musl target for a portable static binary:
```bash
cargo build --release --target x86_64-unknown-linux-musl
```

### `gemma-translate --help` → `GLIBC_2.39 not found`
You compiled on a newer host (e.g. Debian 13) than the target (Debian 12).
Fix: build with `--target x86_64-unknown-linux-musl`.

### `HF_TOKEN required` even though it's set
`sudo` strips the env. Use `sudo -E` or pass it inline:
```bash
sudo HF_TOKEN=$HF_TOKEN gemma-translate llama-install
```

### `llama-install --from-local`: `ModuleNotFoundError: transformers`
The CLI now auto-detects `venv/bin/python3` at these paths:
- `/root/venv/bin/python3`
- `/opt/translate-gemma/venv/bin/python3`
- `/opt/llama.cpp/venv/bin/python3`

If none has `transformers`, install it:
```bash
/root/venv/bin/pip install transformers gguf sentencepiece protobuf
```

## Runtime

### `cudaMalloc failed: out of memory` at model load
Another process is holding VRAM. Check:
```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```
Kill the rogue process or stop the other service (e.g. `ollama`, another `llama-server`).

### `exceeds the available context size (256 tokens)`
Fixed: `--ctx-size 65536` with `--parallel 16` = 4096 tokens per slot.
If you still see this, your systemd unit was generated before the fix — re-run `llama-up`.

### `chat template parsing error: User role must provide content as an iterable...`
The TranslateGemma Jinja template is strict. Fixed by `--no-jinja --chat-template chatml` in the generated unit.

### `libcudart.so.12: cannot open shared object file`
The llama-server binary needs CUDA runtime libs from a PyTorch venv. The generated systemd unit sets `LD_LIBRARY_PATH` automatically; if you run manually:
```bash
export LD_LIBRARY_PATH=/opt/llama.cpp/build/bin:/root/venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:...
```

### `shim` responds but backend 503 "Loading model"
Model load takes 1–3 minutes after `llama-up`. The unit's `ExecStartPre` waits up to 4 minutes for upstream `/health` before starting shim. If shim was started manually without the gate, just wait.

### Translation contains `---` separators
Upstream occasionally echoes the delimiter. Shim strips leading/trailing `---` automatically.

### Placeholders (`%s`, `{name}`) reordered
Short phrases like `"%s %d items"` may still swap. For longer UI strings with enough context it works. Add the exact msgid to glossary:
```bash
gemma-translate glossary add "%s %d items" "%d개의 %s 항목" --target ko
```

## Glossary

### Case/whitespace mismatch
Lookup normalizes case and trims whitespace:
`"Save"` = `"save"` = `"SAVE"` = `"Save "` — all hit the same entry.

### Glossary not applied after edit
Shim watches file mtime and reloads automatically; no restart needed.

### Bulk import format
Either flat:
```json
{ "Save": "저장", "Cancel": "취소" }
```
or per-language:
```json
{ "Save": { "ko": "저장", "ja": "保存" } }
```

## Backend-specific

### Why `llama.cpp` over `vLLM`?
- llama.cpp + BF16 GGUF "just works" with 4-way tensor split on consumer GPUs (RTX 3090 w/o NVLink).
- vLLM needs a separate vllm-compatible checkpoint (`Infomaniak-AI/vllm-translategemma-27b-it`) and AWQ pre-conversion for best throughput.
- AutoAWQ is deprecated; llm-compressor is the replacement but needs setup time.

### Why so many backend processes?
Only one `llama-server` process is running at a time when you use `llama-up` once with 4 GPUs (TP=4). The shim runs as a separate systemd unit so `/translate` stays up even if you restart llama-server.

## Operational

### Stop everything
```bash
sudo gemma-translate llama-down 8080    # shim + llama-server
```

### Service logs
```bash
journalctl -u llama-server-gemma@8080 -f
journalctl -u translate-llama@8080 -f
```

### Upgrade
Pull the repo, rebuild CLI, redeploy binary:
```bash
cd installer && cargo build --release --target x86_64-unknown-linux-musl
sudo install -m 0755 target/x86_64-unknown-linux-musl/release/gemma-translate /usr/local/bin/
# shim gets re-embedded on the next `llama-install`
```

### Zombie processes after stop
The install script uses `Restart=on-failure` with `StartLimitBurst=5` — a crashing service can't restart more than 5 times in 60s. Hard-stop:
```bash
sudo systemctl stop llama-server-gemma@8080 translate-llama@8080
```

## Security

### Expose publicly
Put it behind a reverse proxy (Traefik, Caddy, nginx) with TLS.
Set `TRANSLATE_API_KEY` before starting the instance.
Optionally add rate limiting at the proxy level.

### CORS
Shim doesn't set CORS headers. If calling from browser, terminate CORS at your reverse proxy.
