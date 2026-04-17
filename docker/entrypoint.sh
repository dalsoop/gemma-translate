#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL_PATH:-/model/model.gguf}"
PARALLEL="${PARALLEL:-16}"
CTX_SIZE="${CTX_SIZE:-65536}"
SHIM_PORT="${SHIM_PORT:-8080}"

# GPU count auto-detect
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
TENSOR_SPLIT=$(python3 -c "print(','.join(['1']*${GPU_COUNT:-1}))")

echo "━━━ gemma-translate ━━━"
echo "model:    $MODEL"
echo "GPUs:     $GPU_COUNT (tensor-split: $TENSOR_SPLIT)"
echo "parallel: $PARALLEL  ctx: $CTX_SIZE"
echo "port:     $SHIM_PORT"
echo "auth:     ${TRANSLATE_API_KEY:+enabled}"
echo

# Check model exists
if [[ ! -f "$MODEL" ]]; then
    if [[ -n "${HF_TOKEN:-}" ]]; then
        echo "[auto] 모델 다운로드 + 변환 (최초 1회, ~20분)..."
        pip3 install --quiet huggingface_hub[cli] hf_transfer transformers gguf sentencepiece protobuf
        export HF_HUB_ENABLE_HF_TRANSFER=1
        HF_DIR="/model/hf-cache"
        mkdir -p "$HF_DIR"
        hf download google/translategemma-27b-it --local-dir "$HF_DIR"

        # Convert to GGUF BF16
        if ! command -v convert_hf_to_gguf.py &>/dev/null; then
            pip3 install --quiet llama-cpp-python 2>/dev/null || true
            # Fallback: download convert script
            curl -fsSL "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py" -o /tmp/convert.py
            python3 /tmp/convert.py "$HF_DIR" --outtype bf16 --outfile "$MODEL"
        fi
        echo "[auto] 변환 완료: $MODEL"
    else
        echo "ERROR: model not found at $MODEL"
        echo "  Mount a .gguf file:  -v /path/to/model.gguf:/model/model.gguf:ro"
        echo "  Or set HF_TOKEN for auto-download:  -e HF_TOKEN=hf_xxx"
        exit 1
    fi
fi

# Start llama-server (background)
echo "[start] llama-server..."
llama-server \
    --model "$MODEL" \
    --host 127.0.0.1 --port 18080 \
    --n-gpu-layers 999 \
    --tensor-split "$TENSOR_SPLIT" \
    --parallel "$PARALLEL" \
    --cont-batching --flash-attn on \
    --no-jinja --chat-template chatml \
    --ctx-size "$CTX_SIZE" \
    &
LLAMA_PID=$!

# Wait for llama-server health
echo "[wait] llama-server loading..."
for i in $(seq 1 240); do
    if curl -sf http://127.0.0.1:18080/health >/dev/null 2>&1; then
        echo "[ready] llama-server up (${i}s)"
        break
    fi
    if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
        echo "[FATAL] llama-server died"
        exit 1
    fi
    sleep 1
done

# Start shim (foreground — docker uses this as main process)
echo "[start] shim on :${SHIM_PORT}"
exec python3 /app/shim.py
