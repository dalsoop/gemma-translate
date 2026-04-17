# gemma-translate — one-command self-hosted translation server
#
# Usage:
#   # Build
#   docker build -t gemma-translate .
#
#   # Run (model must be pre-downloaded or mounted)
#   docker run --gpus all \
#     -v /path/to/translategemma-27b-bf16.gguf:/model/model.gguf:ro \
#     -v /path/to/glossary.json:/etc/gemma-translate/glossary.json:ro \
#     -p 8080:8080 \
#     gemma-translate
#
#   # Or with HF_TOKEN to auto-download + convert on first run:
#   docker run --gpus all \
#     -e HF_TOKEN=hf_xxx \
#     -v gemma-model:/model \
#     -p 8080:8080 \
#     gemma-translate

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS base

RUN apt-get update -qq && apt-get install -y -qq \
    python3 python3-pip curl git jq && \
    pip3 install --quiet fastapi httpx uvicorn pydantic && \
    rm -rf /var/lib/apt/lists/*

# llama.cpp server (pre-built from official release)
ARG LLAMA_CPP_VERSION=b5540
RUN curl -fsSL "https://github.com/ggerganov/llama.cpp/releases/download/${LLAMA_CPP_VERSION}/llama-${LLAMA_CPP_VERSION}-bin-ubuntu-x64.zip" \
      -o /tmp/llama.zip && \
    apt-get update -qq && apt-get install -y -qq unzip && \
    mkdir -p /opt/llama && cd /opt/llama && unzip -q /tmp/llama.zip && \
    cp build/bin/llama-server /usr/local/bin/ 2>/dev/null || \
    cp bin/llama-server /usr/local/bin/ 2>/dev/null || true && \
    rm -rf /tmp/llama.zip /var/lib/apt/lists/*

# Shim + entrypoint
COPY server/shim.py /app/shim.py
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Glossary default (empty — user mounts their own)
RUN mkdir -p /etc/gemma-translate && echo '{}' > /etc/gemma-translate/glossary.json

ENV LLAMA_URL=http://127.0.0.1:18080
ENV SHIM_PORT=8080
ENV MODEL_PATH=/model/model.gguf
ENV PARALLEL=16
ENV CTX_SIZE=65536
ENV TRANSLATE_API_KEY=""

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=180s \
  CMD curl -sf http://localhost:8080/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
