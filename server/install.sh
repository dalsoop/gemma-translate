#!/usr/bin/env bash
# TranslateGemma 서버 설치 스크립트 (Debian/Ubuntu, NVIDIA GPU 환경)
#
# 사전 준비:
#   - NVIDIA 드라이버 + CUDA runtime (GPU 컨테이너 내부에 libcuda 보이면 OK)
#   - HF_TOKEN 환경변수 (Gemma 게이트 통과된 HuggingFace 토큰)
#
# 사용:
#   export HF_TOKEN=hf_xxx
#   bash install.sh           # 27B-it (기본, ~54GB 다운로드)
#   MODEL=4b-it bash install.sh  # 4B-it (빠름, ~8GB)
set -euo pipefail

MODEL="${MODEL:-27b-it}"
REPO="google/translategemma-${MODEL}"
MODEL_DIR="${MODEL_DIR:-/opt/translate-gemma/model}"
VENV="${VENV:-/opt/translate-gemma/venv}"
CUDA_IDX="${CUDA_IDX:-0}"
PORT="${PORT:-8080}"

[[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN 필요"; exit 1; }

echo "[1/4] 시스템 패키지 설치"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip git curl

echo "[2/4] Python venv + 패키지"
mkdir -p "$(dirname "$VENV")"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip wheel --quiet
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 --quiet
pip install -r "$(dirname "$0")/requirements.txt" --quiet

echo "[3/4] 모델 다운로드: $REPO"
mkdir -p "$(dirname "$MODEL_DIR")"
export HF_HUB_ENABLE_HF_TRANSFER=1
hf auth login --token "$HF_TOKEN" --add-to-git-credential 2>&1 | tail -1 || true
hf download "$REPO" --local-dir "$MODEL_DIR"

echo "[4/4] systemd 유닛 설치"
install -D -m 0755 "$(dirname "$0")/server.py" /opt/translate-gemma/server.py
cat > /etc/systemd/system/translate-gemma.service <<EOF
[Unit]
Description=TranslateGemma Server (GPU $CUDA_IDX, port $PORT)
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/translate-gemma
Environment="CUDA_VISIBLE_DEVICES=$CUDA_IDX"
Environment="TRANSLATE_PORT=$PORT"
Environment="MODEL_DIR=$MODEL_DIR"
ExecStart=$VENV/bin/python /opt/translate-gemma/server.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now translate-gemma.service

echo "완료. 로그: journalctl -u translate-gemma -f"
echo "테스트: curl http://localhost:$PORT/health"
