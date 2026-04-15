#!/usr/bin/env bash
# TranslateGemma 27B-IT 서버 설치 스크립트
#
# 사전:
#   - NVIDIA 드라이버 + CUDA runtime
#   - HF_TOKEN (Gemma 게이트 통과된 HuggingFace 토큰)
#
# 사용:
#   sudo HF_TOKEN=hf_xxx bash install.sh
#     → venv + server.py + up/down CLI + 모델(27b-it) 다운로드
#   sudo /opt/translate-gemma/up.sh 0 8080   # GPU 0 에 인스턴스 기동
#   sudo /opt/translate-gemma/up.sh 1 8081
#   sudo /opt/translate-gemma/down.sh 8081   # 중지
set -euo pipefail

ROOT="${ROOT:-/opt/translate-gemma}"
VENV="$ROOT/venv"
SERVER_PY="$ROOT/server.py"
MODEL_DIR="$ROOT/model"
REPO="google/translategemma-27b-it"

[[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN 필요"; exit 1; }

echo "[1/4] 시스템 패키지"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip git curl jq

echo "[2/4] venv + python 패키지"
mkdir -p "$ROOT"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip wheel --quiet
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 --quiet
pip install -r "$(dirname "$0")/requirements.txt" --quiet

echo "[3/4] 모델 다운로드: $REPO (~54GB)"
export HF_HUB_ENABLE_HF_TRANSFER=1
hf auth login --token "$HF_TOKEN" --add-to-git-credential 2>&1 | tail -1 || true
hf download "$REPO" --local-dir "$MODEL_DIR"

echo "[4/4] server.py + up/down 스크립트 배치"
install -m 0755 "$(dirname "$0")/server.py" "$SERVER_PY"

cat > "$ROOT/up.sh" <<'EOS'
#!/usr/bin/env bash
# up.sh <gpu_idx> <port>
set -euo pipefail
gpu="${1:?GPU index}"; port="${2:?port}"
ROOT=/opt/translate-gemma
unit="/etc/systemd/system/translate-gemma@${port}.service"
cat > "$unit" <<UNIT
[Unit]
Description=TranslateGemma 27B (GPU $gpu, port $port)
After=network.target
[Service]
Type=simple
WorkingDirectory=$ROOT
Environment="CUDA_VISIBLE_DEVICES=$gpu"
Environment="TRANSLATE_PORT=$port"
Environment="MODEL_DIR=$ROOT/model"
Environment="MODEL_NAME=27b-it"
Environment="QUANT=nf4"
ExecStart=$ROOT/venv/bin/python $ROOT/server.py
Restart=on-failure
RestartSec=10
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now "translate-gemma@${port}.service"
echo "started translate-gemma@${port} (GPU $gpu)"
echo "check: curl http://localhost:${port}/info"
EOS

cat > "$ROOT/down.sh" <<'EOS'
#!/usr/bin/env bash
# down.sh <port>
port="${1:?port}"
systemctl disable --now "translate-gemma@${port}.service" 2>/dev/null || true
rm -f "/etc/systemd/system/translate-gemma@${port}.service"
systemctl daemon-reload
echo "stopped translate-gemma@${port}"
EOS

chmod +x "$ROOT/up.sh" "$ROOT/down.sh"

cat <<EOF

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 설치 완료. 다음:

   sudo $ROOT/up.sh 0 8080     # GPU 0 → port 8080
   sudo $ROOT/up.sh 1 8081     # GPU 1 → port 8081
   sudo $ROOT/up.sh 2 8082
   sudo $ROOT/up.sh 3 8083

   curl http://localhost:8080/info
   sudo $ROOT/down.sh 8080
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EOF
