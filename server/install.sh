#!/usr/bin/env bash
# TranslateGemma 서버 설치 스크립트 (다중 모델 지원)
#
# 사전:
#   - NVIDIA 드라이버 + CUDA runtime
#   - HF_TOKEN 환경변수 (Gemma 게이트 통과된 HuggingFace 토큰)
#
# 사용:
#   # 공통 venv + server.py + manage.sh 설치 (처음 1회)
#   sudo HF_TOKEN=hf_xxx bash install.sh
#
#   # 모델 다운로드
#   sudo HF_TOKEN=hf_xxx /opt/translate-gemma/manage.sh download 27b-it
#   sudo HF_TOKEN=hf_xxx /opt/translate-gemma/manage.sh download 4b-it
#
#   # GPU 마다 인스턴스 기동 (systemd)
#   sudo /opt/translate-gemma/manage.sh up 27b-it 0 8080
#   sudo /opt/translate-gemma/manage.sh up 27b-it 1 8081
#   sudo /opt/translate-gemma/manage.sh up 4b-it  2 8082

set -euo pipefail

ROOT="${ROOT:-/opt/translate-gemma}"
VENV="$ROOT/venv"
SERVER_PY="$ROOT/server.py"
MANAGE_SH="$ROOT/manage.sh"
MODEL_ROOT="$ROOT/models"

[[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN 필요"; exit 1; }

echo "[1/3] 시스템 패키지"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip git curl jq

echo "[2/3] venv + python 패키지"
mkdir -p "$ROOT" "$MODEL_ROOT"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip wheel --quiet
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 --quiet
pip install -r "$(dirname "$0")/requirements.txt" --quiet

echo "[3/3] server.py + manage.sh 배치"
install -m 0755 "$(dirname "$0")/server.py" "$SERVER_PY"
install -m 0755 "$(dirname "$0")/manage.sh" "$MANAGE_SH"

# Global HF login (한 번만)
export HF_HUB_ENABLE_HF_TRANSFER=1
hf auth login --token "$HF_TOKEN" --add-to-git-credential 2>&1 | tail -1 || true

cat <<EOF

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 설치 완료.  다음 단계:

   # 모델 다운로드
   sudo HF_TOKEN=\$HF_TOKEN $MANAGE_SH download 27b-it
   sudo HF_TOKEN=\$HF_TOKEN $MANAGE_SH download 4b-it

   # 인스턴스 기동 (예: GPU 0 에 27b, GPU 1 에 4b)
   sudo $MANAGE_SH up 27b-it 0 8080
   sudo $MANAGE_SH up  4b-it 1 8081

   # 상태 확인
   sudo $MANAGE_SH list
   curl http://localhost:8080/info

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EOF
