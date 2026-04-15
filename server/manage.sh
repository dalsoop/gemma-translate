#!/usr/bin/env bash
# gemma-translate 모델 관리 CLI
#
# 사용:
#   manage.sh list                         # 설치된 프로파일 + 실행중 인스턴스
#   manage.sh download 4b-it               # HF 에서 모델만 다운로드 (systemd 유닛 설치 X)
#   manage.sh download 27b-it
#   manage.sh up   <profile> <gpu> <port>  # 인스턴스 기동 (systemd 유닛 생성 + 시작)
#     예: manage.sh up 4b-it 0 8080
#   manage.sh down <gpu> <port>            # 인스턴스 중지 + 유닛 삭제
#   manage.sh info <port>                  # curl /info
#
# 환경변수:
#   HF_TOKEN       HuggingFace 토큰 (Gemma 게이트 통과)
#   MODEL_ROOT     모델 저장 루트 (기본 /opt/translate-gemma/models)
#   VENV           venv 경로 (기본 /opt/translate-gemma/venv)
#   SERVER_PY      server.py 경로 (기본 /opt/translate-gemma/server.py)
#   QUANT          nf4|int8|none (기본 nf4)

set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-/opt/translate-gemma/models}"
VENV="${VENV:-/opt/translate-gemma/venv}"
SERVER_PY="${SERVER_PY:-/opt/translate-gemma/server.py}"
QUANT="${QUANT:-nf4}"

cmd="${1:-help}"; shift || true

case "$cmd" in
  list)
    echo "── 설치된 모델 ──"
    [[ -d "$MODEL_ROOT" ]] && ls "$MODEL_ROOT" 2>/dev/null | sed 's/^/  /' || echo "  (없음)"
    echo
    echo "── 실행중 인스턴스 ──"
    systemctl list-units 'translate-gemma@*' --no-legend 2>/dev/null \
      | awk '{print "  "$1"  "$4}' || echo "  (없음)"
    ;;
  download)
    profile="${1:?profile 이름 필요 (예: 4b-it)}"
    [[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN 환경변수 필요"; exit 1; }
    repo="google/translategemma-${profile}"
    target="$MODEL_ROOT/$profile"
    mkdir -p "$target"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    export HF_HUB_ENABLE_HF_TRANSFER=1
    hf auth login --token "$HF_TOKEN" --add-to-git-credential 2>&1 | tail -1 || true
    hf download "$repo" --local-dir "$target"
    echo "다운로드 완료: $target"
    ;;
  up)
    profile="${1:?profile 이름 필요}"
    gpu="${2:?GPU index 필요}"
    port="${3:?port 필요}"
    target="$MODEL_ROOT/$profile"
    [[ -d "$target" ]] || { echo "모델 없음: $target  먼저 download"; exit 1; }
    unit="/etc/systemd/system/translate-gemma@${port}.service"
    cat > "$unit" <<EOF
[Unit]
Description=TranslateGemma ($profile on GPU $gpu, port $port)
After=network.target

[Service]
Type=simple
WorkingDirectory=$(dirname "$SERVER_PY")
Environment="CUDA_VISIBLE_DEVICES=$gpu"
Environment="TRANSLATE_PORT=$port"
Environment="MODEL_DIR=$target"
Environment="MODEL_NAME=$profile"
Environment="QUANT=$QUANT"
ExecStart=$VENV/bin/python $SERVER_PY
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload
    systemctl enable --now "translate-gemma@${port}.service"
    echo "기동: translate-gemma@${port}  ($profile, GPU $gpu)"
    echo "확인: curl http://localhost:$port/info"
    ;;
  down)
    port="${1:?port 필요}"
    systemctl disable --now "translate-gemma@${port}.service" 2>/dev/null || true
    rm -f "/etc/systemd/system/translate-gemma@${port}.service"
    systemctl daemon-reload
    echo "중지: translate-gemma@${port}"
    ;;
  info)
    port="${1:-8080}"
    curl -sS "http://localhost:$port/info" | (command -v jq >/dev/null && jq || cat)
    ;;
  help|*)
    sed -n '1,30p' "$0"
    ;;
esac
