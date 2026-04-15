# gemma-translate

Google **TranslateGemma** (Gemma 3 기반, 55개 언어) 기반 셀프호스티드 번역 서버 + 범용 CLI.

**다중 모델 프로파일 지원** — 같은 서버에 4b-it / 27b-it 를 GPU 별로 섞어 기동 가능.

## 요구사양

- NVIDIA GPU
  - 27b-it NF4: VRAM ≥ 16 GB
  - 4b-it NF4: VRAM ≥ 6 GB
- NVIDIA 드라이버 + CUDA 12.4 호환
- Python 3.10+
- HuggingFace 계정 + [Gemma 라이선스 동의](https://huggingface.co/google/translategemma-27b-it)

## 설치

```bash
git clone https://github.com/dalsoop/gemma-translate.git
cd gemma-translate/server
export HF_TOKEN=hf_xxx
sudo bash install.sh
```

공통 venv + `server.py` + `manage.sh` 가 `/opt/translate-gemma/` 에 배치됩니다.

## 모델 다운로드 / 기동

```bash
# 원하는 프로파일 다운로드 (필요한 만큼 반복)
sudo HF_TOKEN=$HF_TOKEN /opt/translate-gemma/manage.sh download 27b-it
sudo HF_TOKEN=$HF_TOKEN /opt/translate-gemma/manage.sh download 4b-it

# GPU 마다 인스턴스 기동 (GPU idx, port)
sudo /opt/translate-gemma/manage.sh up 27b-it 0 8080
sudo /opt/translate-gemma/manage.sh up 27b-it 1 8081
sudo /opt/translate-gemma/manage.sh up  4b-it 2 8082
sudo /opt/translate-gemma/manage.sh up  4b-it 3 8083

# 중지
sudo /opt/translate-gemma/manage.sh down 8083

# 상태
sudo /opt/translate-gemma/manage.sh list
curl http://localhost:8080/info   # 모델/VRAM/로딩시간 확인
```

`QUANT=nf4|int8|none` 로 양자화 선택 가능 (기본 `nf4`).

## CLI (`translate`)

```bash
cp cli/translate /usr/local/bin/translate
chmod +x /usr/local/bin/translate

# 여러 엔드포인트 라운드로빈
export TRANSLATE_API="http://localhost:8080,http://localhost:8081,http://localhost:8082,http://localhost:8083"

translate "Hello"
translate -c "short UI button label, Korean noun form" "Save"
translate --list "Save,Cancel,Delete" -w 8
translate -i en.json -o ko.json
translate --po django.po
translate -s en -t ja "Hello"
```

## HTTP API

```bash
# 번역
curl -X POST http://localhost:8080/translate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello","source_lang_code":"en","target_lang_code":"ko"}'
# {"translation":"안녕하세요","elapsed_s":0.8}

# 인스턴스 정보
curl http://localhost:8080/info
# {"model":"27b-it","model_dir":"/opt/translate-gemma/models/27b-it",
#  "quant":"nf4","vram_gb":16.29,"load_time_s":54.2,
#  "cuda_visible_devices":"0","port":8080}
```

## 다중 모델 사용 전략

| 시나리오 | 구성 |
|---------|------|
| 품질 최우선 | 전 GPU `27b-it` |
| 속도 최우선 | 전 GPU `4b-it` (4배 빠름) |
| 하이브리드 | 1~2장 `27b-it` (긴 문장), 2~3장 `4b-it` (UI 레이블) — 클라이언트에서 길이로 라우팅 |
| 개발/테스트 | 1장에 `4b-it`, 1장에 `27b-it` — 품질 비교용 |

## 구조

```
gemma-translate/
├── server/
│   ├── server.py          FastAPI + transformers (4b/27b/... 공용)
│   ├── requirements.txt
│   ├── install.sh         시스템 설치 (venv + 공통 환경)
│   └── manage.sh          list/download/up/down/info 관리 CLI
├── systemd/
│   └── translate.service  (레거시 예시, 실제 유닛은 manage.sh 가 생성)
├── cli/
│   └── translate          범용 번역 CLI
└── README.md
```

## 라이선스

- TranslateGemma 모델: [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
- 이 리포지토리 코드: MIT
