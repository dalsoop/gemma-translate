# gemma-translate

Google **TranslateGemma** (Gemma 3 기반, 55개 언어) 기반 셀프호스티드 번역 서버 + 범용 CLI.

RTX 3090 1장으로 27B 모델을 NF4 양자화해 돌릴 수 있습니다. `/translate` HTTP 엔드포인트로 동작.

## 요구사양

- NVIDIA GPU (VRAM: 27B NF4 기준 ≥16GB, 4B 기준 ≥8GB)
- NVIDIA 드라이버 + CUDA 12.4 호환
- Python 3.10+
- HuggingFace 계정 + [Gemma 라이선스 동의](https://huggingface.co/google/translategemma-27b-it)

## 설치 (서버)

```bash
git clone <this-repo>
cd gemma-translate/server
export HF_TOKEN=hf_xxx           # huggingface 토큰
sudo bash install.sh             # 기본: 27b-it / GPU 0 / 포트 8080
# or:
sudo MODEL=4b-it CUDA_IDX=1 PORT=8081 bash install.sh
```

설치 완료 후:

```bash
curl http://localhost:8080/health
# {"ok":true,"vram_gb":16.29}
```

## 사용 (CLI)

```bash
cp cli/translate /usr/local/bin/translate
chmod +x /usr/local/bin/translate

# 환경변수로 서버 주소 지정 (여러 개면 쉼표구분 → 라운드로빈)
export TRANSLATE_API="http://localhost:8080"

translate "Hello, how are you?"
# 안녕하세요, 어떻게 지내세요?

# UI 버튼처럼 짧은 문자열엔 context 힌트
translate -c "short UI button label, Korean noun form" "Save"
# 저장

# 쉼표 구분 리스트
translate --list "Save,Cancel,Delete" -w 4
# Save  → 저장
# Cancel → 취소
# Delete → 삭제

# JSON 파일 {key:value} 전체 번역
translate -i en.json -o ko.json -w 8

# 텍스트 파일 (줄당 1개 청크, 체크포인트 이어받기)
translate -i article.en.txt -o article.ko.txt

# gettext .po 파일 in-place (빈 msgstr 만 채움)
translate --po django.po

# 언어 변경
translate -s en -t ja "Hello"
```

## API

```bash
curl -X POST http://localhost:8080/translate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello","source_lang_code":"en","target_lang_code":"ko"}'
# {"translation":"안녕하세요","elapsed_s":0.8}
```

파라미터:
- `text` (str) — 번역할 텍스트
- `source_lang_code` (str, 기본 "en") — BCP-47 코드
- `target_lang_code` (str, 기본 "ko")
- `max_new_tokens` (int, 기본 512)

## 다중 GPU / 복수 인스턴스

GPU 마다 별도 서비스 기동:

```bash
sudo CUDA_IDX=0 PORT=8080 bash server/install.sh
sudo CUDA_IDX=1 PORT=8081 bash server/install.sh
```

CLI에서 라운드로빈:
```bash
export TRANSLATE_API="http://localhost:8080,http://localhost:8081"
translate ...
```

## 구성

```
gemma-translate/
├── server/
│   ├── server.py          FastAPI + transformers NF4 추론
│   ├── requirements.txt
│   └── install.sh         시스템 설치 (venv + 모델 다운로드 + systemd)
├── systemd/
│   └── translate.service  systemd unit 예시
├── cli/
│   └── translate          범용 번역 CLI (단일/리스트/JSON/PO/텍스트)
└── README.md
```

## 라이선스

- TranslateGemma 모델: [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
- 이 리포지토리 코드: MIT
