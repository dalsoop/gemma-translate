# gemma-translate

Google **TranslateGemma 27B-IT** (Gemma 3 기반, 55개 언어) 셀프호스티드 번역 서버.

RTX 3090 한 장에 NF4 양자화로 27B 모델을 띄우고 여러 GPU 에 인스턴스를 분산시킵니다.
설치/관리는 단일 **Rust 바이너리** (`gemma-translate`) 로 합니다.

## 요구사양

- NVIDIA GPU (VRAM ≥ 16 GB per instance)
- NVIDIA 드라이버 + CUDA 12.4
- Python 3.10+ (서버 런타임용 — 설치 CLI가 venv 자동 관리)
- HuggingFace 계정 + [Gemma 라이선스 동의](https://huggingface.co/google/translategemma-27b-it)
- Rust toolchain (설치 CLI 빌드용)

## 설치

```bash
git clone https://github.com/dalsoop/gemma-translate.git
cd gemma-translate/installer
cargo build --release
sudo install -m 0755 target/release/gemma-translate /usr/local/bin/

# 모델 다운로드 + 공통 환경 구축 (~54 GB, 10~20분)
sudo HF_TOKEN=hf_xxx gemma-translate install
```

## 인스턴스 기동

```bash
sudo gemma-translate up 0 8080    # GPU 0 → port 8080
sudo gemma-translate up 1 8081
sudo gemma-translate up 2 8082
sudo gemma-translate up 3 8083

gemma-translate list              # 설치 상태 + 실행중 인스턴스
gemma-translate info 8080         # 특정 포트의 /info 조회

sudo gemma-translate down 8080    # 중지
```

`QUANT=int8` 또는 `QUANT=none` env 로 올리면 다른 양자화 사용 (`sudo QUANT=int8 gemma-translate up 0 8080`).

## CLI (`translate`) — 범용 번역

```bash
sudo cp cli/translate /usr/local/bin/translate

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
# {"model":"27b-it","quant":"nf4","vram_gb":16.29,...}
```

## 구조

```
gemma-translate/
├── installer/               Rust 설치/관리 CLI
│   ├── Cargo.toml
│   └── src/main.rs          install / up / down / list / info
├── server/
│   ├── server.py            FastAPI + transformers (installer 가 embed)
│   └── requirements.txt
├── cli/
│   └── translate            범용 번역 CLI (Python)
└── README.md
```

## 라이선스

- TranslateGemma 모델: [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
- 이 리포지토리 코드: MIT
