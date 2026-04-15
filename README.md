# gemma-translate

Google **TranslateGemma 27B-IT** (Gemma 3, 55 언어) 셀프호스티드 번역 인프라.

설치/관리는 단일 **Rust 바이너리** (`gemma-translate`) 로 합니다. 백엔드 3가지 지원:

| 백엔드 | 특징 | 서브커맨드 |
|--------|------|----------|
| **transformers (NF4)** | Python + bitsandbytes 양자화. 단순. 0.3 req/s | `install` / `up` / `down` |
| **llama.cpp (BF16 GGUF)** | C++ 추론, 4 GPU tensor-split, continuous batching. **3~10x 빠름** | `llama-install` / `llama-up` / `llama-down` |
| **vLLM (BF16/AWQ)** | 최고 throughput. 셋업 복잡. | `vllm-install` / `vllm-up` / `vllm-down` |

## 요구사양

- NVIDIA GPU
  - transformers NF4: VRAM ≥ 16 GB / 인스턴스
  - llama.cpp BF16 (4 GPU 분산): VRAM ≥ 14 GB / 카드 × 4
  - vLLM BF16: VRAM ≥ 16 GB / 카드 (TP=4 권장)
- HuggingFace 계정 + [Gemma 라이선스 동의](https://huggingface.co/google/translategemma-27b-it)

## 설치 (Rust CLI)

```bash
git clone https://github.com/dalsoop/gemma-translate.git
cd gemma-translate/installer
cargo build --release --target x86_64-unknown-linux-musl  # static
sudo install -m 0755 target/x86_64-unknown-linux-musl/release/gemma-translate /usr/local/bin/
```

## 백엔드별 사용

### A. transformers (가장 단순)

```bash
sudo HF_TOKEN=hf_xxx gemma-translate install
sudo gemma-translate up 0 8080
sudo gemma-translate up 1 8081
gemma-translate info 8080
```

### B. llama.cpp (가장 빠른 단일 모델)

```bash
# 옵션 1: 사전 변환된 GGUF (bullerwins) 다운로드
sudo HF_TOKEN=hf_xxx gemma-translate llama-install

# 옵션 2: 기존 safetensors 를 BF16 GGUF 로 로컬 변환
sudo gemma-translate llama-install --from-local /root/models/translategemma-27b-it

# 4 GPU 분산 + shim 포함 기동
sudo gemma-translate llama-up 0,1,2,3 8080

# 단일 GPU
sudo gemma-translate llama-up 0 8080

curl http://localhost:8080/health
```

`llama-up` 은 자동으로 두 systemd 유닛 생성:
- `llama-server-gemma@<port>.service` — llama-server (내부 :18080)
- `translate-llama@<port>.service` — `/translate` 호환 shim

### C. vLLM (PagedAttention)

```bash
sudo HF_TOKEN=hf_xxx gemma-translate vllm-install
sudo gemma-translate vllm-up 0,1,2,3 8080   # TP=4
```

## /translate API (모든 백엔드 공통)

```bash
curl -X POST http://localhost:8080/translate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello","source_lang_code":"en","target_lang_code":"ko"}'
# {"translation":"안녕하세요"}
```

## 범용 CLI (`translate`)

```bash
sudo cp cli/translate /usr/local/bin/translate

export TRANSLATE_API="http://localhost:8080"  # 여러 개면 쉼표구분 round-robin

translate "Hello"
translate -c "short UI button label" "Save"
translate --list "Save,Cancel,Delete" -w 8
translate -i en.json -o ko.json
translate --po django.po
```

## 구조

```
gemma-translate/
├── installer/            Rust CLI (clap + reqwest)
│   └── src/main.rs       install/up/down/info  +  llama-* / vllm-*
├── server/
│   ├── server.py         transformers FastAPI 서버 (CLI 가 embed)
│   └── requirements.txt
├── cli/
│   └── translate         범용 번역 CLI (Python)
└── README.md
```

## 라이선스

- TranslateGemma 모델: [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
- 이 리포지토리 코드: MIT
