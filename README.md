# gemma-translate

Google **TranslateGemma 27B-IT** (Gemma 3 기반, 55개 언어) 셀프호스티드 번역 서버 + 범용 CLI.

RTX 3090 한 장에 NF4 양자화로 27B 모델을 띄울 수 있습니다. 여러 GPU 에 각각 인스턴스를 기동해 라운드로빈으로 쓰는 구조입니다.

## 요구사양

- NVIDIA GPU (VRAM ≥ 16 GB per instance)
- NVIDIA 드라이버 + CUDA 12.4
- Python 3.10+
- HuggingFace 계정 + [Gemma 라이선스 동의](https://huggingface.co/google/translategemma-27b-it)

## 설치

```bash
git clone https://github.com/dalsoop/gemma-translate.git
cd gemma-translate/server
export HF_TOKEN=hf_xxx
sudo bash install.sh
```

`install.sh` 가 하는 일:
1. venv + PyTorch cu124 + transformers + bitsandbytes 설치
2. `google/translategemma-27b-it` 모델 `/opt/translate-gemma/model` 에 다운로드 (~54 GB)
3. `server.py` 배치 + `up.sh` / `down.sh` 생성

## 인스턴스 기동

```bash
# GPU 마다 기동
sudo /opt/translate-gemma/up.sh 0 8080
sudo /opt/translate-gemma/up.sh 1 8081
sudo /opt/translate-gemma/up.sh 2 8082
sudo /opt/translate-gemma/up.sh 3 8083

# 상태/정보
curl http://localhost:8080/info
# {"model":"27b-it","model_dir":"/opt/translate-gemma/model",
#  "quant":"nf4","vram_gb":16.29,"load_time_s":54.2,
#  "cuda_visible_devices":"0","port":8080}

# 중지
sudo /opt/translate-gemma/down.sh 8080

# 전체 상태
systemctl list-units 'translate-gemma@*'
```

## CLI (`translate`)

```bash
sudo cp cli/translate /usr/local/bin/translate

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

## 양자화 (선택)

기본 `nf4` (16 GB VRAM). 품질을 더 올리려면 systemd 유닛에서 `QUANT=int8` 또는 `QUANT=none` (풀 BF16, ~54 GB 필요) 로 변경.

## 구조

```
gemma-translate/
├── server/
│   ├── server.py          FastAPI + transformers
│   ├── requirements.txt
│   └── install.sh         venv + 모델 다운로드 + up.sh/down.sh 생성
├── cli/
│   └── translate          범용 번역 CLI
└── README.md
```

## 라이선스

- TranslateGemma 모델: [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
- 이 리포지토리 코드: MIT
