"""
TranslateGemma 27B-IT 번역 서버.

환경변수:
  MODEL_DIR       모델 로컬 경로 (기본 /opt/translate-gemma/model)
  TRANSLATE_PORT  리스닝 포트 (기본 8080)
  CUDA_VISIBLE_DEVICES  GPU 인덱스
  QUANT           "nf4" (default) | "int8" | "none"
"""
import os, time, logging
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("translate")

MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/translate-gemma/model")
MODEL_NAME = os.environ.get("MODEL_NAME", "27b-it")
QUANT = os.environ.get("QUANT", "nf4").lower()

log.info("model=%s  dir=%s  quant=%s", MODEL_NAME, MODEL_DIR, QUANT)
log.info("loading tokenizer…")
tok = AutoTokenizer.from_pretrained(MODEL_DIR)

log.info("loading model…")
t0 = time.time()
kwargs = {"device_map": "auto", "dtype": torch.bfloat16}
if QUANT == "nf4":
    kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
elif QUANT == "int8":
    kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
# QUANT=none → 풀 BF16

model = Gemma3ForConditionalGeneration.from_pretrained(MODEL_DIR, **kwargs).eval()
LOAD_TIME_S = time.time() - t0
log.info("loaded in %.1fs  vram=%.2f GB", LOAD_TIME_S, torch.cuda.memory_allocated() / 1e9)

app = FastAPI(title="TranslateGemma 27B")


class Req(BaseModel):
    text: str
    source_lang_code: str = "en"
    target_lang_code: str = "ko"
    max_new_tokens: int = 512


# health/info 는 asyncio 이벤트 루프에서 즉시 응답 (inference threadpool 과 분리)
@app.get("/health")
async def health():
    return {"ok": True, "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 2)}


@app.get("/info")
async def info():
    return {
        "model": MODEL_NAME,
        "model_dir": MODEL_DIR,
        "quant": QUANT,
        "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "load_time_s": round(LOAD_TIME_S, 2),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "port": int(os.environ.get("TRANSLATE_PORT", 8080)),
    }


@app.post("/translate")
def translate(r: Req):
    if not r.text.strip():
        raise HTTPException(400, "empty text")
    msgs = [{
        "role": "user",
        "content": [{
            "type": "text",
            "source_lang_code": r.source_lang_code,
            "target_lang_code": r.target_lang_code,
            "text": r.text,
        }],
    }]
    enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True,
    ).to(model.device)
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=r.max_new_tokens, do_sample=False)
    input_len = enc["input_ids"].shape[-1]
    gen = out[0][input_len:]
    text = tok.decode(gen, skip_special_tokens=True).strip()
    return {"translation": text, "elapsed_s": round(time.time() - t0, 3)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("TRANSLATE_PORT", 8080)))
