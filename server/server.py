"""
TranslateGemma 27B-IT — 번역 전용 서버
단일 RTX 3090 (GPU 2) + bitsandbytes NF4
chat template 요구 포맷: source_lang_code + target_lang_code + text
"""
import os, time, logging
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration, BitsAndBytesConfig

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("translate")

MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/translate-gemma/model")

log.info("loading tokenizer…")
tok = AutoTokenizer.from_pretrained(MODEL_DIR)

log.info("loading model (NF4 quant)…")
t0 = time.time()
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_DIR, quantization_config=bnb, device_map="auto", dtype=torch.bfloat16
).eval()
log.info("loaded in %.1fs  vram=%.2f GB", time.time() - t0,
         torch.cuda.memory_allocated() / 1e9)

app = FastAPI(title="TranslateGemma 27B")


class Req(BaseModel):
    text: str
    source_lang_code: str = "en"    # BCP-47 like "en", "ko", "ja", "zh-CN"
    target_lang_code: str = "ko"
    max_new_tokens: int = 512


@app.get("/health")
def health():
    return {"ok": True, "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 2)}


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
        out = model.generate(**enc, max_new_tokens=r.max_new_tokens,
                             do_sample=False)
    input_len = enc["input_ids"].shape[-1]
    gen = out[0][input_len:]
    text = tok.decode(gen, skip_special_tokens=True).strip()
    return {"translation": text, "elapsed_s": round(time.time() - t0, 3)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("TRANSLATE_PORT", 8080)))
