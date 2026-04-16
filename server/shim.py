"""llama-server (TranslateGemma) → /translate 호환 shim.

Known issues fixed:
- 플레이스홀더 (%s, %d, {x}, [tag]) 순서 보존 명시
"""
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

LLAMA_URL = os.environ.get("LLAMA_URL", "http://127.0.0.1:18080")
MODEL_NAME = os.environ.get("MODEL_NAME", "translategemma-27b")
SHIM_PORT = int(os.environ.get("SHIM_PORT", "8080"))

app = FastAPI()
client = httpx.AsyncClient(timeout=120)


def build_prompt(text: str, src: str, tgt: str) -> str:
    # Critical: 플레이스홀더 preservation instruction. 긴 반복 / 형식 토큰 안 전달되는 버그 방지.
    rules = (
        "Rules:\n"
        "- Preserve all format placeholders exactly as-is in their original positions: "
        "%s %d %i %f %x {0} {1} {name} ${var} [tag] [name] <code> <tag>.\n"
        "- Preserve line breaks, leading/trailing whitespace, punctuation.\n"
        "- Output ONLY the translation. No commentary, no quotes around it."
    )
    return (
        "<start_of_turn>user\n"
        f"Translate from {src} to {tgt}.\n{rules}\n\n"
        f"---\n{text}\n---\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


class Req(BaseModel):
    text: str
    source_lang_code: str = "en"
    target_lang_code: str = "ko"
    max_new_tokens: int = 1024


@app.get("/health")
async def health():
    try:
        r = await client.get(f"{LLAMA_URL}/health")
        return {"ok": r.status_code == 200, "backend": "llama.cpp"}
    except Exception:
        return {"ok": False}


@app.get("/info")
async def info():
    return {"model": MODEL_NAME, "backend": "llama.cpp", "upstream": LLAMA_URL, "port": SHIM_PORT}


@app.post("/translate")
async def translate(r: Req):
    if not r.text.strip():
        raise HTTPException(400, "empty text")
    prompt = build_prompt(r.text, r.source_lang_code, r.target_lang_code)
    payload = {
        "prompt": prompt,
        "n_predict": r.max_new_tokens,
        "temperature": 0,
        "stop": ["<end_of_turn>", "<start_of_turn>", "</s>"],
        "cache_prompt": False,
    }
    try:
        resp = await client.post(f"{LLAMA_URL}/completion", json=payload)
        data = resp.json()
        if resp.status_code >= 400:
            raise HTTPException(500, f"llama: {data}")
        out = data.get("content", "").strip()
        # 혹시 모델이 "---" 같은 구분자 포함 응답하면 제거
        if out.startswith("---"):
            out = out[3:].lstrip()
        if out.endswith("---"):
            out = out[:-3].rstrip()
        return {"translation": out}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"shim error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SHIM_PORT)
