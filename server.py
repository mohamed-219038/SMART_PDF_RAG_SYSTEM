import uuid
import tempfile
from typing import Dict, Any, List, Optional

import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from langserve import add_routes
from langchain_core.runnables import RunnableLambda


from rag.pipeline import build_session, answer_question, answer_question_with_history
from rag.config import load_config

app = FastAPI(title="PDF Q&A API (LangServe)")

from session_store import SESSIONS, HISTORIES

# -----------------------------
# REST: upload + index PDF
# -----------------------------
class IndexResponse(BaseModel):
    session_id: str
    meta: Dict[str, Any]

@app.post("/upload", response_model=IndexResponse)
async def upload_pdf(pdf: UploadFile = File(...)):
    ext = os.path.splitext(pdf.filename or "")[1].lower()
    if ext not in (".pdf", ".docx"):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")

    # Save uploaded file to a temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(await pdf.read())
    tmp.close()

    cfg = load_config()  # uses your config.yaml if present
    session, meta = build_session(tmp.name, cfg)

    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {"session": session, "cfg": cfg, "meta": meta}
    HISTORIES[session_id] = []  # init conversation history

    return IndexResponse(session_id=session_id, meta=meta)

# -----------------------------
# LangServe runnable: /qa/invoke
# -----------------------------
class QAInput(BaseModel):
    session_id: str
    question: str
    top_k: Optional[int] = None

class QAOutput(BaseModel):
    answer: str
    citations: List[dict]

def _qa_invoke(inp: dict) -> dict:
    data = QAInput(**inp)

    if data.session_id not in SESSIONS:
        raise ValueError("Invalid session_id. Upload a PDF first.")

    pack = SESSIONS[data.session_id]
    cfg = pack["cfg"]

    if data.top_k is not None:
        cfg = dict(cfg)
        cfg["retrieval"] = dict(cfg.get("retrieval", {}))
        cfg["retrieval"]["top_k"] = int(data.top_k)

    # use history-aware pipeline with guardrails
    history = HISTORIES.get(data.session_id, [])
    answer, retrieved = answer_question_with_history(
        data.question, pack["session"], cfg, history=history
    )

    # track conversation turns
    HISTORIES.setdefault(data.session_id, []).append({"role": "user", "content": data.question})
    HISTORIES[data.session_id].append({"role": "assistant", "content": answer})

    # retrieved is your list of dicts: {"text", "page_number", "metadata"}
    return {"answer": answer, "citations": retrieved}

qa_runnable = RunnableLambda(_qa_invoke).with_types(input_type=QAInput, output_type=QAOutput)


add_routes(app, qa_runnable, path="/qa")


from summarize_chain import summarize_full_runnable
add_routes(app, summarize_full_runnable, path="/summarize_full", enabled_endpoints=["invoke", "batch"])


