from typing import Any, Dict, List, Tuple

from rag.loader import load_file
from rag.chunking import chunk_docs
from rag.embeddings import get_embeddings
from rag.index import build_faiss_index
from rag.retriever import make_retriever
from rag.generator import make_llm, make_chain, make_history_chain
from rag.guardrails import check_relevance
from rag.schema import RetrievedChunk


def build_session(pdf_path: str, cfg: Dict[str, Any]):
    docs = load_file(pdf_path)

    chunks = chunk_docs(
        docs,
        chunk_size=cfg["chunking"]["chunk_size"],
        chunk_overlap=cfg["chunking"]["chunk_overlap"],
        separators=cfg["chunking"].get("separators"),
        add_start_index=cfg["chunking"].get("add_start_index", True),
    )

    emb = get_embeddings(cfg["embeddings"]["model_name"])
    vec_db = build_faiss_index(chunks, emb)

    retriever = make_retriever(vec_db, top_k=cfg["retrieval"]["top_k"])

    llm = make_llm(
        dotenv_path=cfg["env"]["dotenv_path"],
        api_key_name=cfg["env"]["api_key_name"],
        model=cfg["llm"]["model"],
        temperature=cfg["llm"]["temperature"],
    )

    chain = make_chain(retriever, llm)
    history_chain = make_history_chain(retriever, llm)

    meta = {
        "chunks": len(chunks),
        "chunk_size": cfg["chunking"]["chunk_size"],
        "chunk_overlap": cfg["chunking"]["chunk_overlap"],
    }

    # session object used by UI
    session = {
        "pdf_path": pdf_path,
        "docs": docs,
        "chunks": chunks,
        "vec_db": vec_db,
        "retriever": retriever,
        "llm": llm,
        "chain": chain,
        "history_chain": history_chain,
        "embeddings": emb,
    }
    return session, meta


def answer_question(question: str, session: Dict[str, Any], cfg: Dict[str, Any]):
    retriever = session["retriever"]
    chain = session["chain"]

    # apply top_k safely
    try:
        retriever.search_kwargs["k"] = int(cfg["retrieval"]["top_k"])
    except Exception:
        pass

    # ✅ LangChain-safe retrieval across versions
    docs = retriever.invoke(question)

    retrieved = []
    for d in docs:
        retrieved.append(
            {
                "text": d.page_content,
                "page_number": str(d.metadata.get("page_number", "NA")),
                "metadata": d.metadata,
            }
        )

    answer = chain.invoke(question)
    return answer, retrieved


def answer_question_with_history(
    question: str,
    session: Dict[str, Any],
    cfg: Dict[str, Any],
    history: List[Dict[str, str]] = None,
):
    retriever = session["retriever"]
    history_chain = session["history_chain"]
    emb = session.get("embeddings")

    # apply top_k safely
    try:
        retriever.search_kwargs["k"] = int(cfg["retrieval"]["top_k"])
    except Exception:
        pass

    # retrieve docs
    docs = retriever.invoke(question)

    retrieved = []
    for d in docs:
        retrieved.append(
            {
                "text": d.page_content,
                "page_number": str(d.metadata.get("page_number", "NA")),
                "metadata": d.metadata,
            }
        )

    # --- semantic similarity guardrail ---
    if emb and docs:
        is_relevant, avg_score = check_relevance(question, docs, emb, threshold=0.10)
        if not is_relevant:
            return (
                "Not found — the uploaded document does not appear to contain "
                f"information relevant to your question (relevance score: {avg_score:.2f}).",
                retrieved,
            )

    # generate answer with conversation history
    answer = history_chain.invoke({"question": question, "history": history or []})
    return answer, retrieved
