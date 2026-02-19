import gradio as gr
import requests
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

API_BASE = "http://127.0.0.1:8000"   # change if server elsewhere


@dataclass
class UIState:
    session_id: Optional[str] = None
    pdf_name: Optional[str] = None


def upload_pdf(pdf_file, state: UIState):
    state = state or UIState()

    if pdf_file is None:
        return state, "❌ Please upload a PDF.", "", ""

    try:
        with open(pdf_file.name, "rb") as f:
            files = {
                "pdf": (
                    pdf_file.orig_name if hasattr(pdf_file, "orig_name") else "file.pdf",
                    f,
                    "application/pdf",
                )
            }
            r = requests.post(f"{API_BASE}/upload", files=files, timeout=300)

        r.raise_for_status()
        data = r.json()

        state.session_id = data["session_id"]
        state.pdf_name = pdf_file.orig_name if hasattr(pdf_file, "orig_name") else pdf_file.name

        meta = data.get("meta", {})
        status = (
            f"✅ Indexed successfully\n\n"
            f"- **session_id:** `{state.session_id}`\n"
            f"- **file:** `{state.pdf_name}`\n"
            f"- **chunks:** {meta.get('chunks','?')}\n"
            f"- **chunk_size:** {meta.get('chunk_size','?')}\n"
            f"- **overlap:** {meta.get('chunk_overlap','?')}"
        )
        return state, status, "", ""

    except Exception as e:
        return state, f"❌ Upload/index failed: {e}", "", ""


def _quote_snip(txt: str, max_len: int = 220) -> str:
    txt = (txt or "").strip().replace("\n", " ")
    if len(txt) > max_len:
        txt = txt[:max_len].rstrip() + "..."
    return f"\"{txt}\""


def ask_question(message: str, history, state: UIState, top_k: int):
    history = history or []
    state = state or UIState()

    if not state.session_id:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "❌ Upload a PDF first (Index PDF)."})
        return "", history, ""

    payload = {
        "input": {
            "session_id": state.session_id,
            "question": message,
            "top_k": int(top_k),
        }
    }

    try:
        r = requests.post(f"{API_BASE}/qa/invoke", json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()

        out = data.get("output", {})
        answer = out.get("answer", "")
        citations_list = out.get("citations", []) or []

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})

        citations_txt = "\n\n".join(
            [
                f"[Chunk {i+1} | Page {c.get('page_number','NA')}] {_quote_snip(c.get('text',''))}"
                for i, c in enumerate(citations_list)
            ]
        )

        return "", history, citations_txt

    except Exception as e:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"❌ API error: {e}"})
        return "", history, ""


def summarize_full_pdf(state: UIState):
    state = state or UIState()

    if not state.session_id:
        return "❌ Upload and index a PDF first."

    payload = {"input": {"session_id": state.session_id}}
    try:
        r = requests.post(f"{API_BASE}/summarize_full/invoke", json=payload, timeout=600)
        r.raise_for_status()
        out = r.json().get("output", {})
        return out.get("summary", "")
    except Exception as e:
        return f"❌ Summary API error: {e}"


def clear_all():
    return UIState(), "🧹 Cleared. Upload and index a new PDF.", [], "", ""


with gr.Blocks(title="PDF Q&A (Frontend → LangServe)") as demo:
    gr.Markdown("## 🌐 PDF Q&A Frontend (Gradio → LangServe API)")

    state = gr.State(UIState())

    with gr.Row():
        with gr.Column(scale=4):
            pdf_file = gr.File(label="Upload PDF / DOCX", file_types=[".pdf", ".docx"])
            status = gr.Markdown("⬅️ Upload a PDF, then click **Index PDF**.")
            top_k = gr.Slider(1, 12, value=4, step=1, label="Top-K retrieval")

            with gr.Row():
                index_btn = gr.Button("Index PDF/DOCX", variant="primary")
                clear_btn = gr.Button("Clear", variant="secondary")

        with gr.Column(scale=6):
            chatbot = gr.Chatbot(label="Chat", height=420)
            msg = gr.Textbox(label="Ask a question", placeholder="e.g., What are the faculties?")
            send = gr.Button("Send", variant="primary")

            with gr.Tabs():
                with gr.TabItem("Citations"):
                    citations = gr.Textbox(lines=12, label="Evidence (chunks/pages)")

                with gr.TabItem("Summary"):
                    summ_btn = gr.Button("Summarize Full PDF", variant="secondary")
                    summary_box = gr.Textbox(lines=14, label="Full PDF Summary")

    # ---- wiring ----
    index_btn.click(upload_pdf, inputs=[pdf_file, state], outputs=[state, status, citations, summary_box])
    send.click(ask_question, inputs=[msg, chatbot, state, top_k], outputs=[msg, chatbot, citations])
    msg.submit(ask_question, inputs=[msg, chatbot, state, top_k], outputs=[msg, chatbot, citations])

    summ_btn.click(summarize_full_pdf, inputs=[state], outputs=[summary_box])
    clear_btn.click(clear_all, outputs=[state, status, chatbot, citations, summary_box])

demo.launch()
