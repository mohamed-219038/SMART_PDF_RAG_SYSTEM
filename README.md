# Smart Contract Summary & Q&A Assistant

A web application that lets users upload long documents (contracts, insurance policies, reports) and interact with them via a conversational AI assistant powered by a RAG (Retrieval Augmented Generation) pipeline.

## Features

- **File Ingestion** – Upload PDF or DOCX files for processing
- **Chunking & Embedding** – Documents are split into chunks and embedded using SentenceTransformers
- **Vector Store** – FAISS-based semantic search over document chunks
- **LLM Q&A with Citations** – Ask questions and get grounded answers with source page references
- **Conversation History** – Multi-turn chat with context from previous exchanges
- **Document Summarization** – Map-reduce summarization of the full document
- **Guard-rails** – Prompt-injection defense + semantic similarity threshold to reject irrelevant queries
- **LongContextReorder** – Reorders retrieved chunks for optimal placement in the LLM context window
- **Evaluation Pipeline** – Automated metrics for retrieval and answer quality

## Architecture

```
┌──────────────┐       ┌───────────────────────────────┐
│  Gradio UI   │──────▶│  FastAPI + LangServe Backend   │
│  (Frontend)  │◀──────│  /upload  /qa  /summarize_full │
└──────────────┘       └──────────┬────────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼              ▼
              ┌──────────┐ ┌──────────┐  ┌────────────┐
              │ Ingestion│ │ Retrieval│  │Summarization│
              │ Pipeline │ │ + Q&A    │  │   Chain     │
              └────┬─────┘ └────┬─────┘  └─────┬──────┘
                   ▼            ▼               ▼
              ┌─────────┐ ┌─────────┐    ┌────────────┐
              │ FAISS   │ │  Groq   │    │  Groq LLM  │
              │ VecDB   │ │  LLM    │    │ (map/reduce)│
              └─────────┘ └─────────┘    └────────────┘
```

## Technology Stack

| Component | Library |
|---|---|
| LLM | Groq (Llama 3.1-8B-Instant) via `langchain-groq` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS (`faiss-cpu`) |
| Framework | LangChain, LangServe |
| Backend | FastAPI + Uvicorn |
| Frontend | Gradio |
| File Parsing | PyMuPDF (PDF), python-docx (DOCX) |

## Project Structure

```
final_code/
├── server.py                # FastAPI + LangServe endpoints
├── gradio_client_app.py     # Gradio frontend
├── summarize_chain.py       # Map-reduce summarization
├── session_store.py         # In-memory session & history stores
├── config.yaml              # Pipeline configuration
├── pass.env                 # API key (not committed)
├── requirements.txt         # Python dependencies
├── evaluate.py              # Evaluation pipeline
├── eval_dataset.json        # Sample evaluation Q&A pairs
├── README.md                # This file
└── rag/
    ├── config.py            # Config loader with defaults
    ├── loader.py            # PDF + DOCX file loading
    ├── chunking.py          # Recursive text splitting
    ├── embeddings.py        # HuggingFace embeddings
    ├── index.py             # FAISS index builder
    ├── retriever.py         # Vector DB retriever
    ├── generator.py         # LLM chain (standard + history-aware)
    ├── guardrails.py        # Semantic similarity guardrail
    ├── pipeline.py          # End-to-end session + QA orchestration
    └── schema.py            # Data classes
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API key

Create a `pass.env` file in the project root:

```
API_KEY=your_groq_api_key_here
```

### 3. (Optional) Edit `config.yaml`

Adjust chunk size, overlap, top-k, model, or temperature as needed.

## Running

### Start the backend server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

### Start the Gradio frontend (in a separate terminal)

```bash
python gradio_client_app.py
```

Then open the Gradio URL shown in the terminal (usually `http://127.0.0.1:7860`).

### Usage

1. Upload a PDF or DOCX file and click **Index PDF**.
2. Ask questions in the chat box — answers include source citations.
3. Use the **Summary** tab to generate a full document summary.
4. Adjust the **Top-K** slider to control how many chunks are retrieved.

## Evaluation

Run the evaluation pipeline against any document:

```bash
python evaluate.py <path_to_pdf_or_docx>
```

Options:

```
--dataset eval_dataset.json    # Custom Q&A dataset
--output  eval_results.json    # Output file for results
```

The script reports:

| Metric | Description |
|---|---|
| Keyword Precision | Fraction of expected keywords found in the answer |
| Retrieval Precision@K | Fraction of retrieved chunks with meaningful content |
| Non-Empty Retrieval Rate | % of questions that retrieved at least one chunk |
| Average Latency | Seconds per question |

## Configuration Reference

```yaml
env:
  dotenv_path: "pass.env"
  api_key_name: "API_KEY"

chunking:
  chunk_size: 512
  chunk_overlap: 50
  separators: ["\n\n", "\n", ".", " ", ""]
  add_start_index: true

embeddings:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"

retrieval:
  top_k: 4

llm:
  provider: "groq"
  model: "llama-3.1-8b-instant"
  temperature: 0
```

## Limitations

- English-only documents
- In-memory session storage (not persisted across restarts)
- Single-user local deployment
- Evaluation metrics are keyword-based (not model-based)
