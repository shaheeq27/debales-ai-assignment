# Debales AI Assistant — LangGraph Agent

A production-ready AI chatbot that answers questions about **Debales AI** using RAG,
falls back to live web search (SERP API) for general queries, and routes intelligently
between both using a **LangGraph** workflow.

---

## Architecture

```
User query
    │
    ▼
┌─────────────┐
│ Router Node │  (LLM decides: rag | serp | both)
└──────┬──────┘
       │
   ┌───┴────────────┐
   │                │                │
   ▼                ▼                ▼
RAG Node       SERP Node        Both Node
(FAISS         (SerpAPI         (RAG + SERP
 retrieval)     search)          combined)
   │                │                │
   └────────────────┴────────────────┘
                    │
                    ▼
            ┌──────────────┐
            │ Answer Node  │  (GPT-4o-mini, grounded in context)
            └──────────────┘
                    │
                    ▼
              Final answer
```

### Components

| File | Role |
|------|------|
| `src/agent.py` | LangGraph graph definition (all nodes + edges) |
| `scripts/ingest.py` | Website crawler → FAISS vector index builder |
| `cli.py` | Interactive terminal chat interface |
| `app.py` | Streamlit web UI |

### Routing logic

| Query type | Route | Source |
|------------|-------|--------|
| About Debales AI (products, pricing, integrations, team, blog) | `rag` | FAISS vector store |
| General / external questions | `serp` | SerpAPI → Google |
| Mixed queries | `both` | Both sources combined |

---

## Setup

### 1. Clone & install

```bash
git clone <repo-url>
cd debales-agent
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment variables

```bash
cp .env.example .env
# Edit .env and add your keys:
#   OPENAI_API_KEY   – from https://platform.openai.com/api-keys
#   SERPAPI_API_KEY  – from https://serpapi.com/manage-api-key
```

### 3. Build the knowledge base

```bash
python scripts/ingest.py
```

This crawls Debales AI's website (up to 60 pages), chunks the content, embeds it with
OpenAI embeddings, and saves a FAISS index to `data/faiss_index/`.

Typical runtime: **2–4 minutes** (depends on site size and OpenAI API latency).

---

## Running

### CLI (terminal)

```bash
python cli.py
```

### Web UI (Streamlit)

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## Example interactions

```
You: What is Debales AI?
→ [Route: rag]  Retrieves from knowledge base.

You: What is the current EUR/USD exchange rate?
→ [Route: serp] Searches the web and returns live data.

You: How does Debales AI compare to Intercom?
→ [Route: both] RAG for Debales context + SERP for Intercom info.

You: Who is the CEO of Debales AI?
→ [Route: rag]  Answers from scraped data; says "I don't know" if not found.
```

---

## Key design decisions

- **No hallucination**: The answer prompt explicitly instructs the LLM to say "I don't
  know" if context is insufficient. It never fabricates Debales AI facts.
- **Routing via LLM**: A fast, prompt-based classifier decides the route before any
  retrieval happens — cheap and accurate.
- **BFS crawler**: Follows internal links breadth-first so product/blog/integration
  pages are all captured.
- **Chunking strategy**: 800-token chunks with 100-token overlap preserve sentence
  boundaries while keeping retrieval focused.
- **Multi-turn memory**: Both CLI and UI maintain conversation history (last 10 turns).

---

## Project structure

```
debales-agent/
├── src/
│   └── agent.py          # LangGraph workflow (core logic)
├── scripts/
│   └── ingest.py         # Scraper + FAISS index builder
├── data/
│   └── faiss_index/      # Generated – not committed to git
├── cli.py                # CLI chat interface
├── app.py                # Streamlit web UI
├── requirements.txt
├── .env.example
└── README.md
```

---

## Environment variables reference

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI key for GPT-4o-mini + embeddings |
| `SERPAPI_API_KEY` | SerpAPI key for Google Search |
