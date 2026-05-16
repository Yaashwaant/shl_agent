# SHL Assessment Advisor — AI Intern Assignment

> A conversational AI agent that guides hiring managers from vague role intent to a grounded shortlist of SHL assessments through multi-turn dialogue.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI 0.128 |
| Agent Orchestration | LangGraph 0.6 |
| LLM | Google Gemini 1.5 Flash |
| Vector Store | ChromaDB (persistent) |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| Resilience | Custom Circuit Breaker |
| Logging | Structured JSON logging |
| Testing | pytest + FastAPI TestClient |

---

## Project Structure

```
shl_agent/
├── app/
│   ├── api/
│   │   └── routes.py           # FastAPI endpoints (/health, /chat)
│   ├── core/
│   │   ├── config.py           # pydantic-settings configuration
│   │   ├── logging_config.py   # JSON structured logging
│   │   └── circuit_breaker.py  # Thread-safe circuit breaker
│   ├── models/
│   │   └── schemas.py          # Pydantic request/response models
│   ├── services/
│   │   ├── agent.py            # LangGraph agent (classify→retrieve→recommend)
│   │   └── vector_store.py     # ChromaDB semantic search + metadata filtering
│   └── main.py                 # FastAPI app factory + lifespan
├── data/
│   ├── shl_catalog.json        # Scraped catalog (seeded with 55 products)
│   └── chroma_db/              # Persistent vector index (auto-generated)
├── logs/
│   └── app.log                 # Structured JSON log file
├── scripts/
│   ├── build_catalog.py        # Web scraper for SHL catalog
│   ├── scrape_catalog.py       # Alternative scraper
│   └── build_index.py          # Rebuild vector index from JSON
├── tests/
│   ├── test_api.py             # API endpoint tests
│   └── test_agent.py           # Unit tests (circuit breaker, schemas, filters)
├── main.py                     # Entry point
├── requirements.txt
└── .env.example
```

---

## Quick Start

### 1. Install dependencies

```bash
cd shl_agent
pip install -r requirements.txt
```

### 2. Set up environment

```bash
cp .env.example .env
# Edit .env and add your Google Gemini API key:
# GOOGLE_API_KEY=your_key_here
```

Get a free Gemini API key at: https://aistudio.google.com/app/apikey

### 3. (Optional) Re-scrape the catalog

```bash
python scripts/build_catalog.py
```

The catalog is already seeded with 55 known SHL Individual Test Solutions. Run this only to refresh.

### 4. Start the server

```bash
python main.py
# OR
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The server will:
1. Load catalog JSON from `data/shl_catalog.json`
2. Build ChromaDB vector index (skipped if already built)
3. Compile the LangGraph agent graph
4. Start serving on port 8000

### 5. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I am hiring a senior Java developer who works with stakeholders."}
    ]
  }'
```

### 6. Run tests

```bash
cd shl_agent
python -m pytest tests/ -v
```

---

## API Reference

### GET /health

Returns service readiness.

```json
{"status": "ok"}
```

### POST /chat

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are 5 assessments...",
  "recommendations": [
    {
      "name": "Core Java (Advanced Level) (New)",
      "url": "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}
```

**Constraints:**
- Max 8 turns per conversation (enforced by API)
- 1–10 recommendations when returning a shortlist
- `recommendations` is `null` when still gathering context
- `end_of_conversation: true` only when task is complete

---

## Agent Architecture

```
User Message
    │
    ▼
[classify] ──── intent=clarify ──────► [clarify] → reply with question
    │
    ├── intent=recommend/refine ──► [retrieve] → [recommend] → reply + recs
    │
    ├── intent=compare ──────────► [compare] → reply with comparison
    │
    ├── intent=confirm/end ──────► [confirm] → reply + end_of_conversation=true
    │
    └── intent=refuse ───────────► [refuse] → decline off-topic query
```

### Metadata Filtering

The vector store supports filtering by:
- `test_types`: `["K"]`, `["A", "K"]`, `["P"]` etc.
- `remote_testing`: `yes` / `no`
- `adaptive_irt`: `yes` / `no`

### Circuit Breaker

Both LLM calls and vector store queries are protected by circuit breakers:
- **Failure threshold**: 5 failures → OPEN
- **Reset timeout**: 60s → HALF_OPEN → probe → CLOSED
- **Fallback**: Graceful error message instead of cascading failures

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | (required) | Gemini API key |
| `LOG_LEVEL` | `INFO` | Logging level |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | ChromaDB storage path |
| `CATALOG_JSON_PATH` | `./data/shl_catalog.json` | Catalog data file |
| `CB_FAILURE_THRESHOLD` | `5` | Circuit breaker failure count |
| `CB_RESET_TIMEOUT` | `60.0` | Circuit breaker reset time (seconds) |

---

## Deployment (Render / Railway)

1. Set `GOOGLE_API_KEY` as an environment variable in your hosting platform
2. Set start command to: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
3. The `data/` directory with `shl_catalog.json` must be included in your repo or mounted as a volume
4. First /health call allows up to 2 minutes for cold start (index building)

---

## Evaluation Notes

- **Schema compliance**: Every response includes `reply`, `recommendations` (null or array), and `end_of_conversation`
- **Recall@10**: Semantic search retrieves 15 candidates; LLM narrows to 1–10 grounded in catalog
- **Hallucination guard**: All URLs validated against `shl.com`; catalog-only items returned
- **Turn cap**: API rejects requests with >8 messages (HTTP 400)
- **Off-topic refusal**: Explicit refuse node in agent graph
