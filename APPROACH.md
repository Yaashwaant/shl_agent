# Approach Document — SHL Assessment Advisor

**Candidate:** Final Year B.Tech Student  
**Assignment:** AI Intern Take-Home — Conversational SHL Assessment Advisor  
**Stack:** FastAPI · LangGraph · Gemini 1.5 Flash · ChromaDB · sentence-transformers

---

## Problem Understanding

The core challenge is bridging the vocabulary gap between a hiring manager's vague intent ("I need something for a Java developer") and a precise SHL catalog selection. This requires:
1. **Understanding intent** across a multi-turn dialogue
2. **Grounding every recommendation** in the actual catalog
3. **Never hallucinating** assessment names or URLs

---

## Design Choices

### 1. Agent Orchestration: LangGraph

I chose LangGraph over a simple ReAct loop because the assessment dialog has distinct, predictable states:

```
classify → [clarify | retrieve → recommend | compare | refuse | confirm]
```

LangGraph's explicit graph structure makes each state testable and debuggable. The classify node runs first on every turn, determining the routing. This prevents the model from accidentally recommending when it should be asking questions.

**Key design decision**: The classify node explicitly prevents recommending on turn 1 for short/vague queries. This is enforced by code logic, not just prompt instructions.

### 2. Retrieval: ChromaDB + Semantic Search + Metadata Filtering

**Why ChromaDB**: Zero infrastructure overhead for a take-home project; persists to disk so the index survives restarts.

**Why sentence-transformers (`all-MiniLM-L6-v2`)**: Fast, local, no API cost. Adequate for the catalog size (~55 products).

**Metadata filtering** is implemented as a first-class feature. The classify node extracts structured context:
- `test_type_preferences`: ["K", "A"] — filters vector search to only those types
- `remote_required`: true/false — filters to remote-testable only
- `adaptive_required`: true/false — filters to adaptive/IRT items

This allows a query like "I need remote-testable knowledge tests for a Python developer" to bypass hundreds of irrelevant results before the LLM even sees them.

**Retrieval strategy**: Retrieve 15 candidates, let the LLM narrow to 1–10. This gives the LLM room to reason about fit, not just raw similarity.

### 3. LLM: Gemini 1.5 Flash

Free tier, fast, large context window. Flash vs Pro tradeoff: Flash is ~5x faster at the expense of some reasoning depth. For a structured-output task on a small catalog, Flash is sufficient.

**Prompt design**:
- System prompt emphasises: SHL-only, no fabrication, clarify before recommending
- Classify prompt uses few-shot JSON output format for reliable parsing
- Recommend prompt provides the retrieved catalog as context and asks for JSON block output
- Every recommendation URL is validated against `shl.com` before returning

### 4. Resilience: Circuit Breaker

Both the LLM client and ChromaDB are wrapped in circuit breakers:
- 5 failures → OPEN (fast-fail for 60 seconds)
- HALF_OPEN after timeout → one probe call → CLOSED on success

This prevents cascading failures under load and makes the service degrade gracefully (returns a service-unavailable message rather than hanging).

### 5. API Design

The spec mandates a stateless API (no server-side session). Every `/chat` call carries the full conversation history. The agent reconstructs context from the full history on every call using the classify+extract pass.

Turn cap (8) is enforced at the API layer (HTTP 400) before reaching the agent.

---

## Retrieval Strategy Details

```
User query
    │
    ▼
[Extract context from conversation history]
    │  job_title, skills, seniority, purpose, test_type_preferences
    │
    ▼
[Build semantic query string]
    │  "assessment for Senior Java Developer skills: Java, Spring, SQL"
    │
    ▼
[ChromaDB vector search + metadata filters]
    │  test_types ∈ {K}, remote_testing = "yes"  (if specified)
    │  n_results = 15
    │
    ▼
[LLM selection pass]
    │  Given 15 candidates + full conversation, select 1–10 best fits
    │
    ▼
[URL validation]
    │  All URLs must contain shl.com
    │
    ▼
[Return structured recommendations]
```

If the filtered search returns 0 results (e.g., no items tagged with requested types), it falls back to unfiltered semantic search. This prevents empty responses when the catalog data is sparse.

---

## What Didn't Work

1. **Rule-based classification**: I initially tried keyword matching for intent classification. It broke on paraphrased inputs ("tweak the list" vs "update recommendations"). Switched to LLM-based classification which handles natural variation.

2. **Single-pass LLM**: A single prompt asking the LLM to "classify AND recommend" created token pressure and unreliable JSON output. Separating classify and recommend into distinct nodes (with their own prompts) made output parsing reliable.

3. **Direct SHL.com scraping of table data**: The catalog uses dynamic rendering. The scraper falls back to link-following when the table parser finds nothing. A seeded JSON catalog is provided as the primary data source to avoid scraping fragility at evaluation time.

---

## Evaluation Approach

Tested against all 10 sample conversation traces:

| Trace | Behavior | Result |
|---|---|---|
| C1 | Senior leadership → OPQ32r | Pass |
| C2 | Rust engineer (no catalog match) → closest alternatives | Pass |
| C5 | Sales org audit → multi-assessment stack | Pass |
| C9 | Full-stack JD → iterative refinement over 7 turns | Pass |

**Behavior probes checked manually**:
- ✅ Refuses off-topic queries (legal advice, general HR)
- ✅ No recommendation on turn 1 for vague query
- ✅ Honors mid-conversation refinements (add/remove)
- ✅ Comparison answers grounded in catalog data
- ✅ Turn cap enforced (HTTP 400 at 9+ messages)
- ✅ All returned URLs are valid shl.com paths

---

## AI Tools Used

- **Gemini 1.5 Flash**: LLM backbone for all agent reasoning
- **sentence-transformers**: Local embedding generation (no API)
- **Antigravity (AI coding assistant)**: Assisted with boilerplate generation and project structure scaffolding. All design decisions, architecture choices, and evaluation were made independently.

---

## Possible Improvements (given more time)

1. **LLM caching**: Cache identical conversation + context hashes to reduce latency
2. **Reranker**: Add a cross-encoder reranker after initial vector retrieval for better precision
3. **Conversation memory compression**: Summarise long conversations to stay within token limits
4. **Streaming**: Implement SSE streaming for faster perceived response time
5. **Richer scraper**: Playwright-based scraper to handle JavaScript-rendered catalog tables
6. **Evaluation harness**: Automated Recall@10 measurement against the 10 sample traces
