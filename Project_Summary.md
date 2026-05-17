# SHL Assessment Advisor — Project Summary

**Stack:** FastAPI · LangGraph · Gemini 1.5 Flash (via OpenRouter) · ChromaDB · sentence-transformers · BM25

---

## Problem

Hiring managers describe roles in natural language ("I need something for a senior Java developer who works with stakeholders"), but SHL's catalog contains hundreds of discrete assessments with specific codes, types, and URLs. The goal is a conversational agent that maps vague intent to a grounded shortlist of catalog items — without hallucinating names, types, or URLs.

---

## Architecture

### Agent Graph (LangGraph)

The agent is a finite-state machine with seven nodes:

```
classify → [clarify | retrieve → rerank → recommend | compare | refuse] → END
```

**classify** runs on every turn. It analyses the full conversation history and routes the turn to the appropriate handler:
- `clarify` — too little context; ask one focused question
- `retrieve` — enough context; fetch candidates from the catalog
- `compare` — user wants to understand differences between two assessments
- `refuse` — query outside SHL scope (legal, compliance, general HR advice)
- `confirm` — user accepts the current shortlist; close the conversation

This separation prevents the model from accidentally recommending when it should be asking questions, and vice versa.

### Retrieval Pipeline

The retrieval flow is a three-stage hybrid search:

1. **Context extraction** — the classify node also extracts `job_levels` and `assessment keys` (e.g. Knowledge & Skills, Personality & Behavior, Simulations) from the conversation. These are used as pre-filters.

2. **Per-key hybrid search** — for each assessment type key, the system runs a ChromaDB semantic search (cosine similarity on `all-MiniLM-L6-v2` embeddings) and a BM25Okapi lexical search in parallel. The two ranked lists are fused using **Reciprocal Rank Fusion (RRF)** with weights 0.6 semantic / 0.4 lexical. This captures both semantic similarity ("nursing assessment for hospital") and exact phrase matching ("HIPAA compliance").

3. **Cross-encoder reranking** — a local `ms-marco-MiniLM-L-6-v2` cross-encoder scores each candidate against the user's original query in a single forward pass (<100ms). The LLM then makes the final selection from this reranked pool.

4. **recommend node** — the LLM receives the conversation history, the full retrieved candidate set grouped by assessment type, and produces 1–10 entity IDs as a JSON block. IDs are validated against the in-memory catalog before being returned.

### Two-Channel Response

Every response has two parts:
- **Prose reply** (1–3 sentences) — a confident consultant summary, no bullet points or URLs
- **JSON block** — `recommended_ids` array and `end_of_conversation` flag

---

## Prompt Design

The system prompt encodes nine hard rules enforced at the model level:
1. Only recommend from catalog — never invent names or URLs
2. Refuse out-of-scope queries immediately
3. Clarify before recommending — one question per turn
4. Honor in-place refinements without restarting
5. Acknowledge catalog gaps honestly
6. Repeat the shortlist as a table on confirmation
7. Distinguish instruments from reports
8. Respect language/regional variants when specified
9. Match seniority to test difficulty level

The classify prompt uses few-shot JSON examples to ensure reliable parsing. The recommend prompt instructs the LLM to output a balanced battery (not just technical tests) — including personality for roles with stakeholder interaction, and cognitive ability for selection use cases.

---

## Evaluation

An offline evaluation harness (`scripts/eval_recall.py`) runs all 30 scenarios from `evaluation_dataset.json` against the live `/chat` API and measures **Recall@10**:

```
Recall@K = (Number of relevant assessments in top K) / (Total relevant assessments)
```

For each scenario, the harness sends the full conversation history, extracts `recommended_ids` from the JSON block in the reply, and matches them against the ground-truth `expected_assessment_ids` by name→entity_id lookup. The result is a per-scenario recall score and a mean across all 30.

The `evaluation_dataset.json` was built by manually tracing each of the 30 assignment scenarios to their expected assessment IDs, covering diverse use cases: senior leadership, Rust engineers, contact centre agents, graduates, plant operators, healthcare admin, full-stack engineers, DevOps, project managers, and more.

---

## What Did Not Work

**LLM-only classification** initially produced unreliable intent labels because the prompt relied on free-text reasoning. Switching to explicit few-shot JSON examples in the classify prompt made intent parsing robust.

**Reranking via a second LLM call** was the original design but timed out on Render's cold-start constraints (~90s for 30 candidates with a hosted model). Replacing it with a local cross-encoder brought reranking to under 100ms.

**Direct web scraping of SHL.com** was unreliable — the catalog pages use dynamic rendering. A seeded `shl_catalog.json` with 55 known products is the primary data source; the scraper scripts serve as a refresh mechanism.

**Rule-based keyword classification** (e.g. "hire" → recommend, "compare" → compare) failed on paraphrased inputs. LLM-based intent classification handles natural variation in how users phrase the same intent.

---

## Deployment

A `render.yaml` Blueprint file configures the service for Render's free tier. Key settings:
- `HEALTHCHECK --start-period=300s` accommodates the first-start ChromaDB model download
- `OPENROUTER_API_KEY` and `GOOGLE_API_KEY` are set as secrets in the Render dashboard — never committed to source
- The `data/chroma_db/` directory is ephemeral on Render's free tier; the index rebuilds on each cold start from the catalog JSON
