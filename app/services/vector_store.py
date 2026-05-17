"""
Vector Store Service — Hybrid Search (ChromaDB Semantic + BM25 Lexical).

Each SHL catalog item is converted to a fluent natural-language sentence
(`_item_to_natural_sentence`) before being embedded by the sentence-transformer
model (all-MiniLM-L6-v2). This dense vector is stored in a persistent ChromaDB
collection. An in-memory BM25Okapi index is rebuilt from the same sentences on
every startup for lexical / exact-phrase matching.

At query time, the `search()` method runs both retrievers in parallel and fuses
their ranked results using Reciprocal Rank Fusion (RRF) with configurable
semantic/lexical weights (default 60%/40%).

Metadata stored per item:
  - entity_id, name, url
  - test_types: comma-separated single-letter codes ("K", "A,P")
  - job_levels: pipe-delimited string ("|Manager|Director|") for $contains filtering
  - duration, remote_testing, adaptive_irt, languages, description
"""
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from app.core.config import get_settings
from app.core.circuit_breaker import vector_store_circuit_breaker

logger = logging.getLogger(__name__)

# Metadata filter key constants
FILTER_TEST_TYPE = "test_types"    # e.g. "K" or "P"
FILTER_REMOTE = "remote_testing"   # "yes" / "no" / "unknown"
FILTER_ADAPTIVE = "adaptive_irt"   # "yes" / "no" / "unknown"

# Map full type labels to single-letter codes for filtering
CODE_MAP = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

# Reverse map for readable names in sentences
CODE_TO_LABEL = {v: k for k, v in CODE_MAP.items()}


def _item_to_natural_sentence(item: Dict) -> str:
    """
    Convert a catalog item dict into a fluent natural-language sentence for embedding.

    Every metadata field (name, duration, job levels, test types, description,
    languages, remote, adaptive) is woven into prose so the dense vector captures
    full semantics rather than bare key-value structure.
    """
    name = item.get("name", "Unknown Assessment")
    duration = item.get("duration", item.get("duration_raw", "unknown duration"))
    
    job_levels = item.get("job_levels_raw", "")
    if not job_levels:
        jl_list = item.get("job_levels", [])
        job_levels = ", ".join(jl_list) if isinstance(jl_list, list) else str(jl_list)
        
    keys = item.get("keys", item.get("test_types", []))
    if isinstance(keys, list):
        keys = ", ".join(keys)
        
    description = item.get("description", "")
    
    languages = item.get("languages_raw", "")
    if not languages:
        lang_list = item.get("languages", [])
        languages = ", ".join(lang_list) if isinstance(lang_list, list) else str(lang_list)
        
    remote = item.get("remote", item.get("remote_testing", "unknown"))
    adaptive = item.get("adaptive", item.get("adaptive_irt", "unknown"))
    
    extra_details = []
    if languages:
        extra_details.append(f"Available in {languages.strip().rstrip(',')}.")
    if remote == "yes":
        extra_details.append("Supports remote testing.")
    elif remote == "no":
        extra_details.append("Does not support remote testing.")
    if adaptive == "yes":
        extra_details.append("Uses adaptive testing.")

    extra_str = " " + " ".join(extra_details) if extra_details else ""

    return (
        f"{name} is a {duration} assessment "
        f"for {job_levels} covering {keys}. "
        f"{description}{extra_str}"
    )


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokeniser for BM25."""
    return re.findall(r"\w+", text.lower())


class VectorStoreService:
    """
    Manages a persistent ChromaDB collection of SHL catalog items plus an
    in-memory BM25 index.  Searches combine both signals via Reciprocal
    Rank Fusion (RRF) to get the best of semantic + exact-phrase matching.
    """

    COLLECTION_NAME = "shl_catalog"
    # RRF constant (standard value from literature)
    RRF_K = 60
    # Weight balance: 0.0 = pure BM25, 1.0 = pure semantic
    SEMANTIC_WEIGHT = 0.6
    BM25_WEIGHT = 0.4

    def __init__(self):
        self.settings = get_settings()
        self._client: Optional[chromadb.ClientAPI] = None
        self._collection = None
        self._embedder: Optional[SentenceTransformer] = None
        self._catalog: List[Dict] = []
        # BM25 index (rebuilt in-memory on every startup)
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_docs: List[Dict] = []  # parallel list to BM25 corpus

    def _get_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            logger.info(f"Loading embedding model: {self.settings.embedding_model}")
            self._embedder = SentenceTransformer(self.settings.embedding_model)
        return self._embedder

    def _get_client(self) -> chromadb.ClientAPI:
        if self._client is None:
            persist_dir = Path(self.settings.chroma_persist_dir)
            persist_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Connecting to ChromaDB at {persist_dir}")
            self._client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    def _get_collection(self):
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def load_catalog(self) -> List[Dict]:
        """Load catalog JSON from disk."""
        catalog_path = Path(self.settings.catalog_json_path)
        if not catalog_path.exists():
            logger.error(f"Catalog JSON not found at {catalog_path}")
            return []
        with open(catalog_path, "r", encoding="utf-8") as f:
            self._catalog = json.load(f)
        logger.info(f"Loaded {len(self._catalog)} items from catalog JSON")
        return self._catalog

    # ─────────────────────────── Index Building ─────────────────────────── #

    def _build_bm25_index(self, catalog: List[Dict], doc_texts: List[str]) -> None:
        """Build the in-memory BM25 index from natural-language doc texts."""
        tokenized_corpus = [_tokenize(text) for text in doc_texts]
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._bm25_docs = list(catalog)
        logger.info(f"BM25 index built with {len(tokenized_corpus)} documents")

    def build_index(self, force: bool = False) -> None:
        """
        Build (or rebuild) both indexes from the catalog JSON.

        ChromaDB vector index: skipped if already populated unless force=True.
        BM25 lexical index:    always rebuilt (in-memory, fast, ~1s for 400 items).

        Args:
            force: If True, deletes and recreates the ChromaDB collection.
        """
        collection = self._get_collection()
        existing_count = collection.count()

        catalog = self._catalog or self.load_catalog()
        if not catalog:
            raise RuntimeError("Catalog is empty — run scraper first.")

        # Always rebuild BM25 (in-memory, fast)
        need_chroma_rebuild = existing_count == 0 or force

        if not need_chroma_rebuild:
            logger.info(
                f"ChromaDB already has {existing_count} items — skipping vector rebuild. "
                "Use force=True to rebuild.  Building BM25 only."
            )

        logger.info(f"Building index for {len(catalog)} items...")

        # Prepare documents, embeddings, and metadata
        documents = []
        metadatas = []
        ids = []

        for i, item in enumerate(catalog):
            # Catalog JSON uses 'keys' for test types (e.g. ['Knowledge & Skills'])
            # Fall back to 'test_types' for backwards compatibility
            test_types_list = item.get("keys", item.get("test_types", []))

            # ── Natural sentence document for embedding ──────────────────
            doc_text = _item_to_natural_sentence(item)

            code_list = [CODE_MAP.get(t.strip(), t.strip()[:1]) for t in test_types_list]

            # Job levels — store as pipe-delimited for safe $contains matching
            job_levels_list = item.get("job_levels", [])
            if isinstance(job_levels_list, list):
                job_levels_str = "|" + "|".join(job_levels_list) + "|" if job_levels_list else ""
            else:
                job_levels_str = ""

            # Metadata for filtering
            metadata = {
                "entity_id": str(item.get("entity_id", "")),
                "name": item.get("name", ""),
                "url": item.get("link", item.get("url", "")),
                "duration": item.get("duration", ""),
                "remote_testing": item.get("remote", item.get("remote_testing", "unknown")),
                "adaptive_irt": item.get("adaptive", item.get("adaptive_irt", "unknown")),
                "languages": item.get("languages_raw", str(item.get("languages", "")))[:500],
                "description": item.get("description", "")[:1000],
            }
            # Store as lists for ChromaDB $contains filtering (must be non-empty)
            if code_list:
                metadata["test_types"] = code_list
            if isinstance(job_levels_list, list) and job_levels_list:
                metadata["job_levels"] = job_levels_list

            doc_id = f"shl_{i:04d}"
            documents.append(doc_text)
            metadatas.append(metadata)
            ids.append(doc_id)

        # ── Build BM25 index (always) ────────────────────────────────────
        self._build_bm25_index(catalog, documents)

        # ── Build ChromaDB index (conditional) ───────────────────────────
        if need_chroma_rebuild:
            # Compute embeddings in batches
            embedder = self._get_embedder()
            batch_size = 64
            all_embeddings = []
            for start in range(0, len(documents), batch_size):
                batch = documents[start : start + batch_size]
                embs = embedder.encode(batch, show_progress_bar=False).tolist()
                all_embeddings.extend(embs)
                logger.info(f"Embedded {min(start + batch_size, len(documents))}/{len(documents)}")

            # Clear existing and add new
            if existing_count > 0:
                collection.delete(where={"name": {"$ne": "__nonexistent__"}})

            collection.add(
                documents=documents,
                embeddings=all_embeddings,
                metadatas=metadatas,
                ids=ids,
            )
            logger.info(f"ChromaDB vector index built with {collection.count()} items")

    # ─────────────────────────── Filtering ──────────────────────────────── #

    def _build_where_filter(
        self,
        test_types: Optional[List[str]] = None,
        job_levels: Optional[List[str]] = None,
        remote_only: Optional[bool] = None,
        adaptive_only: Optional[bool] = None,
    ) -> Optional[Dict]:
        """
        Build a ChromaDB `where` clause for metadata pre-filtering.

        job_levels uses pipe-delimited format: |Manager| stored as "|Manager|Director|"
        so $contains "|Manager|" matches exactly without partial-word false positives.

        Supported ChromaDB operators used: $eq, $contains, $and, $or.
        """
        conditions = []

        if test_types:
            # Filter items whose test_types string CONTAINS at least one requested type
            type_conditions = [
                {"test_types": {"$contains": t}} for t in test_types
            ]
            if len(type_conditions) == 1:
                conditions.append(type_conditions[0])
            else:
                conditions.append({"$or": type_conditions})

        if job_levels:
            # Filter items whose job_levels list CONTAINS at least one requested level
            level_conditions = [
                {"job_levels": {"$contains": level}} for level in job_levels
            ]
            if len(level_conditions) == 1:
                conditions.append(level_conditions[0])
            else:
                conditions.append({"$or": level_conditions})

        if remote_only is True:
            conditions.append({"remote_testing": {"$eq": "yes"}})

        if adaptive_only is True:
            conditions.append({"adaptive_irt": {"$eq": "yes"}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    # ─────────────────────────── BM25 Search ────────────────────────────── #

    def _bm25_search(
        self,
        query: str,
        n_results: int = 30,
        test_types: Optional[List[str]] = None,
        job_levels: Optional[List[str]] = None,
        remote_only: Optional[bool] = None,
        adaptive_only: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        BM25 lexical search over the catalog with optional in-memory metadata filtering.

        Tokenises the query, scores all docs, and applies the same job_levels /
        test_types / remote / adaptive filters as the ChromaDB path (in Python).
        Returns items with bm25_score and bm25_rank fields appended.
        """
        if self._bm25 is None:
            logger.warning("BM25 index not built — skipping lexical search")
            return []

        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)

        # Build (index, score) pairs and sort descending
        scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results = []
        for rank, (idx, score) in enumerate(scored):
            if score <= 0:
                break  # BM25 scores of 0 mean no match at all
            if len(results) >= n_results:
                break

            item = self._bm25_docs[idx]

            # Apply metadata filters in-memory
            if test_types:
                item_types = item.get("keys", item.get("test_types", []))
                item_codes = [CODE_MAP.get(t.strip(), t.strip()[:1]) for t in item_types]
                if not any(t in item_codes for t in test_types):
                    continue

            if job_levels:
                item_levels = item.get("job_levels", [])
                if isinstance(item_levels, list):
                    if not any(jl in item_levels for jl in job_levels):
                        continue
                else:
                    continue  # skip items with no job_levels data

            if remote_only is True:
                remote_val = item.get("remote", item.get("remote_testing", "unknown"))
                if remote_val != "yes":
                    continue

            if adaptive_only is True:
                adaptive_val = item.get("adaptive", item.get("adaptive_irt", "unknown"))
                if adaptive_val != "yes":
                    continue

            results.append({
                "entity_id": str(item.get("entity_id", "")),
                "name": item.get("name", ""),
                "url": item.get("link", item.get("url", "")),
                "test_types": [CODE_MAP.get(t.strip(), t.strip()[:1])
                               for t in item.get("keys", item.get("test_types", []))],
                "duration": item.get("duration", ""),
                "remote_testing": item.get("remote", item.get("remote_testing", "unknown")),
                "adaptive_irt": item.get("adaptive", item.get("adaptive_irt", "unknown")),
                "languages": item.get("languages_raw", str(item.get("languages", "")))[:500],
                "description": item.get("description", "")[:1000],
                "bm25_score": float(score),
                "bm25_rank": rank,
            })

        logger.info(f"BM25 search returned {len(results)} results for query: {query!r}")
        return results

    # ─────────────────────────── Hybrid Search (RRF) ────────────────────── #

    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        n_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Fuse semantic and BM25 result lists using weighted Reciprocal Rank Fusion.

        RRF formula:
          score(d) = SEMANTIC_WEIGHT / (RRF_K + rank_sem(d))
                   + BM25_WEIGHT    / (RRF_K + rank_bm25(d))

        Items found by only one retriever still receive a score from that source.
        Default weights: semantic=0.6, BM25=0.4; RRF_K=60 (standard literature value).
        """
        k = self.RRF_K
        w_s = self.SEMANTIC_WEIGHT
        w_b = self.BM25_WEIGHT

        # entity_id → {item_data, rrf_score, semantic_score, bm25_score}
        fused: Dict[str, Dict[str, Any]] = {}

        # Score semantic results
        for rank, item in enumerate(semantic_results):
            eid = item.get("entity_id", "")
            if not eid:
                continue
            rrf_score = w_s / (k + rank + 1)  # rank is 0-indexed, RRF uses 1-indexed
            fused[eid] = {
                **item,
                "rrf_score": rrf_score,
                "semantic_rank": rank + 1,
                "bm25_rank": None,
            }

        # Score BM25 results and merge
        for rank, item in enumerate(bm25_results):
            eid = item.get("entity_id", "")
            if not eid:
                continue
            bm25_rrf = w_b / (k + rank + 1)

            if eid in fused:
                # Item found by both — combine scores
                fused[eid]["rrf_score"] += bm25_rrf
                fused[eid]["bm25_rank"] = rank + 1
                fused[eid]["bm25_score"] = item.get("bm25_score", 0)
            else:
                # Item found only by BM25
                fused[eid] = {
                    **item,
                    "rrf_score": bm25_rrf,
                    "semantic_rank": None,
                    "bm25_rank": rank + 1,
                    "score": 0,  # no semantic score
                }

        # Sort by RRF score descending
        ranked = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)

        # Log fusion diagnostics for top results
        for i, r in enumerate(ranked[:5]):
            logger.debug(
                f"  RRF #{i+1}: {r['name']} "
                f"(rrf={r['rrf_score']:.4f}, sem_rank={r.get('semantic_rank')}, "
                f"bm25_rank={r.get('bm25_rank')})"
            )

        return ranked[:n_results]

    # ─────────────────────────── Main Search ────────────────────────────── #

    def search(
        self,
        query: str,
        n_results: int = 10,
        test_types: Optional[List[str]] = None,
        job_levels: Optional[List[str]] = None,
        remote_only: Optional[bool] = None,
        adaptive_only: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search entry point: semantic (ChromaDB) + lexical (BM25) fused via RRF.

        Called once per assessment key in retrieve_node, with the key-specific
        query and test_type code. Results from all keys are merged upstream.

        Args:
            query:         Natural-language query tailored to the assessment key.
            n_results:     Max items to return (10 per key in the pipeline).
            test_types:    Single-letter codes to filter by, e.g. ["K"], ["A"].
            job_levels:    Job level strings to filter by, e.g. ["Manager"].
            remote_only:   If True, restrict to remote-testable items.
            adaptive_only: If True, restrict to adaptive/IRT items.

        Returns:
            List of matching catalog items with merged metadata and fused RRF scores.
        """
        def _do_search():
            # ── 1. Semantic search via ChromaDB ──────────────────────────
            semantic_results = []
            try:
                collection = self._get_collection()
                embedder = self._get_embedder()

                query_embedding = embedder.encode(query, show_progress_bar=False).tolist()

                where = self._build_where_filter(test_types, job_levels, remote_only, adaptive_only)

                # Fetch more candidates than needed so RRF has a good pool
                n_semantic = min(n_results * 3, collection.count() or 1)

                kwargs: Dict[str, Any] = {
                    "query_embeddings": [query_embedding],
                    "n_results": n_semantic,
                    "include": ["documents", "metadatas", "distances"],
                }
                if where:
                    kwargs["where"] = where

                results = collection.query(**kwargs)

                metadatas = results.get("metadatas", [[]])[0]
                distances = results.get("distances", [[]])[0]

                for meta, dist in zip(metadatas, distances):
                    # handle both old (string) and new (list) metadata formats during transition
                    test_types_meta = meta.get("test_types", [])
                    if isinstance(test_types_meta, str):
                        test_types_meta = test_types_meta.split(",")

                    semantic_results.append({
                        "entity_id": meta.get("entity_id", ""),
                        "name": meta.get("name", ""),
                        "url": meta.get("url", ""),
                        "test_types": test_types_meta,
                        "duration": meta.get("duration", ""),
                        "remote_testing": meta.get("remote_testing", "unknown"),
                        "adaptive_irt": meta.get("adaptive_irt", "unknown"),
                        "languages": meta.get("languages", ""),
                        "description": meta.get("description", ""),
                        "score": 1 - dist,  # cosine distance → similarity
                    })
            except Exception as e:
                logger.error(f"Semantic search failed: {e}")

            # ── 2. BM25 lexical search ───────────────────────────────────
            bm25_results = self._bm25_search(
                query=query,
                n_results=n_results * 3,
                test_types=test_types,
                job_levels=job_levels,
                remote_only=remote_only,
                adaptive_only=adaptive_only,
            )

            # ── 3. Fuse with RRF ────────────────────────────────────────
            if semantic_results and bm25_results:
                fused = self._reciprocal_rank_fusion(
                    semantic_results, bm25_results, n_results=n_results
                )
                logger.info(
                    f"Hybrid search returned {len(fused)} results for query: {query!r} "
                    f"(semantic={len(semantic_results)}, bm25={len(bm25_results)})"
                )
                return fused
            elif semantic_results:
                # Assign synthetic scores for consistent output schema
                for rank, item in enumerate(semantic_results[:n_results]):
                    item["rrf_score"] = item.get("score", 0)
                    item["semantic_rank"] = rank + 1
                    item["bm25_rank"] = None
                logger.info(
                    f"Semantic-only search returned {len(semantic_results[:n_results])} "
                    f"results (BM25 returned 0)"
                )
                return semantic_results[:n_results]
            elif bm25_results:
                # Assign synthetic scores for consistent output schema
                for rank, item in enumerate(bm25_results[:n_results]):
                    item["rrf_score"] = item.get("bm25_score", 0)
                    item["semantic_rank"] = None
                logger.info(
                    f"BM25-only search returned {len(bm25_results[:n_results])} "
                    f"results (semantic returned 0)"
                )
                return bm25_results[:n_results]
            else:
                logger.warning(f"Both semantic and BM25 returned 0 results for: {query!r}")
                return []

        return vector_store_circuit_breaker.call(_do_search)

    def get_all(self) -> List[Dict]:
        """Return all catalog items (used as fallback when search returns nothing)."""
        return self._catalog or self.load_catalog()

    def get_by_entity_id(self, entity_id: str) -> Optional[Dict]:
        """
        Exact entity_id lookup against the in-memory catalog.

        Used by recommend_node to resolve LLM-returned entity IDs back to full
        catalog metadata (name, url, test_types, etc.).
        Ensures the catalog is loaded before searching.
        """
        catalog = self._catalog or self.load_catalog()
        for item in catalog:
            if str(item.get("entity_id", "")) == str(entity_id):
                return item
        return None


# Singleton instance
_vector_store: Optional[VectorStoreService] = None


def get_vector_store() -> VectorStoreService:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreService()
    return _vector_store
