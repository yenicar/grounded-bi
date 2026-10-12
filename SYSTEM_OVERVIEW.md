# Business Intelligence RAG — System Overview

## What This Is

**Business Intelligence RAG** is a local-first retrieval-augmented generation system purpose-built for corporate intelligence analysis. It ingests a corpus of documents (PDF, DOCX, XLSX and CSV; the current build is pointed at 65 public HR policy PDFs), and enables analysts to ask natural-language questions that are answered *exclusively* from the evidence in those documents — with full citations and confidence scoring.

The prompt tells the model to answer only from the retrieved evidence and to say plainly when that evidence is insufficient, and every answer lists the document and page it drew from. The model can still add details its sources do not support; the evaluation section of the README shows how often.

---

## What It Offers

### Core Capabilities

| Capability | Description |
|---|---|
| **Grounded Q&A** | Ask natural-language questions and receive answers backed by specific document evidence and page-level citations |
| **Hybrid Search** | Combines semantic vector search (ChromaDB) with BM25 keyword scoring via Reciprocal Rank Fusion (RRF), so exact terms and paraphrases both rank |
| **Parent-Child Retrieval** | Uses small, precise "child" chunks for search, then expands to large, context-rich "parent" chunks for answer generation — the "Small-to-Big" pattern |
| **Table-Aware Intelligence** | Extracts structured tables from PDFs into Markdown format and boosts table-bearing evidence when quantitative queries are detected |
| **Multi-Turn Chat** | Supports follow-up questions with LLM-powered query rewriting that resolves pronouns and implicit context from conversation history |
| **Automated Evaluation** | LLM-as-a-judge system that scores every answer for faithfulness (is it grounded?) and relevance (does it answer the question?) |
| **Web UI** | FastAPI + React app with Ask, Chat, Library and Evaluation pages, background indexing jobs with progress, and a query audit log |
| **PDF/PPTX Export** | Professional export of generated answers and source citations to presentation-ready formats |
| **Full Caching** | Every pipeline stage — ingestion, chunking, embedding, and query answers — is cached with content-hash invalidation |

### Technical Differentiators

These are the engineering decisions that separate this system from a typical "chat with your docs" tutorial:

1. **Reciprocal Rank Fusion (RRF):** Instead of relying on a single retrieval signal, the system independently ranks documents by semantic similarity AND keyword match, then fuses those rankings using RRF. This means a document that is semantically similar *and* contains exact business terms will rank higher than one that only matches on one dimension.

2. **Hierarchical Chunking with Parent Collapse:** The system doesn't just split documents into equal-sized chunks. It creates two levels: small children (220 tokens) optimized for search precision, and large parents (800 tokens) optimized for LLM context. At retrieval time, multiple child hits from the same parent are *collapsed* into a single parent context, preventing the LLM from seeing redundant information.

3. **Source Diversity Enforcement:** A per-document cap prevents any single document from dominating the context window. For quantitative queries, the system allows at most 2 chunks from the same source, forcing the LLM to cross-reference multiple documents — exactly how a human analyst works.

4. **Structured Table Extraction with Heuristic Filtering:** The system uses `pdfplumber` to detect grid-style tables in PDFs and converts them to Markdown. Crucially, it applies a heuristic filter (`_looks_like_structured_table`) that rejects false positives — paragraphs that pdfplumber incorrectly identifies as tables. This prevents "garbage tables" from contaminating the index.

5. **Pipeline Versioning and Automatic Rebuild:** Every stage of the pipeline is fingerprinted. When the ingestion logic changes (e.g., a new table extraction heuristic), the pipeline version string forces a full re-extraction. When chunking parameters change, only the chunking and downstream stages rebuild. This ensures the system never serves stale data.

---

## How It Works

### Architecture

┌──────────────────────────────────────────────────────────────────┐
│                    GroundedBI Analyst Console                    │
│      (React Dashboard + FastAPI Web API + bi-rag CLI)            │
└─────┬──────────────────┬──────────────────┬──────────────────────┘
      │                  │                  │
      ▼                  ▼                  ▼
┌─────────────┐    ┌─────────────┐    ┌───────────────────────────┐
│ Ingestion   │    │  Job Queue  │    │      Retrieval Engine     │
│ (Multi-fmt) │    │  (Async)    │    │ (Hybrid RRF + Metadata)   │
└──────┬──────┘    └──────┬──────┘    └─────────────┬─────────────┘
       │                  │                         │
       ▼                  ▼                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                        data/ directory                          │
│  raw_docs/ │ normalized_md/ │ chunks/ │ index/ │ audit/ │ export/   │
└─────────────────────────────────────────────────────────────────┘

### Data Pipeline

1. **Ingestion** (`bi-rag ingest`): PDFs are extracted to normalized Markdown. Tables are detected and appended as structured Markdown blocks.
2. **Chunking** (`bi-rag index`): Markdown is split into parent chunks (800 tokens) and child chunks (220 tokens) with sliding-window overlap.
3. **Indexing** (`bi-rag index`): Child chunks are embedded via Gemini and stored in ChromaDB. A BM25 keyword index and parent lookup table are built in parallel.
4. **Retrieval** (`bi-rag ask`): Queries trigger both semantic and keyword search. Results are fused, collapsed to parents, optionally table-boosted, and filtered for diversity.
5. **Generation**: The top-k parent chunks are assembled into a grounded prompt. The LLM generates an answer with a confidence rating.
6. **Evaluation** (`bi-rag eval`): The evaluation suite runs a JSON file of questions (16 in the recorded runs), scores each with both heuristics and an LLM judge, and produces a full report.

---

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| LLM Provider | Google Gemini (via `google-generativeai`) | Embedding and answer generation |
| Vector Store | ChromaDB (persistent, local) | Semantic similarity search |
| PDF Extraction | PyPDF2 + pdfminer.six + pdfplumber | Multi-backend text and table extraction |
| Tokenization | tiktoken (`cl100k_base`) | Accurate token counting for chunk sizing |
| Retry/Resilience | tenacity | Exponential backoff for API failures |
| Concurrency Safety | filelock | Race-condition prevention on cache files |
| Runtime | Python 3.12+ | Modern type hints and dataclasses |

---

## What Could Be Enhanced

### Tier 1 — Modernization & Graph Intelligence

| Enhancement | Why It Matters | Estimated Effort |
|---|---|---|
| **Graph-RAG Alpha** | Implement a lightweight graph bridge to resolve connections across policies that semantic search might miss. | 3-5 days |
| **Advanced Metadata Filtering** | Allow filtering by author, document type, and custom tags in addition to date. | 2 days |
| **Streaming WebSocket UI** | Full WebSocket integration for token-by-token streaming in the Ask and Chat pages. | 3 days |
| **Vector Store Migration** | Move from local ChromaDB to a managed vector database (pgvector, Weaviate) for multi-user deployment. | 1-2 weeks |

---

## Summary

GroundedBI combines hybrid retrieval (RRF, parent-child chunks, table-aware boosting) with a FastAPI and React app, background jobs and an audit log.

It is a local, single-workstation build. It has not been deployed or used by anyone outside the project, and the open items in the README come before any shared use.
