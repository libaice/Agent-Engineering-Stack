# Agent Engineering Stack

A build-in-public laboratory for production-grade AI Agent engineering. This repository implements a progressive, step-by-step technical stack for Retrieval-Augmented Generation (RAG) and Agentic architectures. 

It starts with foundational PDF parsing and slides up to hybrid search, cross-encoder reranking, sufficiency guardrails, query translation, tool registries, and state management frameworks suitable for LangGraph.

---

##  Project Structure

The project code is organized under the `src/` directory:

```
src/
├── rag/          # Progressive RAG pipeline steps (01 to 16)
├── tool/         # Central tool registry and execution routers
└── state/        # LangGraph-compatible state definitions
```

---

##  Progressive RAG Stack (`src/rag/`)

Each step in this module introduces a new layer of complexity, reflecting the lifecycle of engineering a production RAG system:

| Step | Script / Component | Description |
|:---|:---|:---|
| **01** | [Step01_pdf_chunk_demo.py](src/rag/Step01_pdf_chunk_demo.py) | **Document Parsing & Chunking**: Loads PDF files using `fitz` (PyMuPDF) and chunks the text using a sliding-window strategy with character overlap. |
| **02** | [Step02_vector_search_demo.py](src/rag/Step02_vector_search_demo.py) | **Dense Retrieval (Vector Search)**: Normalizes and embeds chunks using `SentenceTransformer` (`BAAI/bge-small-zh-v1.5`) and indexes them in a memory-based FAISS flat index (`IndexFlatIP`). |
| **03** | [Step03_rag_answer_demo.py](src/rag/Step03_rag_answer_demo.py) | **Basic Generation & Prompting**: Synthesizes responses using DeepSeek LLM (`deepseek-v4-pro`), formatting retrieved chunks as evidence and enforcing strict citation rules. |
| **04** | [Step04_rag_eval_demo.py](src/rag/Step04_rag_eval_demo.py) | **Evaluation Suite**: Runs automated verification of hit rate, correctness of expected terms, citation existence, refusal capabilities, and latency on evaluation cases stored in `evals/cases.yaml`. |
| **05** | [Step05_ingest_demo.py](src/rag/Step05_ingest_demo.py) & [Step05_query_demo.py](src/rag/Step05_query_demo.py) | **Persistent Storage**: Serializes document chunks to JSON and writes the FAISS vector index to disk (`storage/faiss.index`) for decoupled ingestion and query serving. |
| **06** | [Step06_ingest_multi_docs.py](src/rag/Step06_ingest_multi_docs.py) | **Multi-Document & Multi-Format Ingestion**: Orchestrates discovery of multiple files under `data/docs/` and indexes different document types using format-specific logic. |
| **07** | [Step07_excel_query_demo.py](src/rag/Step07_excel_query_demo.py) | **Structured Document Parsing**: Uses Pandas to read Excel sheets row-by-row, converting tabular records into text-chunk representations (`Column: Value`) for semantic retrieval. |
| **08** | [Step08_hybrid_search_demo.py](src/rag/Step08_hybrid_search_demo.py) | **Hybrid Search**: Combines BM25 lexical retrieval (via `jieba` tokenization and `rank_bm25`) and dense vector search, applying Min-Max normalization and weighted score fusion. |
| **09** | [Step09_rerank_demo.py](src/rag/Step09_rerank_demo.py) | **Cross-Encoder Reranking**: Uses `BAAI/bge-reranker-base` to evaluate query-document pairs, addressing bi-encoder representation limits to boost retrieval precision. |
| **10** | [Step10_evidence_check_demo.py](src/rag/Step10_evidence_check_demo.py) | **Gating Guardrails**: Implements an LLM-based sufficiency checker that evaluates if retrieved snippets are enough to resolve the query, returning JSON validation. |
| **11** | [Step11_rag_safe_answer_demo.py](src/rag/Step11_rag_safe_answer_demo.py) | **Safe Refusal RAG**: Plugs in the sufficiency check to bypass generation and gracefully refuse to answer questions when evidence is lacking, preventing hallucinations. |
| **12** | [Step12_structured_answer_demo.py](src/rag/Step12_structured_answer_demo.py) | **Structured Outputs**: Parses structured model outputs via Pydantic model validation, checking citation boundaries, confidence levels, and missing context. |
| **13** | [Step13_query_rewrite_demo.py](src/rag/Step13_query_rewrite_demo.py) | **Query Translation (Rewrite & Sub-queries)**: Uses an LLM to rewrite vague queries into standalone questions and generate multiple sub-queries for broader semantic coverage. |
| **14** | [Step14_query_rewrite_rag_demo.py](src/rag/Step14_query_rewrite_rag_demo.py) | **Multi-Query Retrieval**: Multi-threads or loops retrieval for multiple rewritten sub-queries, merging candidates, and deduplicating chunks by their unique identifiers. |
| **15** | [Step15_query_decomposition_demo.py](src/rag/Step15_query_decomposition_demo.py) | **Query Decomposition**: Breaks down multi-dimensional or complex compound questions into sequenced, independent sub-questions with structural metadata. |
| **16** | [Step16_decomposed_rag_demo.py](src/rag/Step16_decomposed_rag_demo.py) | **Sub-query Synthesis & Aggregation**: Executes step-by-step retrieval and structured QA on each decomposed sub-question, then synthesizes a final answer. |

---

## Agent Integration (`src/tool/`)

To wrap RAG tasks in agent environments, the repository features tool-handling structures:

- **`ToolContext` & `ToolRegistry` ([Tool01_tools_demo.py](src/tool/Tool01_tools_demo.py))**:
  Encapsulates shared dependencies (like search stores and rerankers) in a `ToolContext`. The `ToolRegistry` defines standardized schemas, coordinates argument mapping, and exposes capabilities (e.g., `retrieve_documents` and `answer_from_documents`) as callable functions.

---

##  Agent State Management (`src/state/`)

For production architectures involving loop control, multi-turn dialogue, or orchestration frameworks like LangGraph:

- **`RAGAgentState` ([AgentState.py](src/state/AgentState.py))**:
  Defines a structured type tracking the agent's workflow state, including:
  - **Inputs & Intent**: original query, intent mapping, rewritten search queries, and missing context signals.
  - **Intermediate Results**: candidate chunks, selected evidence, and tool execution traces (with list operators).
  - **Outputs & Guardrails**: answerability indicators, confidence ratings, structured citations, and target response.
  - **Control Flow**: retry counters, error logs, and state transition flags.

---

##  Getting Started

### 1. Prerequisites & Environment
Ensure you have [uv](https://github.com/astral-sh/uv) installed. Install project dependencies and register the local package in editable mode:

```bash
uv pip install -e .
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory:

```env
DEEPSEEK_API_KEY="your-deepseek-api-key"
```

### 3. Data Ingestion
Place your raw document files inside `data/docs/` (e.g., PDFs, Excel files), then compile and populate the vector store:

```bash
uv run src/rag/Step06_ingest_multi_docs.py
```

### 4. Running the Demos
Run any script from the progressive RAG pipeline or tool modules:

```bash
uv run src/rag/Step16_decomposed_rag_demo.py
```