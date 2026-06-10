# RAG Agent Architecture Design Decisions

This document outlines the core architecture design decisions for the RAG Agent system.

---

## 1. Decision: Do not rely on vector-only retrieval

### Problem
A basic RAG system usually uses embeddings and vector similarity search. This works well for semantic similarity, but enterprise documents often contain exact identifiers, numbers, company names, contract terms, ticket IDs, stock symbols, API names, and table fields. Vector search can miss these exact-match signals.

### Decision
Use vector search as one retrieval channel, but not the only one.

### Why
Vector search is good at finding semantically similar chunks. However, it is less reliable for exact strings such as:
- contract IDs
- company names
- token symbols
- order numbers
- API endpoints
- exact dollar amounts
- spreadsheet field names

### Trade-offs
Adding non-vector retrieval increases system complexity, but improves recall and robustness.

### Alternatives
- Vector-only search
- BM25-only search
- SQL full-text search
- Elasticsearch / OpenSearch
- Qdrant hybrid search
- pgvector + Postgres full-text search

### When not to use
Vector-only can be enough for a small, clean, semantic-only corpus where users ask broad natural language questions.

### Interview talking point
I do not treat embedding search as a silver bullet. It is strong for semantic recall, but enterprise RAG also needs exact-match retrieval for IDs, numbers, names, and domain-specific terms.
*中文讲法：纯向量检索不是银弹。它适合语义相似，但对数字、ID、公司名、合约地址、字段名这些精确匹配不够稳。所以我不会只依赖 vector search。*

---

## 2. Decision: Use Hybrid Search with BM25 + Vector Search

### Problem
Different queries require different retrieval signals. Some are semantic, others are lexical.
For example:
- "What is the project price?" is semantic.
- "HIP-3 500K HYPE collateral" is lexical and exact-match heavy.

### Decision
Use BM25 and vector search together. Retrieve candidates from both channels, merge, deduplicate, and rerank.

### Why
BM25 improves exact keyword recall.  
Vector search improves semantic recall.  
Together they produce a more complete candidate set.

### Trade-offs
Hybrid search adds complexity:
- two retrieval channels
- score normalization or rank fusion
- deduplication
- more candidates to rerank
It can also increase latency.

### Alternatives
- Vector-only
- BM25-only
- Reciprocal Rank Fusion
- Elasticsearch hybrid
- Qdrant hybrid
- LlamaIndex / LangChain retrievers

### When not to use
If the corpus is tiny or queries are purely semantic, hybrid search may be unnecessary.

### Interview talking point
I use hybrid search because retrieval recall determines the upper bound of the whole RAG system. If the correct evidence does not enter the candidate set, rerank and generation cannot fix it.
*关键句：Retriever 决定上限，Reranker 优化排序，LLM 决定生成质量。*

---

## 3. Decision: Add a cross-encoder reranker after retrieval

### Problem
Hybrid retrieval returns candidates that are potentially relevant, but not always directly useful for answering the question. Some chunks may share keywords but not answer the question. Others may be semantically related but too vague.

### Decision
Use a reranker to score query-chunk pairs and select the final top-k evidence.

### Why
A reranker can directly compare the query and the candidate chunk. It is better at identifying whether a chunk actually helps answer the question.

### Trade-offs
Rerank improves precision but adds latency and compute cost. Candidate size must be controlled.

### Alternatives
- No rerank
- LLM-based rerank
- Embedding-only similarity
- RRF only
- Vendor rerank API
- Local cross-encoder reranker

### When not to use
If latency is extremely sensitive and retrieval precision is already high, rerank may be skipped.

### Interview talking point
Retrieval should be wide enough to avoid missing evidence; rerank narrows it down. I usually recall top-20 or top-50 candidates, then rerank to top-5 evidence.

---

## 4. Decision: Add answerability / evidence sufficiency check

### Problem
Retrievers always return top-k chunks, even when the corpus does not contain the answer. If the LLM is forced to answer, it may hallucinate.

### Decision
Before generating the final answer, check whether retrieved evidence is sufficient to answer the question.

### Why
Relevant evidence is not the same as sufficient evidence. A chunk may mention "pricing" but not contain the actual price.

### Trade-offs
Adding answerability check may require an extra LLM call and can introduce false refusals.

### Alternatives
- Rerank score threshold
- LLM judge
- Extract-then-answer
- Rule-based refusal
- Confidence calibration

### When not to use
For low-risk brainstorming or creative generation, answerability check is less important.

### Interview talking point
In enterprise RAG, saying "I don't know based on the current evidence" is often much safer than producing a confident but unsupported answer.
*中文讲法：相关不等于可回答。检索系统一定会返回 top-k，但这些 evidence 未必足够支持答案。所以我加 answerability，证据不足就拒答。*

---

## 5. Decision: Use structured output with Pydantic validation

### Problem
Free-form LLM output is hard to parse and unreliable for downstream systems. The frontend, API, eval harness, and workflow engine need stable fields such as answer, citations, answerable, confidence, and missing_information.

### Decision
Require the model to output JSON and validate it with Pydantic.

### Why
Structured output turns the LLM from a free-form text generator into a typed system component.

### Trade-offs
Structured output can make prompting stricter and may require retry logic when validation fails.

### Alternatives
- Free-form text
- JSON mode only
- Function calling
- PydanticAI
- Guardrails
- Instructor

### When not to use
For casual chat or creative writing, strict schemas may be unnecessary.

### Interview talking point
I use structured output because Agent systems need machine-consumable outputs, not just human-readable text. Validation also gives me a clean failure point.

---

## 6. Decision: Rewrite user questions before retrieval

### Problem
User questions are often vague, short, conversational, or contain references such as "this", "that", "it", or "the previous one". Raw queries are often poor retrieval queries.

### Decision
Add a query rewrite node before retrieval.

### Why
Query rewrite improves retrieval recall by turning natural user questions into standalone, retrieval-optimized queries.

### Trade-offs
It adds one LLM call and can accidentally change user intent if poorly prompted.

### Alternatives
- Use raw question directly
- Rule-based expansion
- HyDE
- Query expansion
- Multi-query generation

### When not to use
For clear keyword queries or latency-critical paths, raw search may be enough.

### Interview talking point
Many RAG failures are not embedding failures but query understanding failures. Query rewrite lets me fix the input before blaming retrieval.

---

## 7. Decision: Decompose multi-part questions into sub-questions

### Problem
Complex user questions often contain multiple information needs. A single retrieval pass may over-focus on one aspect and miss others.
Example: "What are the pricing, risks, and delivery timeline?"

### Decision
Decompose complex questions into sub-questions, retrieve evidence for each, then synthesize the final answer.

### Why
This improves recall, reduces missing sections, and makes evaluation more granular.

### Trade-offs
More LLM calls, more retrieval calls, more complexity.

### Alternatives
- Single broad retrieval
- LLM planner
- Sub-question query engine
- Agentic retrieval

### When not to use
Simple fact lookup should not be over-decomposed.

### Interview talking point
Query decomposition is the bridge from RAG to Agent workflow. It turns a complex question into a plan-execute-synthesize process.

---

## 8. Decision: Use LangGraph for stateful Agent workflow

### Problem
A simple chain is not enough for production Agent behavior. Real agents need branching, retries, memory, checkpoint, streaming, and human-in-the-loop.

### Decision
Use LangGraph to represent the Agent as a state machine.

### Why
LangGraph makes state, nodes, edges, conditional routing, checkpoint, streaming, and interrupt explicit.

### Trade-offs
It has more upfront complexity than a simple chain.

### Alternatives
- Plain Python orchestration
- LangChain AgentExecutor
- LlamaIndex workflows
- CrewAI
- AutoGen / Microsoft Agent Framework
- Temporal / Celery workflow

### When not to use
For a simple one-shot RAG endpoint, plain Python or LCEL may be enough.

### Interview talking point
I use LangGraph not because it is trendy, but because Agent engineering is state management. LangGraph gives me explicit state transitions, conditional edges, checkpoint, streaming, and human-in-the-loop.
*强表达：我把 Agent 理解成状态机：state_n + node/tool execution → state_{n+1}。这和区块链 state transition 很像。*

---

## 9. Decision: Make Agent state explicit

### Problem
In a simple chain, intermediate values are hidden in function calls. This makes debugging, retrying, checkpointing, and branching difficult.

### Decision
Use an explicit AgentState that stores question, rewritten_query, evidence, answer, tool_calls, steps, errors, retry_count, and status.

### Why
Explicit state makes the system observable, recoverable, and debuggable.

### Trade-offs
State design requires discipline. If too much is stored, state becomes bloated.

### Alternatives
- Local variables in a function
- Conversation history only
- External DB only
- Full event sourcing

### When not to use
For very simple synchronous workflows, explicit graph state may be overkill.

### Interview talking point
State is the Agent's runtime working memory. I keep raw input, transformed input, retrieval candidates, final evidence, answer, errors, and tool calls separate so I can locate failures precisely.

---

## 10. Decision: Persist workflow checkpoints

### Problem
Agent runs can be long-running, interrupted, or dependent on user approval. Without checkpointing, failures or pauses lose state.

### Decision
Use checkpointing keyed by thread_id.

### Why
Checkpoint enables:
- resume after interrupt
- human-in-the-loop
- conversation memory
- time-travel debugging
- fault recovery

### Trade-offs
Checkpoint storage adds persistence complexity and state size must be controlled.

### Alternatives
- No checkpoint
- Store only final result
- Store full event log
- External workflow engine

### When not to use
For stateless, short-lived requests that do not need recovery.

### Interview talking point
Checkpoint is what turns an Agent run from a single request into a recoverable workflow. It is essential for HITL and long-running agents.

---

## 11. Decision: Keep conversation memory for multi-turn chat

### Problem
Large language models are stateless. To support multi-turn conversation, they need access to previous turns. Storing everything in the prompt can exceed context limits and degrade prompt performance.

### Decision
Store recent chat history in the conversation state and checkpointers, and pass it dynamically to the LLM context.

### Why
It keeps conversation context across turns, allowing the agent to resolve references like "it" or "the previous document" and maintain thread continuity.

### Trade-offs
Increases input token count and cost; requires truncation or summarization strategies.

### Alternatives
- Window-based memory
- Summary memory
- No conversational history

### When not to use
Single-turn search tasks where context is not carried over.

### Interview talking point
Short-term conversational memory is managed via thread states and LangGraph checkpointers, making it persistent across sessions but bounded to control context usage.

---

## 12. Decision: Persist long-term memories in structured store

### Problem
Users have persistent preferences or facts that should span across different conversations (threads). Checking database tables every time is inefficient.

### Decision
Store long-term memories in a structured/semantic memory store (e.g. `memories` table) and retrieve relevant memories based on user ID and semantic similarity.

### Why
Allows personalization and persistence of facts (e.g. user preferences, corporate rules) across threads.

### Trade-offs
Adds DB lookups, vector matching, and background memory extraction/consolidation tasks.

### Alternatives
- Prompt-based defaults
- No personalization

### When not to use
Strictly stateless, anonymized tools where personalization is not needed.

### Interview talking point
Long-term memory is updated asynchronously or on-demand, storing key insights in a dedicated `memories` table to guide agent behavior globally.

---

## 13. Decision: Use human-in-the-loop for medium/high-risk actions

### Problem
Agents may propose actions with side effects, such as saving memory, sending email, modifying data, deleting files, or executing transactions. Letting the LLM execute these directly is unsafe.

### Decision
Use interrupt / resume for approval workflows.

### Why
The LLM can propose an action, but the system and human must authorize execution.

### Trade-offs
Human approval adds friction and slows execution.

### Alternatives
- Fully automatic execution
- Rule-based approval
- Role-based policy only
- Manual-only workflow

### When not to use
Low-risk read-only tools usually do not need approval.

### Interview talking point
For irreversible or state-changing actions, LLM should never have final authority. It can draft or propose, but execution requires policy checks and often human approval.
*区块链类比：HITL = 多签 / 签名确认。*

---

## 14. Decision: Stream workflow updates to the frontend

### Problem
Agent runs can take several seconds or longer. If the UI only shows a spinner, users do not know what the system is doing.

### Decision
Use SSE to stream workflow events from LangGraph to the frontend.

### Why
Streaming makes the Agent transparent:
- query rewrite
- retrieval progress
- evidence found
- rerank
- answer generation
- interrupt waiting

### Trade-offs
SSE adds connection management complexity.

### Alternatives
- Polling
- WebSocket
- Long polling
- Final-only response

### When not to use
Very short tasks may not need streaming.

### Interview talking point
Streaming is not just UX decoration. It exposes the Agent's execution trace to the user and builds trust.

---

## 15. Decision: Evaluate internal stages, not only final answer

### Problem
A wrong answer can come from many layers: query rewrite, retrieval, rerank, answerability, generation, citation, or tool selection. Final-answer-only eval cannot tell where the failure happened.

### Decision
Use layered evals. Evaluate:
- standalone question
- retrieval hit
- rerank top-k
- answerability
- citation validity
- structured output
- final answer
- tool trajectory

### Why
Layered eval makes failures actionable.

### Trade-offs
More eval cases and more implementation work.

### Alternatives
- Manual review
- LLM judge only
- Final answer string match
- User feedback only

### When not to use
For prototypes, final-answer eval may be enough temporarily.

### Interview talking point
I treat Agent eval like transaction trace debugging. I need to know which layer failed, not just whether the final output looked right.

---

## 16. Decision: Evaluate tool trajectory, not just response quality

### Problem
An Agent can produce a correct-looking answer while using the wrong tool path. For example, table aggregation should use a structured table query, not free-form RAG and LLM arithmetic.

### Decision
Record and evaluate tool calls, arguments, ordering, forbidden tool usage, and approval compliance.

### Why
Tool trajectory eval ensures the Agent uses the right process, not just the right words.

### Trade-offs
Tool eval requires structured tool traces and more detailed eval cases.

### Alternatives
- Final answer eval only
- Manual trace review
- LLM judge trajectory scoring

### When not to use
If the system has no tools or only one fixed tool.

### Interview talking point
For tool-using Agents, correctness includes tool selection and execution path. I evaluate expected_tools, forbidden_tools, required_order, and tool argument quality.

---

## 17. Decision: Separate run_id and thread_id

### Problem
A conversation or workflow thread can contain multiple user requests. Each request should be tracked independently.

### Decision
Use thread_id for conversation / workflow context and run_id for a single execution.

### Why
This enables:
- multi-turn memory
- run history
- event streaming per run
- replay and audit
- interrupt resume
- eval and debugging

### Trade-offs
More API and storage complexity.

### Alternatives
- Single chat_id
- Request-only stateless API
- Session-only tracking

### When not to use
For a stateless one-shot API.

### Interview talking point
thread_id is the stateful conversation boundary; run_id is the execution instance. This separation is important for streaming, checkpointing, resume, and observability.

---

## 18. Decision: Enforce permissions at system level, not prompt level

### Problem
LLMs are not security boundaries. Prompt instructions can fail, and RAG evidence may contain prompt injection.

### Decision
Apply authorization in retrieval, memory access, and tool execution layers.

### Why
The model should never see unauthorized content in the first place.

### Trade-offs
Requires ACL metadata, user context, permission filters, and tool policy.

### Alternatives
- Prompt-only safety
- Post-generation filtering
- Manual review

### When not to use
Never rely only on prompt for sensitive authorization.

### Interview talking point
Permissions must be enforced before data reaches the LLM. The model is not the access control layer.
*关键金句：The model is not the access control layer.*

---

## 19. Decision: Build the first version from first principles before adopting frameworks

### Problem
Frameworks hide complexity. This is good for speed, but bad for learning and debugging if the underlying workflow is not understood.

### Decision
Implement V1 from first principles, then use frameworks in V2.

### Why
This builds deep understanding of:
- chunking, metadata, embeddings, retrieval, rerank, answerability, structured output, state, memory, eval, and observability.

### Trade-offs
V1 takes more effort than using a framework directly.

### Alternatives
- Start with LangChain
- Start with LlamaIndex
- Start with Dify / Coze
- Start with CrewAI

### When not to use
If the goal is a quick customer PoC, using a framework or low-code platform first may be better.

### Interview talking point
I intentionally built V1 from the bottom up so I could understand and debug each layer. In V2, I can map each layer to LangChain, LlamaIndex, LangGraph, CrewAI, or Dify based on the use case.

---

## 20. V2 Plan: Framework Adapter Layer

V1 implements the system from first principles.

V2 will add framework adapters:

| Framework | Role |
|---|---|
| LangChain | Model, prompt, tool, retriever, output parser components |
| LangGraph | Stateful workflow runtime |
| LlamaIndex | RAG indexing, query engine, document workflows |
| CrewAI | Role-based multi-agent workflows |
| Microsoft Agent Framework | Enterprise multi-agent orchestration |
| Dify / Coze | Low-code customer-facing workflow PoC |
| LangSmith / RAGAS / DeepEval | Observability and eval |

*The goal of V2 is not to replace the architecture, but to compare how different frameworks express the same underlying Agent engineering concepts (用主流框架表达同一套底层 Agent 工程逻辑).*
