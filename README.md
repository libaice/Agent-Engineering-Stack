# Agent Engineering Stack

> 一个持续更新的 AI Agent 工程实践仓库：从 RAG 基础开始，逐步构建可检索、可推理、可记忆、可评测、可观测、可服务化的 Agent 系统。

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent%20Workflow-1C3C3C)](https://www.langchain.com/langgraph)
[![Blog](https://img.shields.io/badge/GitBook-中文实践笔记-3884FF?logo=gitbook&logoColor=white)](https://orderbook-trade.gitbook.io/agent)

这个项目采用 **Build in Public** 的方式记录我学习和实践 Agent Engineering 的完整过程。仓库中的代码与中文博客章节相互对应：代码展示可运行的实现，博客记录设计思路、踩坑过程和工程取舍。

📖 **完整中文博客：<https://orderbook-trade.gitbook.io/agent>**

> [!NOTE]
> 这是一个面向学习、实验和作品集展示的工程实验室，并非开箱即用的生产平台。部分模块仍在持续开发中。

## What You Will Learn

本仓库围绕一条渐进式工程路径展开：

```text
Document Parsing
    ↓
Embedding & Vector Search
    ↓
Hybrid Retrieval & Rerank
    ↓
Evidence Check & Safe Answer
    ↓
Query Rewrite & Decomposition
    ↓
Tools & Reasoning Loop
    ↓
LangGraph, Memory & Human-in-the-Loop
    ↓
Evaluation, Observability & Serving
```

你可以在这里看到：

- PDF、Excel 等文档的解析、切分与持久化入库
- FAISS、Chroma、pgvector、Milvus、Qdrant、Pinecone 和 OpenSearch
- BM25 + Vector Search 混合召回与 Cross-Encoder Rerank
- Evidence Sufficiency、Safe Refusal、Citation 和结构化输出
- Query Rewrite、Multi-Query Retrieval 和 Query Decomposition
- Tool Registry、Tool Router、ReAct 与 Plan-Execute
- LangGraph 条件路由、Checkpoint、Memory、HITL 和 Streaming
- LangChain、LlamaIndex、CrewAI、AutoGen 等框架实践
- Agent Eval、Tracing、成本控制、延迟优化和 FastAPI Serving

## Repository Map

```text
Agent-Engineering-Stack/
├── data/                 # 示例 PDF、Excel 与知识库文档
├── evals/                # RAG / Agent / CrewAI / AutoGen 评测集与结果
├── log/                  # 本地结构化 trace logger
├── sql/                  # runs、events、documents、chunks、memory 等表结构
├── storage/              # 本地 chunks、manifest、memory 与 FAISS 索引
└── src/
    ├── rag/              # Step 01–16：渐进式 RAG 主线
    ├── langgraph/        # Graph、Checkpoint、Memory、HITL、Streaming
    ├── tool/             # Tool schema、registry、router 与 agent loop
    ├── reasoning/        # ReAct、Plan-Execute、Structured CoT
    ├── harness/          # Sandbox、tool policy、planner 与 trajectory
    ├── VectorStore/      # 七种向量数据库的最小实践
    ├── langchain/        # LangChain RAG、Retriever 与 Tools
    ├── llamaIndex/       # LlamaIndex nodes、filters、retriever 与 router
    ├── CrewAI/           # Multi-agent、critic 与 eval 示例
    ├── autogen/          # AutoGen conversation 与 eval 示例
    ├── mcp/              # MCP server 与 client adapter
    ├── memory/           # JSON 长期记忆存储
    ├── eval/             # LangGraph Agent eval runner
    ├── prod/             # 可观测性、成本和延迟实验
    ├── api/              # FastAPI + SSE 服务示例
    ├── state/            # Agent state 类型定义
    ├── design/           # 架构方案与设计决策
    └── finetuning/       # Fine-tuning 决策矩阵
```

## Learning Path

### 1. Progressive RAG

RAG 主线按照 16 个步骤，从最小检索系统逐步演进到复杂问题分解与答案合成。

| Stage | Topics | Code |
| --- | --- | --- |
| Foundation | PDF parsing、chunking、embedding、vector search | `Step01`–`Step02` |
| Generation | LLM answer、citation、automated evaluation | `Step03`–`Step04` |
| Ingestion | Persistent index、multi-document、Excel parsing | `Step05`–`Step07` |
| Retrieval | Hybrid search、BM25、rerank | `Step08`–`Step09` |
| Guardrails | Evidence check、safe answer、structured output | `Step10`–`Step12` |
| Query Intelligence | Rewrite、multi-query、decomposition、synthesis | `Step13`–`Step16` |

- [查看 RAG 代码](src/rag)
- [阅读 RAG 中文教程](https://orderbook-trade.gitbook.io/agent/rag/1.1-wen-dang-chunk-he-pdf-parser)

### 2. Stateful Agents with LangGraph

LangGraph 模块把 RAG pipeline 演进成显式状态机：

1. 基础 RAG Graph
2. 条件分支与拒答路由
3. Checkpoint 与线程状态
4. 对话上下文与长期记忆
5. Human-in-the-Loop 记忆审批
6. Streaming UI events

- [查看 LangGraph 代码](src/langgraph)
- [阅读 LangGraph 中文教程](https://orderbook-trade.gitbook.io/agent/langgraph/2.1-langgraph-rag)

### 3. Agent Harness & Reasoning

`src/harness/` 展示一个 Agent 执行环境的渐进演化：

```text
V0  workspace + tools + tests + reward
V1  tool policy + command allowlist + diff + eval runner
V2  AgentAction / Observation + action loop
V3  JSON action planner
V4  trajectories + DPO pair builder
V5  LLM planner + prompt builder + parse/retry
```

Reasoning 模块补充 ReAct、Plan-Execute 和结构化推理等常见模式。

- [Harness 笔记](https://orderbook-trade.gitbook.io/agent/harness/4.1-harness-shi-sha)
- [Reasoning 笔记](https://orderbook-trade.gitbook.io/agent/reasoning/5.1-react)

### 4. Framework & Infrastructure Experiments

仓库还包含同一类 Agent 能力在不同框架和基础设施上的最小实现，便于理解抽象边界，而不是绑定某一个框架：

- Frameworks：LangChain、LlamaIndex、CrewAI、AutoGen
- Protocol：MCP server / client adapter
- Vector stores：FAISS、Chroma、pgvector、Milvus、Qdrant、Pinecone、OpenSearch
- Production concerns：Evaluation、Tracing、Cost、Latency、API Serving

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- DeepSeek API key（运行需要 LLM 的示例时）

### Installation

```bash
git clone https://github.com/libaice/Agent-Engineering-Stack.git
cd Agent-Engineering-Stack
uv sync
```

创建 `.env`：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
```

### Run the RAG Pipeline

先处理 `data/docs/` 下的文档：

```bash
uv run python src/rag/Step06_ingest_multi_docs.py
```

然后运行一个完整的分解式 RAG 示例：

```bash
uv run python src/rag/Step16_decomposed_rag_demo.py
```

你也可以从 `Step01` 开始逐个运行。每个文件尽量保持独立，以便观察每一层能力解决了什么问题、又引入了什么成本。

### Run the Streaming API

```bash
uv run uvicorn api.serving_api_demo:api \
  --app-dir src \
  --host 127.0.0.1 \
  --port 8000 \
  --reload
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

SSE streaming：

```bash
curl -N -X POST http://127.0.0.1:8000/runs/stream \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the project price?","thread_id":"demo-thread"}'
```

> [!WARNING]
> API 模块当前仍是 Demo：`/runs/stream` 和 `/health` 已提供实现，其余 run persistence / resume 端点仍在建设中。

## Evaluation

评测数据位于 `evals/`，覆盖：

- 检索命中与目标页检查
- 预期关键词与引用检查
- Answerability 与安全拒答
- 多轮上下文理解
- Agent / CrewAI / AutoGen pattern evaluation
- Latency 与错误记录

运行 LangGraph Agent evaluation：

```bash
uv run python src/eval/agent_eval_runner.py
```

评测需要访问配置的 LLM，并可能加载本地 embedding / reranker 模型。

## Observability

项目支持 LangSmith 与 Langfuse 实验。使用本地 Langfuse：

```bash
docker compose up -d
```

然后访问 <http://localhost:3000>。`docker-compose.yml` 中的账号、密码和 secret 仅用于本地开发，部署前请替换。

## Design Principles

- **Retriever 决定上限，Reranker 优化排序，LLM 决定生成质量。**
- **相关不等于可回答。** 检索返回 top-k，不代表证据足以支持答案。
- **Vector search 不是银弹。** 企业数据中的 ID、数字、名称和字段需要 lexical signal。
- **Agent engineering is state management.** 状态、分支、重试、记忆和人工介入应该显式建模。
- **Evaluation is part of the system.** 没有可复现评测，就无法判断一次改动是否真的提升了系统。

更多背景和架构取舍见：

- [Architecture Design Decisions](src/design/design_decisions.md)
- [Design Solution](src/design/design_solution.md)

## Project Status

当前仓库重点是持续扩展实验覆盖面。不同目录的成熟度并不完全一致：RAG 主线和 LangGraph 示例相对完整，部分 API、MCP、框架适配和 production 模块仍是最小实现或实验性代码。

如果你发现问题或希望讨论某个 Agent Engineering 主题，欢迎提交 Issue。

## Blog & Repository

- 中文博客：<https://orderbook-trade.gitbook.io/agent>
- GitHub：<https://github.com/libaice/Agent-Engineering-Stack>

如果这个项目对你有帮助，欢迎 Star。也欢迎沿着博客和 Git history，一起观察它如何从一个最小 RAG Demo 演进为完整的 Agent Engineering Stack。

## License

本项目尚未添加开源许可证。在许可证明确前，代码版权归项目作者所有。
