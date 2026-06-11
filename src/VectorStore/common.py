import os
# Force Hugging Face Hub to run in offline mode using local cached models
os.environ["HF_HUB_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer
import time, json

MODEL = SentenceTransformer("BAAI/bge-small-zh-v1.5")  
# MODEL = SentenceTransformer("all-MiniLM-L6-v2")  # eng 

def embed(texts: list[str]) -> list[list[float]]:
    return MODEL.encode(texts, normalize_embeddings=True).tolist()

DOCS = [
    {"id": "doc_001", "text": "LangGraph 是 LangChain 的状态机扩展，支持循环、条件分支和持久化 checkpoint，适合构建复杂 Agent 工作流。", "source": "langchain_docs", "category": "framework"},
    {"id": "doc_002", "text": "RAG（检索增强生成）通过将外部知识库与 LLM 结合，有效减少幻觉，提升回答的准确性和可追溯性。", "source": "rag_guide", "category": "concept"},
    {"id": "doc_003", "text": "Milvus 支持 HNSW、IVF_FLAT、IVF_PQ 等多种索引类型，适合亿级向量的高性能检索场景。", "source": "milvus_docs", "category": "database"},
    {"id": "doc_004", "text": "pgvector 是 PostgreSQL 的向量扩展，支持 L2、余弦、内积三种距离度量，可与 SQL 联合查询做 metadata 过滤。", "source": "pgvector_docs", "category": "database"},
    {"id": "doc_005", "text": "Prompt Engineering 的核心技巧包括：角色设定、少样本示例、思维链（CoT）、输出格式约束等。", "source": "prompt_guide", "category": "concept"},
    {"id": "doc_006", "text": "向量检索的混合搜索（Hybrid Search）结合稠密向量和稀疏向量（BM25），在关键词精确匹配和语义理解之间取得平衡。", "source": "search_guide", "category": "search"},
    {"id": "doc_007", "text": "LlamaIndex 的 NodeParser 将文档切割为带 metadata 的节点，支持 source tracing 和层级检索。", "source": "llamaindex_docs", "category": "framework"},
    {"id": "doc_008", "text": "HNSW 索引的核心参数：M（每节点连接数，越大精度越高但内存越多）、ef_construction（构建时搜索深度）、ef_search（查询时搜索深度）。", "source": "hnsw_guide", "category": "database"},
    {"id": "doc_009", "text": "企业知识库建设的关键步骤：文档预处理、分块策略选择、embedding 模型选型、向量库部署、检索效果评估。", "source": "kb_guide", "category": "concept"},
    {"id": "doc_010", "text": "LangSmith 提供 LLM 应用的 trace、评估和回归测试能力，是生产环境可观测性的核心工具。", "source": "langsmith_docs", "category": "framework"},
]

QUERIES = [
    "如何构建 Agent 工作流？",
    "向量数据库的索引策略有哪些？",
    "什么是混合检索？",
    "如何减少 LLM 的幻觉问题？",
]

def timer(fn):
    """装饰器：测量函数执行时间"""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  ⏱ {fn.__name__} 耗时: {elapsed:.2f}ms")
        return result
    return wrapper

def print_results(results: list[dict], top_k: int = 3):
    for i, r in enumerate(results[:top_k], 1):
        score = r.get('score', r.get('distance', 'N/A'))
        print(f"  [{i}] score={score:.4f} | {r['text'][:60]}...")