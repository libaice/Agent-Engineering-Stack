# src/VectorStore/VS07_Opensearch.py
# 依赖：pip install opensearch-py sentence-transformers python-dotenv
# Docker 启动命令：
# docker run -d -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" -e "DISABLE_INSTALL_DEMO_CONFIG=true" -e "DISABLE_SECURITY_PLUGIN=true" opensearchproject/opensearch:latest

import time
from opensearchpy import OpenSearch
from VectorStore.common import embed, DOCS, QUERIES


def main():
    print("=" * 60)
    print("OpenSearch 混合检索与向量库实验")
    print("=" * 60)

    # ── 1. 连接 & 健康检查 ────────────────────────────────────────
    # 默认连接本地 9200 端口，且禁用安全插件模式（无用户名密码、无 HTTPS）
    client = OpenSearch(
        hosts=[{"host": "localhost", "port": 9200}],
        use_ssl=False,
        verify_certs=False,
        http_compress=True,
    )

    try:
        if not client.ping():
            raise ConnectionError("Ping returned False")
    except Exception:
        print(
            "⚠️  提示：未检测到运行中的本地 OpenSearch 服务！\n"
            "   请先在终端使用以下 Docker 命令启动它：\n"
            "   docker run -d -p 9200:9200 -p 9600:9600 -e \"discovery.type=single-node\" -e \"DISABLE_INSTALL_DEMO_CONFIG=true\" -e \"DISABLE_SECURITY_PLUGIN=true\" opensearchproject/opensearch:latest\n\n"
            "   本次实验将被安全跳过。"
        )
        return

    print("✅ 已成功连接到本地 OpenSearch 服务")

    INDEX_NAME = "knowledge-base"
    DIM = 512  # BGE-small-zh-v1.5 模型的维度是 512

    # ── 2. 创建 Index（开启 k-NN 向量扩展）───────────────────────
    if client.indices.exists(index=INDEX_NAME):
        client.indices.delete(index=INDEX_NAME)
        print(f"✅ 旧索引 '{INDEX_NAME}' 已删除")

    # 定义索引映射（Mapping）：包含文本属性、标量属性，以及 k-NN 向量属性
    index_body = {
        "settings": {
            "index.knn": True,  # 关键：开启 k-NN 向量功能
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "text": {"type": "text", "analyzer": "standard"},  # 使用默认分词器
                "source": {"type": "keyword"},
                "category": {"type": "keyword"},
                "char_count": {"type": "integer"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": DIM,
                    # 配置 HNSW 搜索算法参数
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",  # 余弦相似度距离
                        "engine": "faiss",
                        "parameters": {"ef_construction": 128, "m": 16},
                    },
                },
            }
        },
    }

    client.indices.create(index=INDEX_NAME, body=index_body)
    print(f"✅ 带有 k-NN 属性映射的索引 '{INDEX_NAME}' 创建完成")

    # ── 3. 写入数据 ───────────────────────────────────────────────
    embeddings = embed([d["text"] for d in DOCS])

    print("📝 正在写入数据...")
    for doc, emb in zip(DOCS, embeddings):
        body = {
            "id": doc["id"],
            "text": doc["text"],
            "source": doc["source"],
            "category": doc["category"],
            "char_count": len(doc["text"]),
            "embedding": emb,
        }
        client.index(
            index=INDEX_NAME, id=doc["id"], body=body, refresh=True
        )  # refresh=True 使数据写入后立即立即可搜索
    print(f"✅ 成功导入 {len(DOCS)} 条文档")

    # ── 4. 全文检索演示 (BM25 关键词匹配) ─────────────────────────
    print("\n🔍 1. 全文检索 (BM25) 查询：")
    query_text = "混合检索"
    body_bm25 = {"query": {"match": {"text": query_text}}}
    res_bm25 = client.search(index=INDEX_NAME, body=body_bm25)

    print(f"  查询词: '{query_text}'")
    for hit in res_bm25["hits"]["hits"]:
        print(
            f"    [Score: {hit['_score']:.4f}] | id: {hit['_id']} | {hit['_source']['text'][:55]}..."
        )

    # ── 5. 向量检索演示 (k-NN 近邻匹配) ───────────────────────────
    print("\n🔍 2. 向量检索 (k-NN HNSW) 查询：")
    q_emb = embed([QUERIES[2]])[0]  # "什么是混合检索？"
    body_knn = {
        "size": 3,
        "query": {"knn": {"embedding": {"vector": q_emb, "k": 3}}},
    }
    res_knn = client.search(index=INDEX_NAME, body=body_knn)

    print(f"  查询向量: '{QUERIES[2]}'")
    for hit in res_knn["hits"]["hits"]:
        print(
            f"    [Score: {hit['_score']:.4f}] | id: {hit['_id']} | {hit['_source']['text'][:55]}..."
        )

    # ── 6. 混合检索演示 (Hybrid Search: BM25 + k-NN 融合) ─────────
    # OpenSearch 可以用 bool 联合查询来实现简单的混合搜索
    print("\n🔍 3. 混合搜索 (BM25 关键词 + k-NN 向量) 组合查询：")
    body_hybrid = {
        "size": 3,
        "query": {
            "bool": {
                "should": [
                    {
                        "match": {
                            "text": {"query": "混合检索", "boost": 1.0}
                        }  # 文本打分，权重 1.0
                    },
                    {
                        "knn": {
                            "embedding": {
                                "vector": q_emb,
                                "k": 3,
                                "boost": 2.0,  # 向量相似度打分，权重 2.0
                            }
                        }
                    },
                ]
            }
        },
    }
    res_hybrid = client.search(index=INDEX_NAME, body=body_hybrid)

    for rank, hit in enumerate(res_hybrid["hits"]["hits"], 1):
        print(
            f"    [{rank}] [Score: {hit['_score']:.4f}] | id: {hit['_id']} | {hit['_source']['text'][:55]}..."
        )

    print("""
📌 OpenSearch 关键边界总结：
  - 舒适区：已有 Elasticsearch/OpenSearch 搜索生态，对“传统关键词匹配”和“语义检索”有强烈的混合检索需求。
  - 核心优势：拥有最成熟的分词和文本索引机制，支持复杂的全文 DSL 组合查询（不仅仅是向量）。
  - 局限性：由于是基于 JVM 架构演进，高并发向量计算对内存要求较高；在高维向量 HNSW 图构建速度上略逊于专用向量库。
""")


if __name__ == "__main__":
    main()
