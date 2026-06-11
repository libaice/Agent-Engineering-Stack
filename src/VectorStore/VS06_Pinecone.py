# src/VectorStore/VS06_Pinecone.py
# 依赖：pip install pinecone sentence-transformers python-dotenv
# 需要：https://app.pinecone.io 注册免费账号，并在项目根目录的 .env 中配置 PINECONE_API_KEY

import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from VectorStore.common import embed, DOCS, QUERIES

# 自动加载 .env 配置文件中的环境变量
load_dotenv()


def main():
    print("=" * 60)
    print("Pinecone 托管向量数据库实验")
    print("=" * 60)

    # ── 1. 初始化 & API Key 校验 ──────────────────────────────────
    API_KEY = os.environ.get("PINECONE_API_KEY")
    if not API_KEY or API_KEY.strip() == "" or API_KEY == "你的API_KEY":
        print(
            "⚠️  提示：未检测到有效环境变量 PINECONE_API_KEY！\n"
            "   请先在项目根目录的 .env 文件中添加如下配置：\n"
            "   PINECONE_API_KEY=你的_API_KEY\n"
            "   （注册地址：https://app.pinecone.io，获取 API Key 无需绑定信用卡）\n\n"
            "   本次实验将被安全跳过。"
        )
        return

    pc = Pinecone(api_key=API_KEY)
    INDEX_NAME = "knowledge-base"
    DIM = 512  # BGE-small-zh-v1.5 模型的维度是 512

    # ── 2. 创建 Index（Pinecone 的核心概念）─────────────────────
    # Pinecone 有两种部署模式：
    # Serverless：按用量计费，冷启动慢，适合低频/开发
    # Pod-based：固定资源，延迟稳定，适合生产高并发
    if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIM,
            metric="cosine",  # cosine / euclidean / dotproduct
            spec=ServerlessSpec(
                cloud="aws",  # aws / gcp / azure
                region="us-east-1",  # 选最近的区域降低延迟
            ),
        )
        print(f"✅ Index '{INDEX_NAME}' 创建中，等待就绪...")
        # Serverless 创建需要时间，轮询状态
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(2)
        print(f"✅ Index 就绪")
    else:
        print(f"✅ Index '{INDEX_NAME}' 已存在")

    index = pc.Index(INDEX_NAME)

    # 查看 index 信息
    stats = index.describe_index_stats()
    print(f"📊 当前向量数: {stats.total_vector_count}, 维度: {stats.dimension}")

    # ── 3. 写入（Pinecone 的格式：list of tuples）────────────────
    embeddings = embed([d["text"] for d in DOCS])

    # Pinecone 写入格式：(id, vector, metadata_dict)
    vectors_to_upsert = [
        (
            doc["id"],
            emb,
            {  # metadata：任意 key-value，支持过滤
                "text": doc["text"],
                "source": doc["source"],
                "category": doc["category"],
                "char_count": len(doc["text"]),
            },
        )
        for doc, emb in zip(DOCS, embeddings)
    ]

    # 批量写入（推荐批次大小 100，最大 1000）
    BATCH_SIZE = 100
    t0 = time.perf_counter()
    for i in range(0, len(vectors_to_upsert), BATCH_SIZE):
        batch = vectors_to_upsert[i : i + BATCH_SIZE]
        index.upsert(vectors=batch)
    print(
        f"✅ 写入 {len(vectors_to_upsert)} 条，耗时 {(time.perf_counter()-t0)*1000:.1f}ms"
    )

    # Pinecone 写入是异步的！需要等一下才能查到最新统计
    time.sleep(2)
    stats = index.describe_index_stats()
    print(f"📊 写入后向量数: {stats.total_vector_count}")

    # ── 4. 基础查询 ───────────────────────────────────────────────
    q_emb = embed([QUERIES[1]])[0]  # "向量数据库的索引策略有哪些？"

    print(f"\n🔍 基础查询: '{QUERIES[1]}'")
    results = index.query(
        vector=q_emb,
        top_k=3,
        include_metadata=True,  # 必须显式声明才返回 metadata
        include_values=False,  # 通常不需要返回向量本身
    )
    for rank, match in enumerate(results["matches"], 1):
        print(
            f"  [{rank}] score={match['score']:.4f} | id={match['id']} | {match['metadata']['text'][:55]}..."
        )

    # ── 5. Metadata 过滤（Pinecone 的重要限制！）────────────────
    print("\n🔍 Metadata 过滤查询：")

    # 简单等值过滤
    results_filtered = index.query(
        vector=q_emb,
        top_k=3,
        filter={"category": {"$eq": "database"}},
        # 支持：$eq $ne $gt $gte $lt $lte $in $nin $and $or
        include_metadata=True,
    )
    print(f"  category=database 过滤结果：")
    for rank, match in enumerate(results_filtered["matches"], 1):
        print(
            f"    [{rank}] score={match['score']:.4f} | {match['metadata']['text'][:50]}..."
        )

    # 范围过滤
    results_range = index.query(
        vector=q_emb,
        top_k=3,
        filter={
            "$and": [
                {"char_count": {"$gte": 30}},
                {"char_count": {"$lte": 150}},
                {"category": {"$in": ["database", "framework"]}},
            ]
        },
        include_metadata=True,
    )
    print(f"\n  char_count[30~150] AND category in [database,framework]：")
    for rank, match in enumerate(results_range["matches"], 1):
        print(
            f"    [{rank}] score={match['score']:.4f} | {match['metadata']['text'][:50]}..."
        )

    # ⚠️ 重要限制：Pinecone metadata 过滤是"后过滤"
    # 先向量检索 top_k * 扩展系数，再做 metadata 过滤
    # 如果过滤条件很严（命中率低），可能返回不足 top_k 条！
    # 解决方法：适当加大 top_k，或用 namespace 提前分区
    print("\n  ⚠️ Pinecone 过滤是后过滤：命中率低时需要加大 top_k")

    # ── 6. Namespace（Pinecone 独特特性）────────────────────────
    # Namespace = 同一个 Index 内的逻辑分区，完全隔离，零额外成本
    # 典型用法：多租户（每个客户一个 namespace）
    print("\n🗂️  Namespace 多租户演示：")

    # 写入到不同 namespace
    index.upsert(
        vectors=[
            (
                "extra_001",
                embed(["这是租户A的私有文档"])[0],
                {"text": "这是租户A的私有文档", "category": "private"},
            )
        ],
        namespace="tenant_a",
    )
    index.upsert(
        vectors=[
            (
                "extra_002",
                embed(["这是租户B的私有文档"])[0],
                {"text": "这是租户B的私有文档", "category": "private"},
            )
        ],
        namespace="tenant_b",
    )
    time.sleep(1)

    # 查询时指定 namespace，完全隔离
    r_a = index.query(
        vector=q_emb,
        top_k=2,
        namespace="tenant_a",
        include_metadata=True,
    )
    print(
        f"  namespace=tenant_a 查询，只能看到 A 的数据：{len(r_a['matches'])} 条"
    )

    # 查看各 namespace 统计
    stats = index.describe_index_stats()
    print(f"  Namespaces: {dict(stats.namespaces)}")

    # ── 7. 按 ID 精确查询（非向量检索）─────────────────────────
    print("\n🔍 按 ID 精确拉取（fetch）：")
    fetched = index.fetch(ids=["doc_003", "doc_008"])
    for doc_id, data in fetched["vectors"].items():
        print(f"  id={doc_id} | {data['metadata']['text'][:55]}...")

    # ── 8. 删除操作 ───────────────────────────────────────────────
    print("\n🗑️  删除操作演示：")

    # 按 ID 删除
    index.delete(ids=["extra_001"], namespace="tenant_a")
    print("  按 ID 删除：doc extra_001 已删除")

    # 按 metadata 过滤删除（仅 Pod-based 支持，Serverless 不支持！）
    # index.delete(filter={"category": {"$eq": "test"}})
    print("  ⚠️ 按 metadata 批量删除：仅 Pod-based 支持，Serverless 不支持")

    # ── 9. 延迟测试（感受云端网络开销）─────────────────────────
    print("\n⏱️  延迟测试（本地 vs 云端的核心差距）：")
    latencies = []
    for _ in range(20):
        t0 = time.perf_counter()
        index.query(vector=q_emb, top_k=3)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    print(f"  p50={latencies[10]:.1f}ms")
    print(f"  p90={latencies[18]:.1f}ms")
    print(f"  p99={latencies[19]:.1f}ms")
    print(f"  （本地库 Qdrant/Chroma 通常 p50 < 1ms，差距来自网络RTT）")

    # ── 10. 清理（避免免费额度浪费）────────────────────────────
    # pc.delete_index(INDEX_NAME)  # 取消注释可删除整个 index
    print("\n💡 注意：Pinecone 免费 tier 只有 1 个 index，测完建议 delete")

    print("""
📌 Pinecone 关键边界 & 与自托管库的核心差距：

  【优势】
  - 零运维：无需 Docker/K8s，API key 拿来就用
  - 自动扩容：写入量、查询量自动扩，不用担心容量规划
  - Namespace：多租户零成本，ToB SaaS 场景神器
  - 全托管高可用：SLA 99.9%，自己部署难以达到

  【限制 & 坑】
  - metadata 后过滤：过滤命中率低时 top_k 实际返回不足
    → 解决：top_k * 3~5 倍，或用 namespace 预分区
  - Serverless 不支持按 metadata 批量删除
  - 无法存完整文档（metadata 单条上限 40KB）
    → 需要外挂存储（PostgreSQL/MongoDB）存原文
  - 冷启动延迟：Serverless index 长时间不用后首次查询慢
  - 网络延迟：p50 通常 20~80ms（vs 本地 <1ms）
  - 成本：百万向量免费，千万级别月费约 $70~$200

  【国内使用注意】
  - 服务器在海外，国内访问延迟 50~150ms，生产慎用
  - 如需托管方案，国内替代：Zilliz Cloud（Milvus托管版）
  
  【选型结论】
  - 个人项目/海外SaaS/快速验证 → Pinecone 省事
  - 国内生产/数据合规/低延迟 → Milvus/Qdrant 自托管
  - 已有 PostgreSQL → pgvector 直接加，成本最低
""")


if __name__ == "__main__":
    main()
