# src/VectorStore/VS02_Chroma.py
import chromadb
from chromadb.config import Settings
from VectorStore.common import embed, DOCS, QUERIES, print_results, timer


def main():
    print("=" * 60)
    print("Chroma 向量数据库实验")
    print("=" * 60)

    # ── 1. 两种模式对比 ───────────────────────────────────────────
    # 模式A：纯内存（测试用，重启消失）
    client_mem = chromadb.EphemeralClient()

    # 模式B：持久化到本地磁盘（开发推荐）
    client_disk = chromadb.PersistentClient(path="./chroma_data")

    client = client_disk

    # ── 2. 创建 Collection（相当于一张表）────────────────────────
    # 距离函数选择：cosine / l2 / ip
    collection = client.get_or_create_collection(
        name="knowledge_base",
        metadata={"hnsw:space": "cosine"},  # Chroma 内部用 HNSW
        # 关键参数（影响精度和内存）：
        # hnsw:M=16               # 默认16，越大越准但内存越多
        # hnsw:construction_ef=100 # 构建深度
        # hnsw:search_ef=10        # 查询深度，可以单独设
    )
    print(f"\n📦 Collection: {collection.name}, 当前文档数: {collection.count()}")

    # ── 3. 写入（带 metadata）────────────────────────────────────
    if collection.count() == 0:
        embeddings = embed([d["text"] for d in DOCS])
        collection.add(
            ids=[d["id"] for d in DOCS],
            embeddings=embeddings,
            documents=[d["text"] for d in DOCS],
            metadatas=[
                {"source": d["source"], "category": d["category"]} for d in DOCS
            ],
        )
        print(f"✅ 写入 {len(DOCS)} 条文档")

        # ── 4. 基础查询 ───────────────────────────────────────────────
        print("\n🔍 基础语义查询：")
        query = QUERIES[0]
        results = collection.query(
            query_embeddings=embed([query]),
            n_results=3,
            include=["documents", "distances", "metadatas"],
        )
        print(f"查询: '{query}'")
        for i, (doc, dist, meta) in enumerate(
            zip(
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0],
            ),
            1,
        ):
            print(f"  [{i}] dist={dist:.4f} | cat={meta['category']} | {doc[:55]}...")

        # ── 5. Metadata 过滤查询（Chroma 的核心优势之一）─────────────
        print("\n🔍 带 metadata 过滤查询（只搜 category=database）：")
        results_filtered = collection.query(
            query_embeddings=embed(["索引策略"]),
            n_results=3,
            where={"category": {"$eq": "database"}},  # 关键：先过滤再搜索
            # 支持：$eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $and, $or
            include=["documents", "distances", "metadatas"],
        )
        for i, (doc, dist, meta) in enumerate(
            zip(
                results_filtered["documents"][0],
                results_filtered["distances"][0],
                results_filtered["metadatas"][0],
            ),
            1,
        ):
            print(f"  [{i}] dist={dist:.4f} | source={meta['source']} | {doc[:55]}...")

        # ── 6. 增删改（Chroma 完整支持 CRUD）────────────────────────
        print("\n✏️  CRUD 操作演示：")

        # 增
        collection.add(
            ids=["doc_999"],
            embeddings=embed(["这是一条测试文档用于演示 CRUD"]),
            documents=["这是一条测试文档用于演示 CRUD"],
            metadatas=[{"source": "test", "category": "test"}],
        )
        print(f"  ADD → 当前文档数: {collection.count()}")

        # 改（update 会替换 embedding 和 metadata）
        collection.update(
            ids=["doc_999"],
            documents=["更新后的测试文档内容"],
            embeddings=embed(["更新后的测试文档内容"]),
            metadatas=[{"source": "test", "category": "updated"}],
        )
        print(f"  UPDATE → doc_999 已更新")

    # 删
    collection.delete(ids=["doc_999"])
    print(f"  DELETE → 当前文档数: {collection.count()}")

    # ── 7. 边界 & 调优总结 ────────────────────────────────────────
    print("""
    📌 Chroma 关键边界总结：
    - 舒适区：单 Collection < 100K 文档
    - 超过 500K 建议迁移 Milvus / Qdrant
    - metadata 过滤在内存里做，超大数据集会慢
    - 不支持 BM25 混合检索（需要外挂）
    - 持久化用 SQLite，并发写入有锁竞争问题
    - 生产用法：配合 FastAPI，单进程部署，适合中小型企业内部知识库
    """)


if __name__ == "__main__":
    main()
