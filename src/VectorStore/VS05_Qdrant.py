# src/VectorStore/VS05_Qdrant.py
# 依赖：pip install qdrant-client sentence-transformers
# 本地免 Docker 启动使用内存模式：QdrantClient(":memory:")

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Range,
    HasIdCondition,
    SearchParams,
    HnswConfigDiff,
)
from VectorStore.common import embed, DOCS, QUERIES
import uuid
import time

print("=" * 60)
print("Qdrant 向量数据库实验")
print("=" * 60)

# ── 1. 连接（支持内存模式，方便测试）────────────────────────
client = QdrantClient(":memory:")  # 纯内存，测试用
# client = QdrantClient(host="localhost", port=6333)  # 生产用

DIM = 512  # BGE-small-zh-v1.5 模型的维度是 512
COLLECTION = "knowledge_base"

# ── 2. 创建 Collection（含 HNSW 参数）────────────────────────
client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(
        size=DIM,
        distance=Distance.COSINE,
        # HNSW 参数直接在建 Collection 时定义（Qdrant 特色）
        hnsw_config=HnswConfigDiff(
            m=16,  # 每节点连接数
            ef_construct=100,  # 构建深度
            full_scan_threshold=10,  # 数据量低于此值时走精确搜索（自动切换！）
        ),
    ),
)
print(f"✅ Collection '{COLLECTION}' 创建完成")

# ── 3. 写入（带丰富 payload）────────────────────────────────
embeddings = embed([d["text"] for d in DOCS])
points = [
    PointStruct(
        id=i,  # Qdrant 要求 int 或 UUID
        vector=emb,
        payload={  # payload = metadata，任意 JSON
            "doc_id": doc["id"],
            "text": doc["text"],
            "source": doc["source"],
            "category": doc["category"],
            "char_count": len(doc["text"]),  # 可以存任意计算字段
        },
    )
    for i, (doc, emb) in enumerate(zip(DOCS, embeddings))
]
client.upsert(collection_name=COLLECTION, points=points)
print(f"✅ 写入 {len(points)} 条向量")

# ── 4. 基础查询 ───────────────────────────────────────────────
q_emb = embed([QUERIES[0]])[0]
results = client.query_points(
    collection_name=COLLECTION,
    query=q_emb,
    limit=3,
    with_payload=True,
)
print(f"\n🔍 基础查询: '{QUERIES[0]}'")
for rank, r in enumerate(results.points, 1):
    print(f"  [{rank}] score={r.score:.4f} | {r.payload['text'][:55]}...")

# ── 5. Qdrant 最强特性：复杂过滤（比 Milvus/Chroma 都强）────
print("\n🔍 复杂 Filter 演示（Qdrant 核心优势）：")

# 单条件过滤
filter_single = Filter(
    must=[FieldCondition(key="category", match=MatchValue(value="database"))]
)

# 多条件 AND
filter_and = Filter(
    must=[
        FieldCondition(key="category", match=MatchValue(value="database")),
        FieldCondition(key="char_count", range=Range(gte=30, lte=200)),
    ]
)

# OR 条件
filter_or = Filter(
    should=[
        FieldCondition(key="category", match=MatchValue(value="database")),
        FieldCondition(key="category", match=MatchValue(value="framework")),
    ]
)

# NOT 条件
filter_not = Filter(
    must_not=[
        FieldCondition(key="source", match=MatchValue(value="milvus_docs"))
    ]
)

for fname, f in [
    ("category=database", filter_single),
    ("database AND char_count in [30,200]", filter_and),
    ("database OR framework", filter_or),
    ("排除 milvus_docs", filter_not),
]:
    r = client.query_points(
        collection_name=COLLECTION,
        query=q_emb,
        query_filter=f,
        limit=2,
        with_payload=["text", "category", "source"],
    )
    print(f"\n  过滤条件: [{fname}]")
    for rank, hit in enumerate(r.points, 1):
        print(
            f"    [{rank}] score={hit.score:.4f} | {hit.payload['text'][:50]}..."
        )

# ── 6. 查询时动态调整 HNSW 参数 ───────────────────────────────
print("\n🔷 查询时动态调整 ef（精度 vs 速度权衡）：")
for ef in [32, 64, 128]:
    start = time.perf_counter()
    for _ in range(50):
        client.query_points(
            collection_name=COLLECTION,
            query=q_emb,
            search_params=SearchParams(hnsw_ef=ef, exact=False),
            limit=3,
        )
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  ef={ef:3d} | 50 次查询总耗时: {elapsed:.2f}ms")

print("""
📌 Qdrant 关键边界总结：
  - 舒适区：中大规模向量检索，支持极其复杂的 JSON 条件筛选（Nested Fields, Geo, Match 等）且过滤性能出色。
  - 特色一：Qdrant 速度极快，过滤机制完备，支持 Payload 字段建立 Payload Indexes 加速搜索。
  - 特色二：支持完全在线的 HNSW 参数调优，可以在 `search` 时传递 `SearchParams(hnsw_ef=...)`。
  - 特色三：非常智能的 `full_scan_threshold` 设定，数据量小时自动切换为精确暴力搜索，数据量大时走 HNSW，免去手动调优。
""")
