# src/VectorStore/VS04_Milvus.py
# 依赖：pip install pymilvus[milvus_lite] sentence-transformers
# 本地免 Docker 启动使用 Milvus Lite 模式：uri="./milvus_demo.db"

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
    MilvusException,
)
from VectorStore.common import embed, DOCS, QUERIES
import json

print("=" * 60)
print("Milvus 向量数据库实验")
print("=" * 60)

# ── 1. 连接 (使用 Milvus Lite 本地文件连接模式) ─────────────────
connections.connect("default", uri="./milvus_demo.db")
print("✅ 已通过 Milvus Lite 连接到本地数据库文件 milvus_demo.db")

COLLECTION_NAME = "knowledge_base"
DIM = 512  # BGE-small-zh-v1.5 模型的维度是 512

# ── 2. 定义 Schema ────────────────────────────────────────────
if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)

fields = [
    FieldSchema(
        name="id", dtype=DataType.VARCHAR, max_length=50, is_primary=True
    ),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIM),
]
schema = CollectionSchema(fields=fields, description="企业知识库")
collection = Collection(name=COLLECTION_NAME, schema=schema)
print(f"✅ Collection '{COLLECTION_NAME}' 创建完成")

# ── 3. 写入 ───────────────────────────────────────────────────
embeddings = embed([d["text"] for d in DOCS])
data = [
    [d["id"] for d in DOCS],
    [d["text"] for d in DOCS],
    [d["source"] for d in DOCS],
    [d["category"] for d in DOCS],
    embeddings,
]
collection.insert(data)
collection.flush()  # 重要：flush 才真正持久化
print(f"✅ 写入 {collection.num_entities} 条向量")

# ── 4. 三种索引类型 & 参数对比（Milvus 的核心）────────────────
INDEX_CONFIGS = {
    "HNSW": {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {
            "M": 16,  # 每节点连接数 [4, 64]，越大越准，内存+
            "efConstruction": 200,  # 构建深度 [8, 512]，越大越准，构建慢
        },
    },
    "IVF_FLAT": {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {
            "nlist": 4  # 聚类中心数，推荐 4*sqrt(N)~16*sqrt(N)
        },
    },
    "IVF_PQ": {
        "index_type": "IVF_PQ",
        "metric_type": "L2",
        "params": {
            "nlist": 4,
            "m": 8,  # 子向量数（压缩率），dim必须能被m整除 (512 可以被 8 整除)
            "nbits": 8,  # 每个子向量的位数，通常8
        },
    },
}

print("\n🔷 各索引类型查询对比：")
q_emb = embed([QUERIES[2]])[0]

for idx_name, idx_config in INDEX_CONFIGS.items():
    # IVF_FLAT 和 IVF_PQ 索引需要进行聚类中心训练，通常需要至少 156+ 条数据。
    # 在极小数据集（当前只有 10 条）下运行，底层 Faiss 引擎会因为训练样本不足而发生 Segfault 崩溃。
    # 因此，如果是小数据集，我们对此进行跳过，仅展示配置。
    if idx_name in ["IVF_FLAT", "IVF_PQ"] and collection.num_entities < 156:
        print(f"\n  ⚠️ [{idx_name}] 索引创建被安全跳过（当前数据量 {collection.num_entities} < 156 聚类训练阈值）")
        continue

    # 删除旧索引，创建新索引
    if collection.has_index():
        collection.drop_index()
    collection.create_index(field_name="embedding", index_params=idx_config)
    collection.load()  # 加载到内存

    # 查询参数（查询时可动态调整，不影响索引）
    if idx_name == "HNSW":
        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        # ef: 查询深度 [top_k, 4096]，越大越准越慢，运行时可调！
    elif idx_name == "IVF_FLAT":
        search_params = {"metric_type": "L2", "params": {"nprobe": 2}}
        # nprobe: 检查的聚类数 [1, nlist]，越大越准越慢
    else:  # IVF_PQ
        search_params = {"metric_type": "L2", "params": {"nprobe": 2}}

    results = collection.search(
        data=[q_emb],
        anns_field="embedding",
        param=search_params,
        limit=3,
        output_fields=["id", "text", "category"],
    )

    print(f"\n  [{idx_name}] 查询: '{QUERIES[2]}'")
    for rank, hit in enumerate(results[0], 1):
        print(
            f"    [{rank}] score={hit.score:.4f} | {hit.entity.get('text')[:55]}..."
        )

    collection.release()  # 释放内存，方便下次 load 不同索引

# ── 5. Metadata 过滤（expr 表达式）───────────────────────────
print("\n🔍 带 metadata 过滤查询：")
if collection.has_index():
    collection.drop_index()
collection.create_index(
    field_name="embedding", index_params=INDEX_CONFIGS["HNSW"]
)
collection.load()

results = collection.search(
    data=[q_emb],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"ef": 64}},
    limit=3,
    expr='category == "database"',  # 标量过滤表达式
    # 支持：==, !=, >, <, >=, <=, in, not in, like, and, or
    # 例如：'category in ["database", "framework"] and source != "test"'
    output_fields=["id", "text", "source", "category"],
)
print(f"  只搜 category=database：")
for rank, hit in enumerate(results[0], 1):
    print(
        f"    [{rank}] score={hit.score:.4f} | cat={hit.entity.get('category')} | {hit.entity.get('text')[:45]}..."
    )

collection.release()
connections.disconnect("default")

print("""
📌 Milvus 关键边界总结：
  - HNSW：延迟最低，内存占用大，适合 < 1亿，在线服务
      M: 小数据集用16，大数据集用32~64
      efConstruction: 索引质量，生产用200
      ef(查询): 精度/速度权衡，可运行时调，32~128
  - IVF_FLAT：内存适中，适合1亿~10亿
      nlist = 4*sqrt(N)，nprobe = nlist/10
  - IVF_PQ：内存极省（压缩10x），精度有损，超大规模
      m 越大精度越高，要求 dim % m == 0
  - 标量字段建索引：collection.create_index(field_name="category", index_params={"index_type": "Trie"})
  - 必须 flush() 后 load() 才能搜索！
""")