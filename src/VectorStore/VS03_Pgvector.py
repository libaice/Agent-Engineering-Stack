# src/VectorStore/VS03_Pgvector.py
# 依赖：需要运行 PostgreSQL + pgvector 扩展
# Docker 一行启动已完成：
# docker run -d --name pgvector-test -p 127.0.0.1:5433:5432 -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=pass -e POSTGRES_DB=postgres pgvector/pgvector:pg16

import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np
from VectorStore.common import embed, DOCS, QUERIES

# 连接到本地 Docker pgvector 容器端口 5433
CONN_STR = "postgresql://postgres:pass@127.0.0.1:5433/postgres"

print("=" * 60)
print("pgvector 实验（PostgreSQL + 向量扩展）")
print("=" * 60)

conn = psycopg2.connect(CONN_STR)
conn.autocommit = True
cur = conn.cursor()

# ── 1. 初始化 ─────────────────────────────────────────────────
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
cur.execute("DROP TABLE IF EXISTS documents CASCADE;")
cur.execute("""
    CREATE TABLE documents (
        id TEXT PRIMARY KEY,
        text TEXT NOT NULL,
        source TEXT,
        category TEXT,
        embedding vector(512)  -- BGE-small-zh-v1.5 模型的维度是 512
    );
""")
# 注册 vector 类型，这样 psycopg2 允许我们操作向量类型
register_vector(conn)
print("✅ 表结构创建完成")

# ── 2. 写入 ───────────────────────────────────────────────────
embeddings = embed([d["text"] for d in DOCS])
for doc, emb in zip(DOCS, embeddings):
    cur.execute(
        "INSERT INTO documents (id, text, source, category, embedding) VALUES (%s,%s,%s,%s,%s)",
        (doc["id"], doc["text"], doc["source"], doc["category"], emb),
    )
print(f"✅ 写入 {len(DOCS)} 条文档")

# ── 3. 三种距离函数对比 ───────────────────────────────────────
print("\n📐 三种距离度量对比（同一查询）：")
q_emb = embed([QUERIES[1]])[0]

# 我们使用参数绑定，同时支持不同算子的 SQL 拼装
distance_ops = {
    "L2（欧氏距离，<->）": "embedding <-> %s::vector",
    "余弦距离（<=>）": "embedding <=> %s::vector",
    "内积（负内积，<#>）": "embedding <#> %s::vector",
}

for name, op in distance_ops.items():
    cur.execute(
        f"""
        SELECT id, text, {op} AS score
        FROM documents
        ORDER BY score ASC
        LIMIT 3;
        """,
        (q_emb,),
    )
    rows = cur.fetchall()
    print(f"\n  [{name}] 查询: '{QUERIES[1]}'")
    for rank, (doc_id, text, score) in enumerate(rows, 1):
        print(f"    [{rank}] score={score:.4f} | {text[:55]}...")

# ── 4. 索引类型对比（核心调优知识）──────────────────────────
print("\n\n🔷 索引类型对比：")

# 无索引：精确搜索（基准）
cur.execute("DROP INDEX IF EXISTS idx_hnsw; DROP INDEX IF EXISTS idx_ivfflat;")
print("  [无索引] 精确搜索，小数据集推荐")

# HNSW 索引（pgvector 0.5+ 支持）
cur.execute("""
    CREATE INDEX idx_hnsw ON documents 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
    -- m: 每节点连接数 (推荐 16-64)
    -- ef_construction: 构建深度 (推荐 64-200)
""")
print("  [HNSW] m=16, ef_construction=64 索引已创建")

# 查询时控制 ef_search（搜索深度）
cur.execute("SET hnsw.ef_search = 40;")  # 默认40，越大越准越慢
cur.execute(
    """
    SELECT id, text, embedding <=> %s::vector AS score
    FROM documents
    ORDER BY score ASC
    LIMIT 3;
""",
    (q_emb,),
)
rows = cur.fetchall()
print(f"  HNSW查询结果（ef_search=40）:")
for rank, (doc_id, text, score) in enumerate(rows, 1):
    print(f"    [{rank}] score={score:.4f} | {text[:55]}...")

# ── 5. 混合查询：向量 + SQL 过滤（pgvector 独家优势）────────
print("\n🔍 混合查询：向量相似度 + SQL WHERE 过滤：")
cur.execute(
    """
    SELECT id, text, source, embedding <=> %s::vector AS score
    FROM documents
    WHERE category = 'database'          -- 先用 SQL 过滤
      AND source != 'milvus_docs'        -- 精确排除某来源
    ORDER BY score ASC
    LIMIT 3;
""",
    (q_emb,),
)
rows = cur.fetchall()
print(f"  只搜 category=database，排除 milvus_docs：")
for rank, (doc_id, text, source, score) in enumerate(rows, 1):
    print(f"    [{rank}] score={score:.4f} | source={source} | {text[:50]}...")

# ── 6. IVFFlat 索引（大数据集替代方案）─────────────────────
cur.execute("DROP INDEX IF EXISTS idx_hnsw;")
cur.execute("""
    CREATE INDEX idx_ivfflat ON documents
    USING ivfflat (embedding vector_l2_ops)
    WITH (lists = 4);
    -- lists: 聚类中心数，建议 sqrt(行数) 到 4*sqrt(行数)
    -- 需要在写入数据后再建索引（需要训练）
""")
print("\n  [IVFFlat] lists=4 索引已创建")

# 查询时控制 probes（检查的聚类数，越大越准）
cur.execute("SET ivfflat.probes = 2;")  # 默认1，建议 lists/10 ~ lists/4
cur.execute(
    """
    SELECT id, text, embedding <-> %s::vector AS score
    FROM documents ORDER BY score LIMIT 3;
""",
    (q_emb,),
)
rows = cur.fetchall()
print(f"  IVFFlat查询结果（probes=2）:")
for rank, (doc_id, text, score) in enumerate(rows, 1):
    print(f"    [{rank}] {text[:55]}...")

cur.close()
conn.close()

print("""
📌 pgvector 关键边界总结：
  - 数据量 < 1M：无索引或 HNSW，精度最高
  - 数据量 1M~10M：HNSW（m=16, ef_construction=128）
  - 数据量 > 10M：IVFFlat（lists=sqrt(N)，probes=10）
  - 核心优势：一条 SQL 同时做向量检索 + 业务逻辑过滤
  - 注意：向量列不能为 NULL；维度一旦定义不能修改
  - 并发写入性能好（继承 PG 的 MVCC）
""")