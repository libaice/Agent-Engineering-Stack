import faiss
import numpy as np
from VectorStore.common import embed, DOCS, QUERIES, print_results, timer



def main():
    print("=" * 60)
    print("FAISS 向量检索实验")
    print("=" * 60)

    texts = [d["text"] for d in DOCS]
    vectors = np.array(embed(texts), dtype=np.float32)
    dim = vectors.shape[1]
    print(f"\n📐 向量维度: {dim}, 文档数量: {len(vectors)}")

    indexes = {}

    # Flat：暴力精确搜索，小数据集的基准线
    indexes["IndexFlatL2"] = faiss.IndexFlatL2(dim)


    # IndexFlatIP：内积（余弦相似度，向量需先归一化）
    indexes["IndexFlatIP"] = faiss.IndexFlatIP(dim)

    # HNSW：生产推荐，近似最近邻，速度/精度可调
    hnsw = faiss.IndexHNSWFlat(dim, 32)  # M=32，每节点32条边
    hnsw.hnsw.efConstruction = 200        # 构建深度，越大越准但越慢
    hnsw.hnsw.efSearch = 64               # 查询深度，运行时可动态调整
    indexes["HNSW_M32_ef64"] = hnsw

    # IVF：大数据集，先聚类再搜索（需要训练）
    nlist = 4  # 聚类中心数，生产建议 sqrt(N) ~ 4*sqrt(N)
    quantizer = faiss.IndexFlatL2(dim)
    ivf = faiss.IndexIVFFlat(quantizer, dim, nlist)
    ivf.train(vectors)  # IVF 需要训练步骤！
    ivf.nprobe = 2  # 查询时检查的聚类数，越大越准但越慢
    indexes["IVF_nlist4_nprobe2"] = ivf


    for idx_name, index in indexes.items():
        print(f"\n🔷 索引类型: {idx_name}")
        index.add(vectors)
        print(f"  已写入 {index.ntotal} 条向量")

        query = QUERIES[0]
        q_vec = np.array(embed([query]), dtype=np.float32)
    
        distances, indices = index.search(q_vec, k=3)
    
        print(f"  查询: '{query}'")
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
            print(f"  [{rank}] dist={dist:.4f} | {DOCS[idx]['text'][:60]}...")

    print("\n" + "=" * 40)

    print("HNSW efSearch 参数 vs 速度实验")
    print("=" * 40)

    import time
    q_vec = np.array(embed([QUERIES[0]]), dtype=np.float32)
    hnsw_index = faiss.IndexHNSWFlat(dim, 32)
    hnsw_index.hnsw.efConstruction = 200
    hnsw_index.add(vectors)

    for ef in [16, 32, 64, 128, 256]:
        hnsw_index.hnsw.efSearch = ef
        start = time.perf_counter()
        for _ in range(100):  # 重复100次取平均
            D, I = hnsw_index.search(q_vec, k=3)
        elapsed = (time.perf_counter() - start) / 100 * 1000
        top1 = DOCS[I[0][0]]['text'][:40]
        print(f"  efSearch={ef:3d} | avg={elapsed:.3f}ms | top1: {top1}...")








if __name__ == "__main__":
    main()