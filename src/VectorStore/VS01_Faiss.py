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



if __name__ == "__main__":
    main()