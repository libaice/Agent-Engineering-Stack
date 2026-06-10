import os
import pickle
import faiss
from pathlib import Path

def inspect_faiss_index(index_dir: str):
    index_path = Path(index_dir) / "index.faiss"
    pkl_path = Path(index_dir) / "index.pkl"

    if not index_path.exists() or not pkl_path.exists():
        print(f"Error: Could not find index files in {index_dir}")
        return

    print("=" * 60)
    print(f"Inspecting FAISS Index Directory: {index_dir}")
    print("=" * 60)

    # 1. Inspect the binary FAISS index (index.faiss)
    try:
        index = faiss.read_index(str(index_path))
        print(f"[FAISS Index Info]")
        print(f"  - Total Vectors stored: {index.ntotal}")
        print(f"  - Vector Dimension (dim): {index.d}")
        print(f"  - Index Type: {type(index).__name__}")
        print(f"  - Is Trained: {index.is_trained}")
    except Exception as e:
        print(f"Error reading index.faiss: {e}")

    print("\n" + "-" * 60 + "\n")

    # 2. Inspect the documents mapping (index.pkl)
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        
        # In LangChain FAISS integration, pickle contains:
        # (docstore, index_to_docstore_id)
        if isinstance(data, tuple) and len(data) == 2:
            docstore, index_to_docstore_id = data
            print(f"[Docstore Info]")
            print(f"  - Total Document Chunks: {len(index_to_docstore_id)}")
            
            # Print a few samples
            print(f"\n[Showing up to 3 sample chunks]:")
            sample_ids = list(index_to_docstore_id.values())[:3]
            for idx, doc_id in enumerate(sample_ids):
                doc = docstore.search(doc_id)
                if doc:
                    print(f"\n--- Chunk {idx + 1} (ID: {doc_id}) ---")
                    print(f"Metadata: {doc.metadata}")
                    print(f"Content (snippet):\n{doc.page_content[:200]}...")
                else:
                    print(f"Document ID {doc_id} not found in docstore.")
        else:
            print("Unknown index.pkl structure. Raw data type:", type(data))
    except Exception as e:
        print(f"Error reading index.pkl: {e}")
    print("=" * 60)

if __name__ == "__main__":
    import sys
    # Default path used in Chain01
    default_dir = "storage/langchain_faiss_bge"
    target_dir = sys.argv[1] if len(sys.argv) > 1 else default_dir
    inspect_faiss_index(target_dir)
