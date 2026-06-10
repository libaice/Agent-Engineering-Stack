import os
import pickle
import streamlit as st
import pandas as pd
import faiss
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings

# Page Config
st.set_page_config(
    page_title="FAISS Vector Store Viewer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    h1 {
        color: #1e3a8a;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        border-left: 5px solid #3b82f6;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #2563eb;
    }
    .doc-chunk {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        margin-bottom: 10px;
        transition: all 0.3s ease;
    }
    .doc-chunk:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border-color: #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔍 FAISS Vector Store Viewer")
st.markdown("An interactive web interface to explore, search, and visualize local FAISS index files.")

# Sidebar - Path Configuration
st.sidebar.header("📁 Configuration")
target_dir = st.sidebar.text_input("FAISS Directory Path", "storage/langchain_faiss_bge")

# Load Data helper
@st.cache_resource
def load_faiss_data(dir_path):
    index_path = Path(dir_path) / "index.faiss"
    pkl_path = Path(dir_path) / "index.pkl"
    
    if not index_path.exists() or not pkl_path.exists():
        return None, None, "Files not found"
        
    try:
        # Load FAISS index
        index = faiss.read_index(str(index_path))
        # Load Docstore
        with open(pkl_path, "rb") as f:
            docstore, index_to_docstore_id = pickle.load(f)
        return index, (docstore, index_to_docstore_id), None
    except Exception as e:
        return None, None, str(e)

if not Path(target_dir).exists():
    st.error(f"Directory `{target_dir}` does not exist. Please specify a valid folder path.")
else:
    index, doc_data, error = load_faiss_data(target_dir)
    
    if error:
        st.error(f"Error loading index: {error}")
    else:
        docstore, index_to_docstore_id = doc_data
        
        # Stat Metrics
        st.markdown("### 📊 Index Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="card"><strong>Total Vectors</strong><br><span class="metric-value">{index.ntotal}</span></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="card"><strong>Vector Dimension (dim)</strong><br><span class="metric-value">{index.d}</span></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="card"><strong>Index Type</strong><br><span class="metric-value">{type(index).__name__}</span></div>', unsafe_allow_html=True)
            
        # Prepare Dataframe for table view
        all_docs = []
        for i, (idx, doc_id) in enumerate(index_to_docstore_id.items()):
            doc = docstore.search(doc_id)
            if doc:
                meta = doc.metadata
                all_docs.append({
                    "Index": i,
                    "Doc ID": doc_id,
                    "Source": meta.get("source", "N/A"),
                    "Page": meta.get("page", 0) + 1 if isinstance(meta.get("page"), int) else meta.get("page", 0),
                    "Chunk ID": meta.get("chunk_id", "N/A"),
                    "Snippet": doc.page_content[:150] + "...",
                    "Full Content": doc.page_content,
                    "Metadata": str(meta)
                })
        
        df = pd.DataFrame(all_docs)
        
        # Tabs for Search and Browse
        tab1, tab2 = st.tabs(["🔎 Semantic Search Test", "🗂️ Browse All Chunks"])
        
        with tab1:
            st.subheader("Interactive Retrieval Tester")
            query = st.text_input("Enter a test query to retrieve nearest neighbors:")
            k = st.slider("Top K results to fetch", min_value=1, max_value=10, value=4)
            
            if query:
                with st.spinner("Embedding query and searching FAISS..."):
                    # Use HuggingFaceEmbeddings BGE model matching the store
                    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
                    query_vector = embeddings.embed_query(query)
                    
                    import numpy as np
                    query_arr = np.array([query_vector]).astype('float32')
                    
                    # FAISS search
                    distances, indices = index.search(query_arr, k)
                    
                    st.success(f"Retrieved nearest {k} chunks:")
                    for rank, (dist, idx_pos) in enumerate(zip(distances[0], indices[0])):
                        if idx_pos in index_to_docstore_id:
                            doc_id = index_to_docstore_id[idx_pos]
                            doc = docstore.search(doc_id)
                            st.markdown(f"""
                            <div class="doc-chunk">
                                <strong>Rank {rank + 1}</strong> | <b>Distance/L2 Score:</b> {dist:.4f} | <b>Doc ID:</b> {doc_id}<br>
                                <small><b>Source:</b> {doc.metadata.get('source')} | <b>Page:</b> {doc.metadata.get('page')}</small>
                                <hr style="margin: 8px 0; border: 0; border-top: 1px solid #eee;">
                                <p style="font-size: 14px; white-space: pre-wrap;">{doc.page_content}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.warning(f"Index {idx_pos} returned by FAISS was not found in Docstore mappings.")
                            
        with tab2:
            st.subheader("Saved Document Chunks")
            search_filter = st.text_input("Filter table by text keyword:")
            
            display_df = df
            if search_filter:
                display_df = df[df["Full Content"].str.contains(search_filter, case=False) | df["Metadata"].str.contains(search_filter, case=False)]
                
            st.dataframe(
                display_df[["Index", "Source", "Page", "Snippet", "Chunk ID"]],
                use_container_width=True
            )
            
            # Detailed Inspector accordion
            st.subheader("Detailed Chunk Inspector")
            selected_idx = st.selectbox("Select Chunk Index to view full details:", display_df["Index"].tolist() if not display_df.empty else [])
            if selected_idx is not None:
                selected_row = df[df["Index"] == selected_idx].iloc[0]
                col_left, col_right = st.columns([2, 1])
                with col_left:
                    st.info("📄 **Chunk Text Content**")
                    st.code(selected_row["Full Content"], language="text")
                with col_right:
                    st.info("⚙️ **Metadata Dictionary**")
                    st.json(selected_row["Metadata"])
