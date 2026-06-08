import fitz
from pathlib import Path
from typing import List, Dict, Any


# def chuck_pdf(file_path: str) -> List[Dict[str, Any]]:


def load_pdf_by_page(file_path: str) -> List[Dict[str, Any]]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File {file_path} not exists.")
    doc = fitz.open(path)
    pages = []

    for page_index, page in enumerate(doc):
        text = page.get_text("text")
        pages.append(
            {"source": path.name, "page": page_index + 1, "text": text.strip()}
        )
    doc.close()
    return pages


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    if chunk_size <= overlap:
        raise ValueError("chuck size is must lager than overlap ")

    chucks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chucks.append(chunk)
        start = end - overlap
    return chucks


def chunk_pdf(file_path: str) -> List[Dict[str, Any]]:
    pages = load_pdf_by_page(file_path)
    all_chunks = []

    for page in pages:
        chucks = chunk_text(page["text"])
        for chunk_index, chuck in enumerate(chucks):
            all_chunks.append(
                {
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_id": f"{page['source']}:p{page['page']}:c{chunk_index}",
                    "text": chuck,
                }
            )

    return all_chunks


if __name__ == "__main__":
    # chunks = chunk_pdf("data/startup.pdf")
    chunks = chunk_pdf("data/project.pdf")
    print(f"Total chunks: {len(chunks)}")
    print("=" * 80)
    for i, chunk in enumerate(chunks):
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Source: {chunk['source']}")
        print(f"Page: {chunk['page']}")
        print(f"Text: {chunk['text']}")
        print("-" * 80)
