from pathlib import Path
from typing import List, Dict, Any

import pandas as pd


# row_to_text
def row_to_text(row: pd.Series) -> str:
    parts = []
    for col, value in row.items():
        if pd.isna(value):
            continue
        parts.append(f"{col}: {value}")
    return "；".join(parts)


def load_excel_as_row_chunks(file_path: str) -> List[Dict[str, Any]]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    excel = pd.ExcelFile(path)
    chunks = []

    for sheet_name in excel.sheet_names:
        df = pd.read_excel(excel, sheet_name=sheet_name)
        df.columns = [str(col).strip() for col in df.columns]

        for row_index, row in df.iterrows():
            text = row_to_text(row)

            if not text.strip():
                continue

            chunks.append(
                {
                    "document_id": None,
                    "source": path.name,
                    "file_path": str(path),
                    "file_type": "xlsx",
                    "sheet_name": sheet_name,
                    "row_index": int(row_index) + 2,  # +2 because Excel row 1 is header
                    "chunk_index": int(row_index),
                    "chunk_id": None,
                    "text": text,
                }
            )
    return chunks


def main():
    chunks = load_excel_as_row_chunks("data/excel/vendor_evaluation.xlsx")
    print(f"Total row chunks: {len(chunks)}")
    print("=" * 80)
    for chunk in chunks[:5]:
        print(chunk["chunk_id"])
        print(f'source={chunk["source"]} sheet={chunk["sheet_name"]} row={chunk["row_index"]}')
        print(chunk["text"])
        print("-" * 80)


if __name__ == "__main__":
    main()
