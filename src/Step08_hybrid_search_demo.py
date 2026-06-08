import json
from pathlib import Path
from typing import List, Dict, Any

import faiss
import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer