import numpy as np
from .base import VectorStore
from collections import defaultdict

class MemoryVectorStore(VectorStore):
    def __init__(self):
        self.items = []


    def _validate_item(self, item):
        if not isinstance(item, dict):
            raise TypeError(f"item must be dict, got {type(item).__name__}")

        required = {"id", "text", "embedding", "metadata"}
        missing = required - set(item.keys())
        if missing:
            raise ValueError(f"Missing required keys: {sorted(missing)}")

        
    # def add_one(self, item):
    #     self._validate_item(item)
    #     self.items.append(item)

            
    def upsert(self, items: list[dict]):
        if not isinstance(items, list):
            raise TypeError(f"items must be list, got {type(items).__name__}")

        for item in items:
            self._validate_item(item)

        self.items.extend(items)


    def search(self, query_embedding: list[float], top_k: int, filters=None) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be a positive number")

        if not self.items:
            return []

        scored_pairs = []

        for item in self.items:
            scored_pairs.append((float(np.dot(query_embedding, item["embedding"])), item)) 

        
        scored_pairs.sort(key=lambda x: x[0], reverse=True)

        scored_items = []

        for pair in scored_pairs[:top_k]:
            scored_items.append(
                {
                    "score": pair[0],
                    "id": pair[1]["id"],
                    "text": pair[1]["text"],
                    "metadata":  pair[1]["metadata"],
                    "embedding": pair[1]["embedding"]

                }
            )
        return scored_items

    
    def count(self):
        return len(self.items)

    def delete_doc(self, doc_id: str):
        self.items = [item for item in self.items if item["metadata"].get("doc_id") != doc_id]


    def list_docs(self) -> list[dict]:
        docs = []
        chunk_count = defaultdict(int)
        for item in self.items:
            doc_id = item["metadata"].get("doc_id")
            if doc_id is None:
                continue
            chunk_count[doc_id] += 1

        for doc_id, count in chunk_count.items():

            docs.append(
                {
                    "doc_id": doc_id,
                    "chunk_count": count

                }
            )

        return docs
    
    def clear(self):
        self.items.clear()
