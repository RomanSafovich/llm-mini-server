from typing import Protocol
# from dataclasses import dataclass

# @dataclass
# class StoreItem:
#     id: str
#     text: str
#     embedding: list[float]
#     metadata: dict


# @dataclass
# class StoreHit:
#     id: str
#     text: str
#     score: float
#     embedding: list[float]
#     metadata: dict


class VectorStore(Protocol): 
    def upsert(self, items: list[dict]):
        ...

    def search(self, query_embedding: list[float], top_k: int, filters=None) -> list[dict]:
        ...

    def delete_doc(self, doc_id: str):
        ...

    def clear(self):
        ...

    def list_docs(self) -> list[dict]:
        ...

    def count(self) -> int:
        ...

