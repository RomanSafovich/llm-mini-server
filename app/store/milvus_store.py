from .base import VectorStore


class MilvusVectorStore(VectorStore):
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