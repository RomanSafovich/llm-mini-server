from .base import VectorStore
from pymilvus import MilvusClient, DataType
from collections import Counter


class MilvusVectorStore(VectorStore):
    def __init__(self, uri="http://milvus-standalone:19530", collection_name :str="document_chunks", embedding_dim=384):
        self.uri = uri
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.client = MilvusClient(uri=self.uri)
        index_params = self.client.prepare_index_params()
        index_params.add_index(
                field_name="embedding",
                metric_type="COSINE",
                index_type="IVF_FLAT",
                index_name="vector_index",
                params={ "nlist": 128 }
            )

        if not self.client.has_collection(self.collection_name):
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field("id", DataType.VARCHAR, max_length=256, is_primary=True)
            schema.add_field("chunk_text", DataType.VARCHAR, max_length=8192)
            schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.embedding_dim)
            schema.add_field("doc_id", DataType.VARCHAR, max_length=128)
            schema.add_field("chunk_index", DataType.INT64)
            
            self.client.create_collection(
                collection_name=self.collection_name,
                index_params=index_params,
                schema=schema,
            )

            
        self.client.load_collection(collection_name=self.collection_name)


    def _validate_item(self, item):
        if not isinstance(item, dict):
            raise TypeError(f"item must be dict, got {type(item).__name__}")

        required = {"id", "text", "embedding", "metadata"}
        missing = required - set(item.keys())
        if missing:
            raise ValueError(f"Missing required keys: {sorted(missing)}")


    def upsert(self, items: list[dict]):
        rows = []
        if not isinstance(items, list):
            raise TypeError(f"items must be list, got {type(items).__name__}")

        for item in items: 
            self._validate_item(item)
            row = {
                "id": item["id"],
                "chunk_text": item["text"],
                "embedding": item["embedding"],
                "doc_id": item["metadata"]["doc_id"],
                "chunk_index": item["metadata"]["chunk_index"]
            }
            rows.append(row)

        self.client.upsert(
            collection_name=self.collection_name,
            data=rows
        )

    def search(self, query_embedding: list[float], top_k: int, filters=None) -> list[dict]:
        scored_items = []
        
        if top_k <= 0:
            raise ValueError("top_k must be a positive number")


        res = self.client.search(
            collection_name=self.collection_name,
            anns_field="embedding",
            data=[query_embedding],
            limit=top_k,
            output_fields=["chunk_text", "doc_id", "chunk_index", "embedding"],
            search_params={"metric_type": "COSINE"}
        )

        for hits in res:
            for hit in hits:
                scored_items.append(
                    {
                        "id": hit["id"],
                        "text": hit["entity"]["chunk_text"],
                        "score": hit["distance"],
                        "embedding": hit["entity"]["embedding"],
                        "metadata": {
                            "doc_id": hit["entity"]["doc_id"],
                            "chunk_index": hit["entity"]["chunk_index"]
                        }
                    }
                )
            
        return scored_items

    def delete_doc(self, doc_id: str) -> bool:
        try:
            res = self.client.delete(
                collection_name=self.collection_name,
                filter=f'doc_id == "{doc_id}"'
            )

            deleted = res.get("delete_count", 0)
            return deleted > 0
                
        except Exception as e:
            print(f"Delete failed with error: {e}")
            return False

    def clear(self):
        self.client.truncate_collection(collection_name=self.collection_name)

    def list_docs(self) -> list[dict]:

        iterator = self.client.query_iterator(
            collection_name=self.collection_name,
            # filter="", # Get everything
            output_fields=["doc_id"],
            batch_size=100 
        )

        counts = Counter()
        try:
            while True:
                rows = iterator.next() 
                if not rows:
                    break
                
                counts.update([row["doc_id"] for row in rows])

        finally:
            iterator.close()

        return [{"doc_id": doc_id, "chunk_count": chunk_count} for doc_id, chunk_count in counts.items()]


    def count(self) -> int:
        count_res = self.client.query(
            collection_name=self.collection_name,
            output_fields=["count(*)"],
        )
        return count_res[0]["count(*)"]