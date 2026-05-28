import pytest
from app.embeddings.embedder import Embedder
from app.models import LLMManager
from app.store.milvus_store import MilvusVectorStore

def test_embedder_singleton():
    embedder1 = Embedder()
    embedder2 = Embedder()
    assert embedder1 is embedder2


def test_model_singleton():
    manager1 = LLMManager()
    manager2 = LLMManager()
    assert manager1 is manager2


def test_milvus_vectore_store_singleton():
    store1 = MilvusVectorStore()
    store2 = MilvusVectorStore()
    assert store1 is store2