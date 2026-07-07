from unittest.mock import Mock

from app.store.milvus_store import MilvusVectorStore


def test_search_omits_filter_when_none():
    store = MilvusVectorStore()
    store.client = Mock()
    store.collection_name = "test_collection"
    store.client.search.return_value = [[]]
    store.search([0.1, 0.2], top_k=3, filters=None)
    search_kwargs = store.client.search.call_args.kwargs
    assert "filter" not in search_kwargs


def test_search_includes_filter_when_provided():
    store = MilvusVectorStore()
    store.client = Mock()
    store.collection_name = "test_collection"
    store.client.search.return_value = [[]]

    filter_expr = 'doc_id == "doc_1"'

    store.search(
        [0.1, 0.2],
        top_k=3,
        filters=filter_expr,
    )

    search_kwargs = store.client.search.call_args.kwargs

    assert search_kwargs["filter"] == filter_expr