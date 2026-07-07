

from app.rag import build_augmented_prompt, build_citation, build_context, build_sources_out, retrieve_unique_hits, run_chat_rag
from unittest.mock import Mock, patch

from app.config import settings
from app.schemas import ChatRagRequest


def test_no_hits():

    mock_store = Mock()
    mock_embedder = Mock()
    mock_model = Mock()
    mock_tokenizer = Mock()
    
    mock_store.search.return_value = []
    mock_embedder.encode_one.return_value = [0.1, 0.2]

    req = ChatRagRequest(question="hello world test")
    with patch("app.rag.generate_text") as mock_generate:
        response = run_chat_rag(
            req,
            store=mock_store,
            embedder=mock_embedder,
            model=mock_model,
            tokenizer=mock_tokenizer,
        )
    
    assert response.answer == "No relevant context found. Please ingest documents first."
    assert response.sources == []
    assert response.retrieved_count == 0
    mock_generate.assert_not_called()



def test_low_score():
    mock_store = Mock()
    mock_embedder = Mock()
    mock_model = Mock()
    mock_tokenizer = Mock()
    
    mock_store.search.return_value = [
        {
            "id": "doc_1_0",
            "text": "Some irrelevant document text test.",
            "score": settings.score_threshold - 0.01,
            "embedding": [0.1, 0.2],
            "metadata": {
                "doc_id": "doc_1",
                "chunk_index": 0,
            },
        }
    ]
    mock_embedder.encode_one.return_value = [0.1, 0.2]
    req = ChatRagRequest(question="hello world test")
    with patch("app.rag.generate_text") as mock_generate:
        response = run_chat_rag(
            req,
            store=mock_store,
            embedder=mock_embedder,
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

    assert response.answer == "No sufficiently relevant information was found in the indexed documents."
    assert response.sources == []
    assert response.retrieved_count == 0
    mock_generate.assert_not_called()


def test_confident_hits_generate_grounded_answer():
    mock_store = Mock()
    mock_embedder = Mock()
    mock_model = Mock()
    mock_tokenizer = Mock()
    
    mock_store.search.return_value = [
        {
            "id": "doc_1_0",
            "text": "Some irrelevant document text test1.",
            "score": settings.score_threshold + 0.15,
            "embedding": [0.1, 0.2],
            "metadata": {
                "doc_id": "doc_1",
                "chunk_index": 0,
            },
        },
        {
            "id": "doc_2_0",
            "text": "Some irrelevant document text test2.",
            "score": settings.score_threshold + 0.05,
            "embedding": [0.3, 0.4],
            "metadata": {
                "doc_id": "doc_2",
                "chunk_index": 0,
            },
        },
    ]
    mock_embedder.encode_one.return_value = [0.1, 0.2]
    req = ChatRagRequest(question="hello world test")
    with patch("app.rag.generate_text") as mock_generate:
        mock_generate.return_value = "Grounded answer [doc_1:chunk_0] [doc_2:chunk_0]"
        response = run_chat_rag(
            req,
            store=mock_store,
            embedder=mock_embedder,
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

    assert response.answer == (
        "Grounded answer [doc_1:chunk_0] [doc_2:chunk_0]"
    )
    assert response.sources[0].citation == "doc_1:chunk_0"
    assert response.sources[1].citation == "doc_2:chunk_0"
    assert len(response.sources) == 2
    assert response.retrieved_count == 2
    mock_generate.assert_called_once()


def test_build_citation():

    hit = {
        "metadata": {
            "doc_id": "doc_1",
            "chunk_index": 0,
        },
    }

    response = build_citation(hit)
    assert response == "doc_1:chunk_0"



def test_build_context():
    hits = [
        {
            "id": "doc_1_0",
            "text": "Some irrelevant document text test.",
            "score": settings.score_threshold - 0.01,
            "embedding": [0.1, 0.2],
            "metadata": {
                "doc_id": "doc_1",
                "chunk_index": 0,
            },
        }
    ]

    context, used_hits, used_chunks, used_chars = build_context(hits)
    assert "doc_1:chunk_0" in context
    assert used_hits == hits
    assert used_chunks == 1
    assert used_chars == len(context)


def test_build_sources_out():
    hits = [
        {
            "id": "doc_1_0",
            "text": "Some irrelevant document text test.",
            "score": settings.score_threshold - 0.01,
            "embedding": [0.1, 0.2],
            "metadata": {
                "doc_id": "doc_1",
                "chunk_index": 0,
            },
        }
    ]

    sources = build_sources_out(hits, False)
    assert len(sources) == 1
    assert sources[0].citation == "doc_1:chunk_0"
    assert sources[0].metadata["doc_id"] == "doc_1"
    assert sources[0].metadata["chunk_index"] == 0
    assert sources[0].text is None


def test_build_augmented_prompt():
    context = "test_doc:chunk_7\nRelevant evidence."
    question = "test2"
    prompt = build_augmented_prompt(concat_text=context, question=question)
    assert context in prompt
    assert "[doc_1:chunk_0]" in prompt
    assert question in prompt



def test_retrieve_unique_hits_without_doc_id():
    mock_store = Mock()
    mock_embedder = Mock()
    mock_embedder.encode_one.return_value = [0.1, 0.2]
    mock_store.search.return_value = []
    response = retrieve_unique_hits(
        question="test",
        effective_top_k=3,
        embedder=mock_embedder,
        store=mock_store,
        doc_id=None
    )

    assert response == []
    mock_embedder.encode_one.assert_called_once_with("test")
    mock_store.search.assert_called_once_with(
        [0.1, 0.2],
        3,
        filters=None,
    )



def test_retrieve_unique_hits_with_doc_id():
    hit = {
        "id": "doc_1_0",
        "text": "Test text",
        "score": 0.8,
        "embedding": [0.1, 0.2],
        "metadata": {
            "doc_id": "doc_1",
            "chunk_index": 0,
        },
    }
    mock_store = Mock()
    mock_embedder = Mock()
    mock_embedder.encode_one.return_value = [0.1, 0.2]
    mock_store.search.return_value = [hit]
    response = retrieve_unique_hits(
        question="test",
        effective_top_k=3,
        embedder=mock_embedder,
        store=mock_store,
        doc_id="doc_1"
    )

    assert response == [hit]
    mock_embedder.encode_one.assert_called_once_with("test")
    mock_store.search.assert_called_once_with(
        [0.1, 0.2],
        3,
        filters='doc_id == "doc_1"',
    )
