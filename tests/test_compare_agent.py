
from unittest.mock import Mock, patch

from fastapi import HTTPException
import pytest

from app.compare_agent import DEFAULT_COMPARISON_CRITERIA, _extract_json_object, _parse_comparison_response, _prepare_criteria, _prepare_label, _validate_compare_request, run_compare_documents_agent
from app.schemas import CompareDocumentRef, CompareDocumentsAgentRequest, ComparisonEvidence


@pytest.mark.parametrize(
    "label",
    [
        None,
        "",
        "  ",
        "string",
        "String",
    ],
)
def test_prepare_label_uses_default_for_missing_or_placeholder(label: str | None):
    default_label = "default"
    test_label = _prepare_label(label, default_label)
    assert test_label == default_label


def test_prepare_label_keeps_custom_label():
    default_label = "default"
    label = " Old CV "
    test_label = _prepare_label(label, default_label)
    assert test_label == "Old CV"


@pytest.mark.parametrize(
    "criteria",
    [
        None,
        [],
        ["string"],
        ["   "],
    ],
)
def test_prepare_criteria_uses_defaults_for_missing_or_placeholder(criteria):
    req = CompareDocumentsAgentRequest(
        question="compare",
        documents=[
            CompareDocumentRef(doc_id="doc_a"),
            CompareDocumentRef(doc_id="doc_b"),
        ],
        criteria=criteria,
    )

    result = _prepare_criteria(req)

    assert result == DEFAULT_COMPARISON_CRITERIA


def test_prepare_criteria_cleans_custom_criteria():
    criteria = [" backend strength ", "", "string", " AI relevance "]
    req = CompareDocumentsAgentRequest(
        question="compare",
        documents=[
            CompareDocumentRef(doc_id="doc_a"),
            CompareDocumentRef(doc_id="doc_b"),
        ],
        criteria=criteria,
    )

    result = _prepare_criteria(req)

    assert result == ["backend strength", "AI relevance"]


def test_validate_compare_request_rejects_blank_question():

    req = CompareDocumentsAgentRequest(
        question="       ",
        documents=[
            CompareDocumentRef(doc_id="doc_a"),
            CompareDocumentRef(doc_id="doc_b"),
        ],
    )

    with pytest.raises(HTTPException) as exc:
        _validate_compare_request(req)

    assert exc.value.status_code == 400
    assert exc.value.detail == "question must not be blank"


@pytest.mark.parametrize(
    "documents",
    [
        [],
        [CompareDocumentRef(doc_id="doc_a")],
        [
            CompareDocumentRef(doc_id="doc_a"),
            CompareDocumentRef(doc_id="doc_b"),
            CompareDocumentRef(doc_id="doc_c"),
        ],
    ],
)
def test_validate_compare_request_rejects_wrong_document_count(documents):
    req = CompareDocumentsAgentRequest(
        question="question",
        documents=documents,
    )

    with pytest.raises(HTTPException) as exc:
        _validate_compare_request(req)

    assert exc.value.status_code == 400
    assert exc.value.detail == "exactly two documents are required"


@pytest.mark.parametrize(
    "documents",
    [
        [
            CompareDocumentRef(doc_id=""),
            CompareDocumentRef(doc_id="doc_b"),
        ],
        [
            CompareDocumentRef(doc_id="doc_a"),
            CompareDocumentRef(doc_id=""),
        ],
        [
            CompareDocumentRef(doc_id="   "),
            CompareDocumentRef(doc_id="doc_b"),
        ],
    ],
)
def test_validate_compare_request_rejects_blank_doc_id(documents):
    req = CompareDocumentsAgentRequest(
        question="question",
        documents=documents,
    )
    with pytest.raises(HTTPException) as exc:
        _validate_compare_request(req)

    assert exc.value.status_code == 400
    assert exc.value.detail == "doc_id must not be blank"


def test_validate_compare_request_rejects_duplicate_doc_ids():
    req = CompareDocumentsAgentRequest(
        question="question",
        documents=[
            CompareDocumentRef(doc_id="   doc_a "),
            CompareDocumentRef(doc_id="doc_a"),
        ],
    )
    with pytest.raises(HTTPException) as exc:
        _validate_compare_request(req)

    assert exc.value.status_code == 400
    assert exc.value.detail == "documents must have different doc_id values"


def test_extract_json_object_removes_prefix():
    raw_response = 'result: {"id": "test"}'
    result = _extract_json_object(raw_response)
    assert result == '{"id": "test"}'


@pytest.mark.parametrize(
    "raw_response",
    [
        "{",
        "}",
        "}{",
    ],
)
def test_extract_json_object_rejects_missing_json(raw_response):
    with pytest.raises(HTTPException) as exc:
        _extract_json_object(raw_response)
    assert exc.value.status_code == 502
    assert exc.value.detail == "model returned invalid JSON"


def test_parse_comparison_response_maps_evidence_citations():
    sources = [
        ComparisonEvidence(
            citation="doc_a:chunk_1",
            doc_id="doc_a",
            chunk_index=1,
            snippet="test",
        )
    ]

    raw_response = """Comparison:
    {
        "summary": "summary",
        "recommendation": "recommendation",
        "winner": "document_a",
        "comparison": [
            {
                "criterion": "specific evidence",
                "document_a": "A text",
                "document_b": "B text",
                "winner": "document_a",
                "reasoning": "reason",
                "evidence_citations": ["doc_a:chunk_1", "missing:chunk_9"]
            }
        ],
        "gaps": [],
        "risks": [],
        "next_actions": []
    }
    """
    result = _parse_comparison_response(raw_response=raw_response, sources=sources, debug=True)
    assert result.comparison[0].evidence == sources
    assert result.debug_info is not None
    assert result.debug_info["unknown_citations"] == ["missing:chunk_9"]
    assert result.sources == sources


def test_parse_comparison_response_rejects_invalid_json():
    with pytest.raises(HTTPException) as exc:
        _parse_comparison_response(
            raw_response="not json",
            sources=[],
            debug=False,
        )

    assert exc.value.status_code == 502
    assert exc.value.detail == "model returned invalid JSON"


def test_parse_comparison_response_rejects_invalid_structure():
    raw_response = '{"summary": "x"}'

    with pytest.raises(HTTPException) as exc:
        _parse_comparison_response(
            raw_response=raw_response,
            sources=[],
            debug=False,
        )

    assert exc.value.status_code == 502
    assert exc.value.detail == "model returned invalid comparison structure"


def test_run_compare_documents_agent_returns_structured_response():
    hit_a = {
        "id": "doc_a_0",
        "text": "Document A evidence about backend and RAG.",
        "score": 0.9,
        "embedding": [0.1, 0.2],
        "metadata": {
            "doc_id": "doc_a",
            "chunk_index": 0,
        },
    }

    hit_b = {
        "id": "doc_b_0",
        "text": "Document B evidence about backend and reliability.",
        "score": 0.8,
        "embedding": [0.3, 0.4],
        "metadata": {
            "doc_id": "doc_b",
            "chunk_index": 0,
        },
    }

    documents = [
        CompareDocumentRef(
            doc_id="doc_a",
        ),
        CompareDocumentRef(
            doc_id="doc_b",
        ),
    ]

    req = CompareDocumentsAgentRequest(
        question="question",
        documents=documents,
    )
    with (
        patch("app.compare_agent.retrieve_unique_hits") as mock_retrieve,
        patch("app.compare_agent.generate_text") as mock_generate,
    ):
        def fake_retrieve(_question, _effective_top_k, _store, _embedder, doc_id):
            if doc_id == "doc_a":
                return [hit_a]
            if doc_id == "doc_b":
                return [hit_b]
            return []

        mock_retrieve.side_effect = fake_retrieve
        mock_generate.return_value = """
        {
            "summary": "summary",
            "recommendation": "recommendation",
            "winner": "document_a",
            "comparison": [
            {
                "criterion": "specific evidence",
                "document_a": "A",
                "document_b": "B",
                "winner": "document_a",
                "reasoning": "A is stronger",
                "evidence_citations": ["doc_a:chunk_0", "doc_b:chunk_0"]
            }
            ],
            "gaps": [],
            "risks": [],
            "next_actions": []
        }
        """
        response = run_compare_documents_agent(
            req=req,
            store=Mock(),
            embedder=Mock(),
            model=Mock(),
            tokenizer=Mock(),
        )

        assert response.summary == "summary"
        assert response.winner == "document_a"
        assert len(response.comparison) == 1
        assert response.comparison[0].evidence[0].citation == "doc_a:chunk_0"
        assert response.comparison[0].evidence[1].citation == "doc_b:chunk_0"
        assert len(response.sources) == 2
        mock_generate.assert_called_once()
        assert mock_retrieve.call_count == 2
        called_doc_ids = [call.args[4] for call in mock_retrieve.call_args_list]
        assert set(called_doc_ids) == {"doc_a", "doc_b"}
