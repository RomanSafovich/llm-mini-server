from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import patch
from fastapi import HTTPException


client = TestClient(app)


def test_get_documents():
    with patch("app.store.milvus_store.store.list_docs") as moc_list:
        moc_list.return_value = [{"doc_id": "test_doc", "chunk_count": 5}]
        response = client.get("/documents")
        assert response.status_code == 200
        assert response.json() == [{"doc_id": "test_doc", "chunk_count": 5}]


def test_delete_doc():
    with patch("app.store.milvus_store.store.delete_doc") as moc_val:
        moc_val.return_value = True
        response = client.delete("/documents/test_doc")
        assert response.status_code == 200
        assert response.json() == {"message": 'document test_doc deleted successfully'}

def test_delete_doc_fail():
    with patch("app.store.milvus_store.store.delete_doc") as moc_val:
        moc_val.return_value = False
        response = client.delete("/documents/test_doc")
        assert response.status_code == 404
        assert response.json() == {"detail": 'Document test_doc not found'}


def test_clear_documents():
    with patch("app.store.milvus_store.store.clear"):
        response = client.post("/documents/clear")
        assert response.status_code == 200
        assert response.json() == {"message": "documents have been cleared successfully"}


def test_ingest_text():
    with patch("app.main.run_ingest") as moc_dict:
        moc_dict.return_value = { 
            "doc_id": "test_doc", 
            "chunks_added": 5, 
            "total_chunks": 5
        }
        response = client.post("/ingest_text", json={"doc_id": "test_doc", "text": "test"})
        assert response.status_code == 200
        assert response.json() == { 
            "doc_id": "test_doc", 
            "chunks_added": 5, 
            "total_chunks": 5
        }

def test_ingest_text_fail():
    with patch("app.main.run_ingest") as moc_dict:
        moc_dict.side_effect = HTTPException(status_code=400, detail="doc_id must not be blank")
        response = client.post("/ingest_text", json={"doc_id": "", "text": "test"})
        assert response.status_code == 400
        assert response.json() == { "detail" : "doc_id must not be blank"}



def test_chat_rag():
    with patch("app.main.run_chat_rag") as moc_dict:
        moc_dict.return_value = {
            "answer": "test_ans",
            "sources": [],
            "retrieved_count": 5
        }
        response = client.post("/chat_rag", json={"question": "test_question", "top_k": 3, "debug": False})
        assert response.status_code == 200
        assert response.json() == {
            "answer": "test_ans",
            "sources": [],
            "retrieved_count": 5
        }


def test_chat_llm():
    with patch("app.main.generate_text") as moc_gen:
        moc_gen.return_value = "hello world"
        response = client.post("/chat", json={"prompt": "hello"})
        assert response.status_code == 200
        assert response.json() == {"answer": "hello world"}