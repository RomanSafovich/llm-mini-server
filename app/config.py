from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    llm_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    logger_name: str = "llm-mini-server"
    embedder_model_name: str = "BAAI/bge-small-en-v1.5"
    milvus_uri: str = "http://milvus-standalone:19530"
    near_duplicate_cosine_threshold: float = 0.97
    score_threshold: float = 0.65
    margin_threshold: float = 0.03
    max_top_k: int = 5
    max_context_chars: int = 6000
    max_chunk_snippet_chars: int = 800
    source_snippet_chars: int = 300
    collection_name: str = "document_chunks"
    embedding_dim: int = 384
    doc_id_max_length: int = 128


    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()