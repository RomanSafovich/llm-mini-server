from sentence_transformers import SentenceTransformer
import torch
import numpy as np

class Embedder:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        # optional to do embedding on cuda
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)


    def _validate_text(self, text):
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")

        if not text.strip():
            raise ValueError("text is an empty string")

    def encode_one(self, text: str) -> np.ndarray:
        self._validate_text(text)
        return self.model.encode(text, normalize_embeddings=True)


    def encode_many(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not isinstance(texts, list):
            raise TypeError("texts must be a list")

        if len(texts) == 0:
            raise ValueError("texts is an empty list")


        for text in texts: 
            self._validate_text(text)


        return self.model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=False, 
            normalize_embeddings=True
        )


