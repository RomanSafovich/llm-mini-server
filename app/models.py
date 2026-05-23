from typing_extensions import Self
import torch
from app.config import settings
from app.logger import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)


class LLMManager:
    _instance = None

    def __new__(cls) -> Self:
        if cls._instance == None:
            cls._instance = super().__new__(cls) 
            cls._instance.model = None
            cls._instance.tokenizer = None
            cls._instance.device = None
            
        return cls._instance

    def load_model(self):
        logger.info(f"Loading model {settings.llm_model_name}... this may take a minute ⏳")
        self.tokenizer = AutoTokenizer.from_pretrained(settings.llm_model_name)
        # GPU if available; otherwise CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                settings.llm_model_name,
                quantization_config=bnb_config,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                settings.llm_model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )

        logger.info(f"Model loaded successfully (CUDA: {torch.cuda.is_available()}) ✅")
        self.model.eval()

llm_manager = LLMManager()