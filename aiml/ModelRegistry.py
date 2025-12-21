from aiml.ModelCreator import EmbeddingModelCreator, HFModelCreator
from aiml.ModelWrapper import RetrieverModelWrapper

from app.model_config import EMBEDDING_MODEL_NAME, HF_MODEL_NAME

class ModelRegistry:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__name__(cls)
            cls._instance._load_models()
        
        return cls._instance
    
    
    def _load_models(self):

        # Embedding Model
        self.embedding_model = EmbeddingModelCreator().embedding_model_create(EMBEDDING_MODEL_NAME)

        # LLM Model
        tokenizer, model = HFModelCreator().hf_model_create(HF_MODEL_NAME)

        # Retriever model
        self.retriever_model = RetrieverModelWrapper(model, tokenizer)
