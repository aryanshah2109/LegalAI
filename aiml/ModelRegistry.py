from aiml.ModelCreator import EmbeddingModelCreator, HFModelCreator
from aiml.ModelWrapper import (
    RetrieverModelWrapper,
    RAGModelWrapper
)

from app.model_config import EMBEDDING_MODEL_NAME, HF_MODEL_NAME

class ModelRegistry:
    _instance = None
    
    def __new__(cls):           
        '''
        __new__ is a dunder method that is called after an object is created and before its __init__ is called
        
        This will help create only one instance of a model and then reuse it everywhere in the project
        
        :param cls: Description
        '''
        if cls._instance is None:
            cls._instance = super().__name__(cls)
            cls._instance._load_models()
        
        return cls._instance
    
    
    def _load_models(self):

        # Embedding Model
        self.embedding_model = EmbeddingModelCreator().embedding_model_create(EMBEDDING_MODEL_NAME)

        # LLM Model
        tokenizer, model = HFModelCreator().hf_model_create(HF_MODEL_NAME)

        # Retriever Model
        self.retriever_model = RetrieverModelWrapper(model, tokenizer)

        # RAG Model
        self.rag_model = RAGModelWrapper(model, tokenizer)