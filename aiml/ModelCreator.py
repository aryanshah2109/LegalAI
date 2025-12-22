from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch

class HFModelCreator:
    def __init__(self):
        pass

    def hf_model_create(self, model_name: str):

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code = True
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name ,
            trust_remote_code = True,
            torch_dtype = torch.float16,
            device_map = "auto",
            low_cpu_mem_usage = True
        )

        return tokenizer, model
    
    
class EmbeddingModelCreator:
    def __init__(self):
        pass

    def embedding_model_create(self, model_name: str):

        embedding_model = HuggingFaceEmbeddings(
            model_name = model_name
        )

        return embedding_model