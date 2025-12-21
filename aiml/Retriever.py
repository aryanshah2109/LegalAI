from langchain.retrievers import MultiQueryRetriever
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from aiml.prompts import RETRIEVER_PROMPT
from aiml.ModelRegistry import ModelRegistry

class Retriever:
    def __init__(self, vector_store_path: str):
        registry = ModelRegistry()

        self.embedding_model = registry.embedding_model
        self.retriever_model = registry.retriever_model
        self.vector_store_path = vector_store_path
    
    def retrive_context(self):
        
        parser = StrOutputParser()

        prompt = PromptTemplate(
            template = RETRIEVER_PROMPT,
            input_variables = ["query"],
            partial_variables = {
                "format_instructions" : parser.get_format_instructions()
            } 
        )

        vector_store = FAISS.load_local(
            self.vector_store_path,
            embeddings = self.embedding_model,
            allow_dangerous_deserialization = True
        )

        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever = vector_store.as_retriever(
                search_type = "mmr",
                search_kwargs = {
                    "k" : 5,
                    "fetch_k" : 50
                }
            ),
            llm = self.retriever_model,
            prompt = prompt
        )

        return multi_query_retriever
