import os

from langchain.retrievers import MultiQueryRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

from typing import List

from aiml.prompts import RETRIEVER_PROMPT_TEMPLATE
from aiml.ModelRegistry import ModelRegistry

class Retriever:
    def __init__(self, vector_store_path: str):
        registry = ModelRegistry()

        self.embedding_model = registry.embedding_model
        self.retriever_model = registry.retriever_model
        self.vector_store_path = vector_store_path

        if not os.path.exists(vector_store_path):
            raise RuntimeError("Vector store not found. Startup build failed.")

        self.vector_store = FAISS.load_local(
            vector_store_path,
            embeddings = self.embedding_model,
            allow_dangerous_deserialization = True
        )

        self.build_retriever()

    def build_retriever(self):
        

        prompt = PromptTemplate(
            template = RETRIEVER_PROMPT_TEMPLATE,
            input_variables = ["question"]
        )

        
        self.multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever = self.vector_store.as_retriever(
                search_type = "mmr",
                search_kwargs = {
                    "k" : 5,
                    "fetch_k" : 50
                }
            ),
            llm = self.retriever_model,
            prompt = prompt
        )

    def retrieve_context(self, query: str) -> List[Document]:

        context = self.multi_query_retriever.get_relevant_documents(query)

        return context   
    
