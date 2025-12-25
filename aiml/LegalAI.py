from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import  RunnablePassthrough, RunnableLambda

from aiml.Retriever import Retriever
from aiml.ModelRegistry import ModelRegistry
from aiml.prompts import RAG_PROMPT

class LegalAI:

    def __init__(self, vector_store_path: str):

        registry = ModelRegistry()

        self.rag_model = registry.rag_model
        self.retriever = Retriever(vector_store_path)

        retrieved_docs = RunnableLambda(
            lambda input_data: self.retriever.retrieve_context(
                input_data["query"] if isinstance(input_data, dict) else input_data
            )
        )

        format_context = RunnableLambda(self.format_context)

        self.chain = (
            {
                "retrieved_context" : retrieved_docs | format_context,
                "query" : RunnablePassthrough()
            } 
            | RAG_PROMPT  
            | self.rag_model 
            | StrOutputParser()
        )

    def format_context(self, context: str):
        formatted_docs = "\n\n".join(doc.page_content for doc in context)

        return formatted_docs

    def run(self, input_data):
        if isinstance(input_data, dict):
            input_data = input_data.get("query") or input_data.get("question")

        if not isinstance(input_data, str):
            raise ValueError("Input must be a string query")

        return self.chain.invoke(input_data)