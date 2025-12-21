from langchain_community.vectorstores import FAISS
import faiss
import os

from aiml.DocumentLoader import DocumentLoader
from aiml.ModelRegistry import ModelRegistry

class VectorStore:
    def __init__(self):
        registry = ModelRegistry()

        self.embedding_model = registry.embedding_model


    def vector_store_creator(self, input_path: str = "/data/", batch_size: int = 500, output_path: str = "/vector_store/legal_files"):

        
        embedding_dim = len(self.embedding_model.embed_query("dimensions check"))

        index = faiss.IndexFlatL2(embedding_dim)

        vector_store = FAISS(
            embedding_function = self.embedding_model,
            index = index,
            docstore = {},
            index_to_docstore_id = {}
        )

        loader, splitter = DocumentLoader().load_data(input_path)

        batches = []

        for doc in loader.lazy_load():

            chunk = splitter.split_documents([doc])

            batches.extend(chunk)

            if len(batches) >= batch_size:
                
                vector_store.add_documents(chunk)

                batches = []

        if batches:
            vector_store.add_documents(batches)

        vector_store.save_local(output_path)

        return


