from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
import os

from aiml.DocumentLoader import DocumentLoader
from aiml.ModelRegistry import ModelRegistry

class VectorStore:
    def __init__(self):
        registry = ModelRegistry()

        self.embedding_model = registry.embedding_model


    def vector_store_creator(self, input_path: str = "./data", batch_size: int = 500, output_path: str = "./vector_store/legal_files"):

        if os.path.exists(os.path.join(output_path, "index.faiss")):
            print(f"[VectorStore] Existing vector store found at {output_path}")
            return

        os.makedirs(output_path, exist_ok=True)
        
        embedding_dim = len(self.embedding_model.embed_query("dimensions check"))

        index = faiss.IndexFlatL2(embedding_dim)

        vector_store = FAISS(
            embedding_function = self.embedding_model,
            index = index,
            docstore = InMemoryDocstore({}),
            index_to_docstore_id = {}
        )

        loader, splitter = DocumentLoader().load_data(input_path)

        batches = []

        for doc in loader.lazy_load():

            chunks = splitter.split_documents([doc])

            for chunk in chunks:
                chunk.metadata.update({
                    "source_file" : doc.metadata.get("source"),
                    "page" : doc.metadata.get("page"),
                    "doc_type" : "legal"
                })

            batches.extend(chunks)

            if len(batches) >= batch_size:
                
                vector_store.add_documents(batches)

                batches = []

        if batches:
            vector_store.add_documents(batches)

        
        vector_store.save_local(output_path)

        return


