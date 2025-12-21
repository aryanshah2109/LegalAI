from langchain_community.vectorstores import FAISS
import faiss

from aiml.ModelCreator import EmbeddingModelCreator
from aiml.DocumentLoader import DocumentLoader

class VectorStore:
    def __init__(self, embedding_model_name, embedding_global_model = None):
        self.embedding_model_name = embedding_model_name
        self.embedding_global_model = embedding_global_model

    def get_embedding_model(self):
        if self.embedding_global_model is None:
            embedding_model =  EmbeddingModelCreator().embedding_model_create(self.embedding_model_name)
        
        return embedding_model


    def vector_store_creator(self, input_path: str, batch_size: int, output_path: str, ):

        embedding_model = self.get_embedding_model()

        embedding_dim = len(embedding_model.embed_query("dimensions check"))

        index = faiss.IndexFlatL2(embedding_dim)

        vector_store = FAISS(
            embedding_function = embedding_model,
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


