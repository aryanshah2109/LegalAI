import os

from langchain_community.document_loaders import DirectoryLoader, CSVLoader, TextLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentLoader:
    def __init__(self):
        pass

    def loader_function(self, file_path:str):
        
        _, ext = os.path.splitext(file_path)

        ext = ext.lower()

        match(ext):
            case ".txt":
                return TextLoader(file_path) 
               
            case ".pdf":
                return PyPDFLoader(file_path)
            
            case ".csv":
                return CSVLoader(file_path)
            
            case ".md":
                return UnstructuredMarkdownLoader(file_path)
            
            case _:
                return None

        


    def load_data(self, path:str):

        loader = DirectoryLoader(
            path = path,
            glob = "**/*",
            recursive=True,
            loader_cls = self.loader_function,
            use_multithreading=True,
            max_concurrency=4
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size = 8000,
            chunk_overlap = 500
        )

        return loader, splitter

