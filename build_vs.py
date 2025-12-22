# build_vector_store.py
from aiml.VectorStore import VectorStore

vs = VectorStore()
# vs.vector_store_creator(
#     input_path="./data",
#     output_path="./vector_store/legal_files"
# )

vs.vector_store_creator(
    input_path="/content/drive/MyDrive/legal_data",
    output_path="/content/drive/MyDrive/vector_store/legal_files"
)

print("Vector store built successfully.")
