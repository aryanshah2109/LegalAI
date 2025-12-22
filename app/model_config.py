EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

HF_MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"

RETRIEVER_CONFIG = {
    "max_new_tokens" : 128,
    "do_sample" : False,
    "temperature" : 0.0,
    "top_p" : 1.0
}

RAG_CONFIG = {
    "max_new_tokens": 512,
    "temperature": 0.2,
    "do_sample": False,
    "top_p": 1.0,
    "repetition_penalty": 1.15
}