from fastapi import FastAPI
from contextlib import asynccontextmanager

from aiml.ModelRegistry import ModelRegistry
from aiml.LegalAI import LegalAI

legal_ai : LegalAI | None = None

from app.model_config import *

from dotenv import load_dotenv

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global legal_ai

    # Force models once
    ModelRegistry()

    # Creating RAG pipeline
    legal_ai = LegalAI(
        vector_store_path = "./vector_store/legal_files"
    )

    yield


app = FastAPI(
    title = "LegalAI - A RAG System",
    lifespan = lifespan
)

@app.get("/health")
def health():
    return {"health":"OK"}

@app.post("/rag")
def query_legal_ai(query: str):
    return {
        "answer" : legal_ai.run(query)
    }