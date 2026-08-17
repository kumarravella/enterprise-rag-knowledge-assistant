from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Enterprise RAG Knowledge Assistant",
    description="RAG-based question answering API using Python and LangChain",
    version="1.0.0"
)


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {
        "message": "Enterprise RAG Knowledge Assistant API",
        "status": "running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    """
    RAG question-answering endpoint.

    The retrieval and LLM components can be connected
    through the ingestion and retrieval modules.
    """
    return {
        "question": request.question,
        "answer": "RAG response will be generated from the indexed knowledge base."
    }
