from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI


def build_vector_store(documents):
    """
    Split documents into chunks, create embeddings,
    and store them in a FAISS vector index.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_store


def create_llm():
    """Create the LLM used for response generation."""

    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )


def retrieve_context(vector_store, question, k=4):
    """Retrieve relevant document chunks for a question."""

    return vector_store.similarity_search(
        question,
        k=k
    )


def generate_response(llm, context, question):
    """Generate an answer using retrieved document context."""

    context_text = "\n\n".join(
        document.page_content
        for document in context
    )

    prompt = f"""
You are an enterprise knowledge assistant.

Answer the user's question using only the provided context.

Context:
{context_text}

Question:
{question}

If the answer is not available in the context, say that
the information was not found in the provided documents.
"""

    response = llm.invoke(prompt)

    return response.content
