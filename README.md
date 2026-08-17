# Enterprise RAG Knowledge Assistant

A Python-based Retrieval-Augmented Generation (RAG) application for asking questions against a collection of documents.

The project focuses on the core building blocks of a RAG application, including document ingestion, text chunking, embeddings, vector search, context retrieval, and LLM-based response generation.

## What it does

- Loads documents from a local data source
- Splits documents into smaller chunks
- Generates embeddings for document chunks
- Stores and searches embeddings using FAISS
- Retrieves relevant context for a user question
- Sends the retrieved context to an LLM
- Generates an answer based on the available documents
- Exposes the application through a simple API

## Architecture

```text
Documents
    ↓
Document Loading
    ↓
Text Chunking
    ↓
Embeddings
    ↓
FAISS Vector Store
    ↓
Similarity Search
    ↓
Retrieved Context
    ↓
LLM
    ↓
Generated Response
