from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def load_documents(directory="data"):
    """
    Load PDF documents from the data directory.
    """

    documents = []

    data_path = Path(directory)

    for pdf_file in data_path.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        documents.extend(loader.load())

    return documents
