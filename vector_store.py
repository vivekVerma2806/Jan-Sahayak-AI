import os
import shutil
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma


class ChromaONNXEmbeddings(Embeddings):
    """
    Lightweight wrapper around ChromaDB's built-in ONNX embedding function.
    Uses all-MiniLM-L6-v2 via onnxruntime — no sentence-transformers needed.
    Loads instantly, runs on CPU with no extra dependencies.
    """
    def __init__(self):
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        self._fn = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list) -> list:
        # Convert numpy.float32 → Python float to satisfy ChromaDB type checks
        return [[float(x) for x in v] for v in self._fn(texts)]

    def embed_query(self, text: str) -> list:
        return [float(x) for x in self._fn([text])[0]]


def get_embeddings_model(google_api_key: str = ""):
    """Returns the fast ONNX-based local embedding model."""
    return ChromaONNXEmbeddings()


def get_vector_store(persist_directory: str, google_api_key: str = ""):
    """Initializes or loads a persistent ChromaDB vector store."""
    embeddings = get_embeddings_model()
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vector_store

def add_documents_to_store(vector_store, documents) -> bool:
    """
    Embeds and adds a list of Document chunks to the vector database.
    """
    if not documents:
        return False
    
    vector_store.add_documents(documents)
    
    # In older LangChain versions, manual persistence is needed
    if hasattr(vector_store, 'persist') and callable(getattr(vector_store, 'persist')):
        try:
            vector_store.persist()
        except Exception as e:
            # Modern ChromaDB auto-persists and might raise deprecation or execution warnings
            print(f"Chroma persistence handled automatically: {e}")
            
    return True

def get_chunk_count(vector_store) -> int:
    """
    Returns the total number of chunks (embeddings) stored in the database.
    """
    try:
        # Chroma allows counting records via the collection interface
        if hasattr(vector_store, '_collection') and vector_store._collection is not None:
            return vector_store._collection.count()
    except Exception as e:
        print(f"Error counting vector store elements: {e}")
    return 0

def reset_vector_store(persist_directory: str) -> bool:
    """
    Wipes the ChromaDB directory entirely and rebuilds it as empty.
    """
    try:
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
        os.makedirs(persist_directory, exist_ok=True)
        print(f"Successfully cleared vector database at: {persist_directory}")
        return True
    except Exception as e:
        print(f"Error resetting vector database: {e}")
        return False
