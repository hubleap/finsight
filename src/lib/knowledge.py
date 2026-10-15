from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from chromadb.config import Settings
import chromadb
from langchain_core.tools import StructuredTool
import pandas as pd

from src.lib import splitter
import os

from src.lib.config import OLLAMA_API_KEY, OLLAMA_BASE_URL, OLLAMA_EMBEDDING_MODEL, VECTOR_DIR
from src.lib.logger import get_logger

logger = get_logger('Knowledge')


class Knowledge:
    """Manages vector storage for RAG-based knowledge retrieval.

    Handles ingestion, updating, and retrieval of documents from
    Chroma vector store. Supports syncing from Google Drive and
    ingesting from pandas DataFrames.
    """

    def __init__(self, collection_name: str = 'farm_docs', persistent_dir: str = VECTOR_DIR):
        """Initialize the knowledge manager with Chroma vector store.

        Args:
            drive: Instance of GoogleDrive for document management
            collection_name: Name of the Chroma collection
            persistent_dir: Directory path for persistent storage
        """
        self.collection_name = collection_name
        self.embedding = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        self.persistent_dir = persistent_dir

        self.client = chromadb.PersistentClient(
            path=self.persistent_dir,
            settings=Settings(
                allow_reset=True,
                anonymized_telemetry=False
            )
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        self.store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding,
            persist_directory=self.persistent_dir,
            client=self.client
        )
    
    async def ingest_from_text(self, text: str, metadata: dict, file_id: str):
        """Ingest text into the vector store.

        Args:
            text: Text to ingest
            metadata: Metadata for the text
            file_id: Identifier for the text source
        """
        chunks = splitter.split_text(text)
        await self.delete_document(file_id)

        self.store.add_texts(
            texts=chunks,
            metadatas=[{**metadata, "file_id": file_id}] * len(chunks),
            ids=[f"{file_id}_{i}" for i in range(len(chunks))]
        )
        logger.info(f"Ingested text with id {file_id} into vector store")


    async def ingest_from_dataframe(self, df: pd.DataFrame, file_id: str):
        """Ingest data from a pandas DataFrame into the vector store.

        Args:
            df: DataFrame containing data to ingest
            file_id: Identifier for the data source
        """
        documents = splitter.split_dataframe(df)
        chunks = splitter.split_text("\n".join([doc.page_content for doc in documents]))
        await self.delete_document(file_id)

        self.store.add_texts(
            texts=chunks,
            metadatas=[{"file_id": file_id}] * len(chunks),
            ids=[f"{file_id}_{i}" for i in range(len(chunks))]
        )
        logger.info(f"Ingested dataframe with id {file_id} into vector store")

    async def get_file(self, file_id: str):
        """Retrieve all chunks associated with a file ID.

        Args:
            file_id: File identifier to search for

        Returns:
            Dictionary with ids, metadatas, and documents
        """
        results = self.collection.get(
            where={"file_id": {"$eq": file_id}}
        )
        return results

    async def delete_document(self, file_id: str):
        """Delete all chunks associated with a file ID.

        Args:
            file_id: File identifier to delete
        """
        results = await self.get_file(file_id)
        if results['ids']:
            self.collection.delete(ids=results['ids'])

        logger.info(f"Deleted document with id {file_id} from vector store")

    def retrieve(self, query: str, search_count: int = 3) -> list:
        """Retrieve relevant documents from the vector store.

        Uses MMR (Maximum Marginal Relevance) search for diversity.

        Args:
            query: Search query string
            search_count: Number of results to return

        Returns:
            List of relevant Document objects
        """
        retriever = self.store.as_retriever(search_type='mmr', search_kwargs={'k': search_count})
        data = retriever.invoke(query)
        return data

    def retrieve_tool(self) -> StructuredTool:
        """Create a LangChain tool for knowledge retrieval.

        Returns:
            StructuredTool for querying the vector store
        """
        return StructuredTool.from_function(
            name='knowledge_retriever',
            description=(
                "Used to extract farm knowledge from the vector store. "
                "Can be used to retrieve farm records, operational documents, and historical data."
            ),
            func=lambda query, search_count=3: self.retrieve(query=query, search_count=search_count)
        )