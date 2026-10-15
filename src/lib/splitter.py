from langchain_text_splitters import RecursiveCharacterTextSplitter
from pandas import DataFrame
from langchain_core.documents import Document


def split_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list:
        """Split text into overlapping chunks for vector embedding.

        Args:
            text: Text content to split
            chunk_size: Maximum characters per chunk
            overlap: Number of overlapping characters between chunks

        Returns:
            List of text chunks
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=['\n\n', '\n', ' ', '']
        )
        return splitter.split_text(text)


def split_dataframe(df: DataFrame) -> list:
    """Convert DataFrame rows into document strings.

    Args:
        df: Pandas DataFrame to convert

    Returns:
        List of Document objects with row data as text
    """
    documents = []
    for _, row in df.iterrows():
        text = " | ".join(f'{col}:{row[col]}' for col in df.columns)
        documents.append(Document(text))

    return documents
