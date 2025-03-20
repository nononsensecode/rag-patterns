import os
from typing import List


from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_document(
    file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: List[str] = ["\n\n", "\n", " ", ""],
) -> List[str]:
    """
    Read a text document and split it into chunks using RecursiveCharacterTextSplitter.

    Args:
        file_path (str): Path to the text document
        chunk_size (int): Maximum size of each text chunk
        chunk_overlap (int): Number of characters to overlap between chunks
        separators (List[str]): List of separators to use for splitting, in order of priority

    Returns:
        List[str]: List of text chunks

    Raises:
        FileNotFoundError: If the specified file does not exist
        ValueError: If the file is empty or cannot be read
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Read the text file
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    if not text:
        raise ValueError(f"File is empty: {file_path}")

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators
    )

    # Split the text into chunks
    chunks = text_splitter.split_text(text)

    return chunks
