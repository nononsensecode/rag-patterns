from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from typing import List
import argparse
from dotenv import load_dotenv

load_dotenv()

sys_prompt = """You are an expert in summarizing text. Write a concisesummary of the text provided. 
The summary should be a list of bullet points. The text to summarize is {text}"""


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


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", sys_prompt),
        ("user", "{text}"),
    ]
)

# Initialize the Google Generative AI model
llm = ChatGoogleGenerativeAI(
    temperature=0,
    model="gemini-1.5-flash-8b",
)


def summarize_text(splits: list[str]) -> str:
    """
    Summarize the text using Google Generative AI.

    Args:
        splits (list[str]): List of text chunks to summarize.

    Returns:
        str: The summary of the text.
    """
    # Create a list to store the summaries
    summaries = []

    chain = prompt | llm | StrOutputParser()

    # Iterate over each text chunk and summarize it
    for text in splits:
        # Generate the summary using the LLM
        summary = chain.invoke({"text": text})
        summaries.append(summary)

    # Join the summaries into a single string
    return "\n".join(summaries)


def main():
    """
    Main function to run the text splitter from command line.
    Usage: python text_splitter.py --file path/to/file.txt [--chunk-size 1000] [--chunk-overlap 200]
    """
    parser = argparse.ArgumentParser(
        description="Split a text document into chunks using RecursiveCharacterTextSplitter"
    )
    parser.add_argument(
        "--file", "-f", required=True, help="Path to the text file to process"
    )
    parser.add_argument(
        "--chunk-size",
        "-s",
        type=int,
        default=1000,
        help="Maximum size of each text chunk (default: 1000)",
    )
    parser.add_argument(
        "--chunk-overlap",
        "-o",
        type=int,
        default=200,
        help="Number of characters to overlap between chunks (default: 200)",
    )

    args = parser.parse_args()

    try:
        chunks = split_document(
            file_path=args.file,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        print(f"\nSuccessfully split the document into {len(chunks)} chunks:")

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
