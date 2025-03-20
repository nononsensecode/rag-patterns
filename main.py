import argparse
from text_splitter import split_document
from summarize import summarize_text
from dotenv import load_dotenv

load_dotenv()


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
