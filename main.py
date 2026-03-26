import sys
from src.ingestion.parser import extract_text_from_pdf
from src.ingestion.chunker import split_text_into_chunks
from src.ingestion.embedder import embed_and_store

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    source_file = pdf_path.split("/")[-1]

    print(f"Parsing {source_file}...")
    pages = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(pages)} pages")

    print("Chunking text...")
    chunks = split_text_into_chunks(pages, source_file)
    print(f"Created {len(chunks)} chunks")

    print("Embedding and storing in Qdrant...")
    embed_and_store(chunks)

    print(f"\nDone. {len(chunks)} chunks from '{source_file}' stored successfully.")

if __name__ == "__main__":
    main()