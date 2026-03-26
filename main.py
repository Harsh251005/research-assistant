import sys
from src.ingestion.parser import extract_text_from_pdf
from src.ingestion.chunker import split_text_into_chunks
from src.ingestion.embedder import embed_and_store
from src.retrieval.retriever import retrieve
from src.retrieval.reranker import rerank
from src.retrieval.qa_chain import answer

def ingest(pdf_path):
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


def query(user_query):
    print(f"\nSearching for: {user_query}")
    retrieved = retrieve(user_query, "research_papers")
    print(f"Retrieved {len(retrieved)} chunks")

    reranked = rerank(user_query, retrieved)
    print(f"Reranked to top {len(reranked)} chunks")

    response = answer(user_query, reranked)
    print(f"\nAnswer:\n{response}")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Ingest a PDF  : python main.py <path_to_pdf>")
        print("  Ask a question: python main.py --query 'your question here'")
        sys.exit(1)

    if sys.argv[1] == "--query":
        if len(sys.argv) < 3:
            print("Please provide a question after --query")
            sys.exit(1)
        user_query = sys.argv[2]
        query(user_query)
    else:
        pdf_path = sys.argv[1]
        ingest(pdf_path)


if __name__ == "__main__":
    main()
