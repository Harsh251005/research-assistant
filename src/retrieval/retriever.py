from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

def retrieve(query, collection_name, top_k=10):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embedded_query = model.encode(query).tolist()

    client = QdrantClient(host="localhost", port=6333)

    search_result = client.query_points(
        collection_name=collection_name,
        query=embedded_query,
        with_payload=True,
        limit=top_k,
    )

    scored_results = []

    for result in search_result.points:
        scored_results.append({
            "text": result.payload['text'],
            "page_number": result.payload['page_number'],
            "source_file": result.payload['source_file'],
            "score": result.score,
        })

    print(scored_results)

    return scored_results

retrieve("What is Attention", "research_papers")