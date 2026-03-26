from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

def embed_and_store(chunks, collection_name="research_papers"):

    # --- Embedding model setup ---
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # --- Qdrant client setup ---
    client = QdrantClient(host="localhost", port=6333)

    # --- Create collection if it doesn't exist ---
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE
            )
        )

    # --- Extract texts and generate embeddings ---
    texts = [chunk["text"] for chunk in chunks]

    BATCH_SIZE = 100
    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[i : i + BATCH_SIZE]
        batch_texts = texts[i : i + BATCH_SIZE]

        batch_embeddings = model.encode(batch_texts)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=batch_embeddings[j].tolist(),
                payload={
                    "text": batch_chunks[j]["text"],
                    "page_number": batch_chunks[j]["page_number"],
                    "source_file": batch_chunks[j].get("source_file", "unknown")
                }
            )
            for j in range(len(batch_chunks))
        ]

        client.upsert(
            collection_name=collection_name,
            points=points
        )

        print(f"Stored {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks")