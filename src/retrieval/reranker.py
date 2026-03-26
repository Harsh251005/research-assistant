from flashrank import Ranker, RerankRequest

def rerank(query, chunks, top_n=4):

    passages = [{"id": i, "text": chunk["text"]} for i, chunk in enumerate(chunks)]

    ranker = Ranker()
    request = RerankRequest(query=query, passages=passages)
    reranked = ranker.rerank(request)

    final_results = []
    for result in reranked[:top_n]:
        original = chunks[result["id"]]
        final_results.append({
            "text": original["text"],
            "page_number": original["page_number"],
            "source_file": original["source_file"],
            "score": result["score"]
        })

    return final_results