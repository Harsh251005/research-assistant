from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_text_into_chunks(text, source_file):
    """Split the text into a list of chunks"""
    all_chunks = []

    for page_number, page_text in enumerate(text):
        if len(page_text.strip()) < 50:
            continue

        chunks = text_splitter(page_text)
        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "page_number": page_number + 1,
                "source_file": source_file
            })

    return all_chunks


def text_splitter(text):
    """Text splitter"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=200,
    )

    chunks = splitter.split_text(text)

    return chunks
