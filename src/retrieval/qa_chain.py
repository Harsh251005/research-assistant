import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

def answer(query, chunks):

    context = ""

    for i, chunk in enumerate(chunks):
        context += f"[{i + 1}] {chunk['source_file']} Page: {chunk['page_number']}\n"
        context += f"{chunk['text']}\n\n"

    prompt = f"""
    You are a research assistant. Answer the question using ONLY 
    the provided context. For every claim, cite the source as [1], [2] etc.
    If the answer isn't in the context, say "I don't have enough information."

    Context:
    {context}

    Question: {query}

    Answer:
    """

    llm = ChatGroq(
        model="qwen/qwen3-32b",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    response = llm.invoke([HumanMessage(prompt)])

    return response.content