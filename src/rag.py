import chromadb
import ollama

from config import CHAT_MODEL, COLLECTION_NAME, DB_DIR, EMBED_MODEL

SYSTEM_PROMPT = (
    "You answer questions using ONLY the context provided by the user. "
    "Cite every source you use with its number in square brackets, like [1]. "
    "If the context does not contain the answer, say that you did not find it in the documents. "
    "Never use outside knowledge. Answer in the same language as the question."
)


def retrieve(question: str, top_k: int, max_distance: float, author: str | None = None) -> list[dict]:
    """Return the chunks closest to the question, ignoring the ones that are too far."""
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    # The question must use the SAME embedding model used during ingestion
    query_vector = ollama.embed(model=EMBED_MODEL, input=question)["embeddings"][0]

    where = {"author": author} if author else None
    results = collection.query(query_embeddings=[query_vector], n_results=top_k, where=where)

    hits = []
    for text, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        if distance <= max_distance:
            hits.append({"text": text, "meta": meta, "distance": distance})
    return hits


def format_context(hits: list[dict]) -> str:
    """Number each chunk, so the model can cite it as [1], [2]..."""
    blocks = []
    for number, hit in enumerate(hits, start=1):
        meta = hit["meta"]
        label = f"{meta['book']} | {meta['section']}"
        if meta["author"]:
            label += f" | {meta['signature']}"
        blocks.append(f"[{number}] ({label})\n{hit['text']}")
    return "\n\n".join(blocks)


def answer_stream(question: str, hits: list[dict]):
    """Ask the chat model to answer using only the retrieved chunks, piece by piece."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n\n{format_context(hits)}\n\nQuestion: {question}",
        },
    ]
    for chunk in ollama.chat(model=CHAT_MODEL, messages=messages, stream=True):
        yield chunk["message"]["content"]