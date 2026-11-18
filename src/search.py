import argparse

import chromadb
import ollama

from config import COLLECTION_NAME, DB_DIR, EMBED_MODEL

parser = argparse.ArgumentParser(description="Semantic search over the ingested documents")
parser.add_argument("question", nargs="*", help="the question to search for")
parser.add_argument("--author", help="only search sections signed by this author (e.g. Erasto)")
parser.add_argument("--top-k", type=int, default=3, help="how many chunks to return")
args = parser.parse_args()

question = " ".join(args.question) or input("Question: ")

client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_collection(COLLECTION_NAME)

# The question must use the SAME embedding model used during ingestion
query_vector = ollama.embed(model=EMBED_MODEL, input=question)["embeddings"][0]

# A metadata filter narrows the search BEFORE comparing vectors
where = {"author": args.author} if args.author else None

results = collection.query(
    query_embeddings=[query_vector],
    n_results=args.top_k,
    where=where,
)

for doc, meta, distance in zip(
    results["documents"][0], results["metadatas"][0], results["distances"][0]
):
    print(f"[{meta['book']} | {meta['section']}] distance={distance:.3f}")
    if meta["author"]:
        print(f"Autor: {meta['signature']} ({meta['place_year']})")
    print(doc)
    print("-" * 40)