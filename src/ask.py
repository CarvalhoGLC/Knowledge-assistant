import argparse

from config import MAX_DISTANCE, TOP_K
from rag import answer_stream, retrieve

parser = argparse.ArgumentParser(description="Ask a question and get an answer with sources")
parser.add_argument("question", nargs="*", help="the question")
parser.add_argument("--author", help="only use sections signed by this author (e.g. Erasto)")
parser.add_argument("--top-k", type=int, default=TOP_K)
parser.add_argument("--max-distance", type=float, default=MAX_DISTANCE)
args = parser.parse_args()

question = " ".join(args.question) or input("Question: ")

hits = retrieve(question, args.top_k, args.max_distance, args.author)

# Nothing close enough: do not call the model, so it cannot invent an answer
if not hits:
    print("I could not find relevant passages in the documents.")
    raise SystemExit

print("AI: ", end="", flush=True)
for piece in answer_stream(question, hits):
    print(piece, end="", flush=True)

print("\n\nSources:")
for number, hit in enumerate(hits, start=1):
    meta = hit["meta"]
    print(f"[{number}] {meta['book']} | {meta['chapter']} | {meta['section']} (distance={hit['distance']:.3f})")