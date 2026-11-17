import math

import ollama

EMBED_MODEL = "nomic-embed-text"

sentences = [
    "How do I reset my password?",
    "I forgot my login credentials and cannot sign in.",
    "The cat is sleeping on the sofa.",
    "Python is a popular programming language.",
    "What is the recipe for chocolate cake?",
]

query = "I can't access my account"


def cosine_similarity(a, b):
    # dot product divided by the product of the vector lengths
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


# Turn every text into a vector (a list of numbers)
sentence_vectors = ollama.embed(model=EMBED_MODEL, input=sentences)["embeddings"]
query_vector = ollama.embed(model=EMBED_MODEL, input=query)["embeddings"][0]

print(f"Vector size: {len(query_vector)} numbers")
print(f"First 5 numbers: {query_vector[:5]}\n")

# Compare the query with every sentence
results = []
for sentence, vector in zip(sentences, sentence_vectors):
    score = cosine_similarity(query_vector, vector)
    results.append((score, sentence))

# Show the most similar first
results.sort(reverse=True)

print(f"Query: {query}\n")
for score, sentence in results:
    print(f"{score:.3f}  {sentence}")