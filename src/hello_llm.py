import ollama

response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {"role": "system", "content": "You are a pirate. Answer like one."},
        {"role": "user", "content": "Explain what an LLM is in one sentence."}
    ],
)

print(response["message"]["content"])