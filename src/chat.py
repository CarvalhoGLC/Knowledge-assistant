import ollama

MODEL = "llama3.2:3b"

history = [
    {"role": "system", "content": "You are a helpful assistant. Keep answers short."}
]

print("Chat started. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.strip().lower() == "exit":
        break

    # Add the user message to the history
    history.append({"role": "user", "content": user_input})

    # Send the whole history and receive the answer in small pieces
    stream = ollama.chat(model=MODEL, messages=history, stream=True)

    print("AI: ", end="", flush=True)
    answer = ""
    for chunk in stream:
        piece = chunk["message"]["content"]
        answer += piece
        print(piece, end="", flush=True)
    print("\n")

    # Add the model answer to the history, so the next turn has context
    history.append({"role": "assistant", "content": answer})