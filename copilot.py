import ollama

print("Asking the offline LLM...")

response = ollama.chat(model='llama3.2', messages=[
    {
        'role': 'user',
        'content': 'The network router is experiencing 15% packet loss. Give me 2 quick troubleshooting steps.',
    },
])

print("\nAI Copilot Response:")
print(response['message']['content'])