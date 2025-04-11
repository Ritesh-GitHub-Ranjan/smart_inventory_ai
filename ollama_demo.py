import ollama

# Set your model name (e.g., "mistral" or the one you've pulled)
model_name = "mistral"

# A simple test prompt
prompt = "Explain what Agent AI is in simple words."

# Send prompt to Ollama model
response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])

# Print the model's response
print("AI Response:", response["message"]["content"])
