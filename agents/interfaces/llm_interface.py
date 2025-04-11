# agents/interfaces/llm_interface.py

import requests

class OllamaLLM:
    def __init__(self, host="http://localhost:11434"):
        self.base_url = host

    def get_available_models(self):
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            return [m["name"] for m in response.json().get("models", [])]
        except Exception as e:
            print(f"[LLM Error] Failed to fetch models: {e}")
            return []

    def ask(self, model, messages):
        try:
            payload = {
                "model": model,
                "messages": messages,
                "stream": False
            }
            response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "[No response]")
        except Exception as e:
            print(f"[LLM Error] Failed to get response from model '{model}': {e}")
            return "[Error contacting LLM]"
