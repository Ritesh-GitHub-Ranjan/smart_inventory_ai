# llm_connector.py

import requests
import json
import logging

class LLMConnector:
    def __init__(self, base_url="http://localhost:11434/api/chat"):
        self.url = base_url
        self.headers = {"Content-Type": "application/json"}
        self.timeout = 30
        self.logger = logging.getLogger('LLMConnector')
        self.optimized_settings = {
            "num_ctx": 512,
            "num_thread": 1,
            "num_gqa": 1,
            "num_gpu": 0,
            "low_vram": True
        }

    def get_available_models(self):
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            return [m["name"] for m in response.json().get("models", [])]
        except Exception as e:
            self.logger.error(f"Error fetching models from Ollama: {e}")
            return []

    def ask(self, model, messages):
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": self.optimized_settings
        }

        try:
            response = requests.post(
                self.url,
                headers=self.headers,
                data=json.dumps(payload, ensure_ascii=False),
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "No response received.")
        except Exception as e:
            self.logger.warning(f"Request to model {model} failed: {str(e)}")
            return None
