import os
import json
import logging
import pandas as pd
from datetime import datetime
from agents.interfaces.llm_interface import OllamaLLM

# Optional import for cleaner file loading
try:
    from utils.file_loader import load_csv
except ImportError:
    def load_csv(path):
        return pd.read_csv(path)

class AdvisorAgent:
    def __init__(self, model_name=None):
        self.model = model_name if model_name else "tinyllama"
        self.llm = OllamaLLM()  
        self.history = []
        self.paths = {
            "Pricing Insights": "output/overpriced_underpriced_products.csv",
            "Reorder Suggestions": "output/reorder_suggestions.csv",
            "Demand Forecasts": "output/predicted_demand.csv",
            "Understocked Products": "output/understocked_products.csv"
        }
        self.setup_logger()

    def setup_logger(self):
        self.logger = logging.getLogger('AdvisorAgent')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            os.makedirs("logs", exist_ok=True)
            handler = logging.FileHandler('logs/advisor_agent.log')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def load_data(self):
        self.logger.info("Loading data from output files")
        data = {}
        for label, path in self.paths.items():
            if os.path.exists(path):
                try:
                    df = load_csv(path).head(5)
                    data[label] = df.to_dict(orient="records")
                    self.logger.info(f"Loaded {label}")
                except Exception as e:
                    self.logger.error(f"Failed to load {label}: {e}")
            else:
                self.logger.warning(f"{label} file not found at {path}")
        return data

    def generate_context(self, data):
        self.logger.info("Generating context from loaded data")
        blocks = []
        for section, records in data.items():
            block = f"\n### {section} ###\n"
            for idx, record in enumerate(records):
                clean_record = {k: v for k, v in record.items() if pd.notnull(v)}
                block += f"{idx + 1}. " + ", ".join([f"{k}: {v}" for k, v in clean_record.items()]) + "\n"
            blocks.append(block)
        return "\n".join(blocks)

    def simple_response(self, user_query, context_data):
        query = user_query.lower()

        if "understock" in query:
            products = context_data.get("Understocked Products", [])
            return f"Understocked items: {', '.join([p.get('Product ID', 'Unknown') for p in products[:3]])}"

        if "product" in query or "item" in query:
            products = context_data.get("Pricing Insights", [])
            return f"Products: {', '.join([p.get('Product ID', 'Unknown') for p in products[:3]])}"

        if "price" in query:
            return "Pricing data is available in the pricing insights report."

        if "reorder" in query or "stock" in query:
            return "Reorder suggestions are available in the inventory reports."

        if "forecast" in query:
            return "Forecasted demand data is available for key SKUs."

        return "I can provide information about pricing, stock levels, reorder priorities, or demand forecast. Try asking about 'understocked items' or 'forecast for product X'."

    def ask_llm(self, user_query, context):
        available_models = self.llm.get_available_models()
        models_to_try = [m for m in ["tinyllama", "phi3", "llama3", "mistral"]
                        if any(m in avail for avail in available_models)]


        if not models_to_try:
            self.logger.warning("No LLM models available - using fallback response")
            context_data = self.load_data()
            return self.simple_response(user_query, context_data)

        # Build system context and history
        if not any(msg["role"] == "system" for msg in self.history):
            self.history.extend([
                {
                    "role": "system",
                    "content": (
                        "You are AdvisorAgent, an expert inventory assistant. "
                        "Use the provided context to answer questions about inventory status, "
                        "pricing optimization, and restocking suggestions. "
                        "Respond concisely and clearly."
                    )
                },
                {"role": "system", "content": f"Context: {context}"}
            ])
        self.history.append({"role": "user", "content": user_query})

        for model in models_to_try:
            self.logger.info(f"Attempting with model: {model}")
            response = self.llm.ask(model=model, messages=self.history)
            if response:
                self.history.append({"role": "assistant", "content": response})
                return response

        # All models failed, fallback
        context_data = self.load_data()
        return self.simple_response(user_query, context_data)

    def cleanup(self):
        self.history = []
        self.logger.info("Cleared conversation history")

    def ask(self, user_query):
        self.logger.info(f"User Query: {user_query}")
        try:
            context_data = self.load_data()
            context = self.generate_context(context_data)
            result = self.ask_llm(user_query, context)
            self.cleanup()
            return result
        except Exception as e:
            self.cleanup()
            self.logger.error(f"Error processing query: {str(e)}")
            return f"Error processing your request: {str(e)}"
