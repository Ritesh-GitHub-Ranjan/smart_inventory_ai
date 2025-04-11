# llama_integration.py

import ollama
import pandas as pd
from archive.forecasting_agent_v2 import ForecastingAgent
from agents.orders.reorder_agent import ReorderAgent
from agents.forecasting.pricing_agent import PricingAgent
from agents.core.audit_agent import AuditAgent

# Function to initialize Ollama API
def connect_to_ollama():
    try:
        client = ollama.connect("localhost")  # Connect to your local Ollama instance
        return client
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return None

# Function to parse and process queries
def process_query(client, query):
    response = client.chat(query)
    return response['message']

# Function to trigger relevant agents based on the query
def trigger_agent_based_on_query(query):
    if "forecast" in query:
        return ForecastingAgent.run_forecast()
    elif "reorder" in query:
        return ReorderAgent.run_reorder()
    elif "price" in query:
        return PricingAgent.optimize_price()
    elif "audit" in query:
        return AuditAgent.run_audit()
    else:
        return "Sorry, I couldn't understand your request."

# Example of integrating the two
def handle_user_query(query, forecast_df, inventory_df, pricing_df):
    client = connect_to_ollama()
    if client:
        # If the query is about inventory, forecast, or pricing, provide the relevant data
        if "inventory" in query.lower():
            data = inventory_df[['Product ID', 'Stock Levels']]
            return f"Current Inventory Levels:\n{data.to_string(index=False)}"
        elif "forecast" in query.lower():
            data = forecast_df[['Product ID', 'prediction', 'pred_lower', 'pred_upper']]
            return f"Latest Demand Forecast:\n{data.to_string(index=False)}"
        elif "pricing" in query.lower():
            data = pricing_df[['Product ID', 'Price', 'Competitor Prices']]
            return f"Pricing Data:\n{data.to_string(index=False)}"

        # Default action if no specific data requested
        response = process_query(client, query)
        agent_result = trigger_agent_based_on_query(response)
        return agent_result
    else:
        return "Unable to connect to Ollama."
