import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import logging
from datetime import datetime

from dashboard import render_main_dashboard
from agents.orders.reorder_agent import ReorderAgent
from agents.core.inventory_monitor import InventoryMonitor
from agents.forecasting.pricing_agent import PricingAgent
from agents.interfaces.advisor_agent import AdvisorAgent
from agents.forecasting.sales_impact_agent import SalesImpactAgent
from agents.forecasting.forecasting_agent import ForecastingAgent
from agents.core.audit_agent import AuditAgent
from agents.orders.auto_reorder import AutoReorder
from agents.orders.stock_redistribution_agent import StockRedistribution

# Config
st.set_page_config(page_title="Smart Inventory Monitoring System", layout="wide")
logging.basicConfig(filename='logs/main.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@st.cache_data(ttl=300, show_spinner=False)
def cached_load_csv(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        st.warning(f"File not found: {path}")
        return pd.DataFrame()

def reload_all_data():
    return {
        "forecast_data": cached_load_csv("output/forecasted_demand.csv"),
        "reorder_data": cached_load_csv("output/reorder_suggestions.csv"),
        "inventory_data": cached_load_csv("data/inventory_monitoring.csv"),
        "overpriced": cached_load_csv("output/overpriced_products.csv"),
        "underpriced": cached_load_csv("output/underpriced_products.csv"),
        "sales_impact": cached_load_csv("output/sales_impact_report.csv"),
        "audit_logs": cached_load_csv("output/inventory_audit.csv"),
        "redistribution": cached_load_csv("output/redistributed_inventory.csv")
    }

# Sidebar Controls
st.sidebar.title("Controls")
auto_refresh = st.sidebar.checkbox("Auto-refresh every 5 minutes")
if auto_refresh:
    st_autorefresh(interval=5 * 60 * 1000, key="auto_refresh")

run_now = st.sidebar.button("Run All Agents Now")
refresh_data = st.sidebar.button("🔄 Refresh Data")
product_id = st.sidebar.text_input("Filter by Product ID")
store_id = st.sidebar.text_input("Filter by Store ID")
selected_model = st.sidebar.selectbox("Advisor LLM Model", ["tinyllama", "phi3", "mistral"])

# Run Agents
if run_now or auto_refresh:
    try:
        forecast_agent = ForecastingAgent(demand_data_path="data/demand_forecasting.csv")
        forecast_result = forecast_agent.run()
        forecast_df = forecast_result.get("forecast_df") if isinstance(forecast_result, dict) else None

        InventoryMonitor().run()
        ReorderAgent(forecast_df=forecast_df, inventory_path="data/inventory_monitoring.csv").run()
        PricingAgent().run()
        SalesImpactAgent().run()
        AuditAgent().run()
        AdvisorAgent(model_name=selected_model).run()
        AutoReorder("output/reorder_suggestions.csv", "data/inventory_monitoring.csv").process_reorders()
        StockRedistribution(inventory_df=pd.read_csv("data/inventory_monitoring.csv")).redistribute_stock()

        st.success("All agents executed successfully!")
    except Exception as e:
        st.error(f"Execution failed: {e}")
        logging.error(f"Execution failed: {e}")

# Load Data
if "data_store" not in st.session_state or refresh_data:
    st.session_state.data_store = reload_all_data()
    st.sidebar.success(f"Data refreshed at {datetime.now().strftime('%H:%M:%S')}")

data_store = st.session_state.data_store

# UI Tabs
st.title("Smart Inventory Monitoring Dashboard")
tab1, tab2 = st.tabs(["📊 Dashboard", "💬 Advisor"])

# Dashboard Tab
with tab1:
    render_main_dashboard(data_store, product_id, store_id)

# Advisor Tab
with tab2:
    st.header("Ask the Inventory Advisor")

    if "advisor_agent" not in st.session_state or st.session_state.get("selected_model") != selected_model:
        st.session_state.advisor_agent = AdvisorAgent(model_name=selected_model)
        st.session_state.selected_model = selected_model

    query = st.text_input("Ask your inventory-related question:")
    if query:
        response = st.session_state.advisor_agent.ask(query)
        st.write("📎 Advisor:", response)

# Manual Inventory Update
st.sidebar.subheader("Manual Inventory Update")
operation = st.sidebar.radio("Choose Operation", ["Sell Stock", "Restock Inventory"])
selected_pid = st.sidebar.text_input("Product ID")
selected_sid = st.sidebar.text_input("Store ID")
amount = st.sidebar.number_input("Amount", min_value=1, step=1)

if st.sidebar.button("Update Inventory"):
    df = pd.read_csv("data/inventory_monitoring.csv")
    mask = (df["Product ID"] == selected_pid) & (df["Store ID"] == selected_sid)
    if df[mask].empty:
        st.sidebar.error("No matching record found.")
    else:
        df.loc[mask, "Stock Levels"] += (-amount if operation == "Sell Stock" else amount)
        df.to_csv("data/inventory_monitoring.csv", index=False)
        st.sidebar.success("Inventory updated successfully.")
