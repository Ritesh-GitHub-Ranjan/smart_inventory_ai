# main.py

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from agents.orders.reorder_agent import ReorderAgent
from agents.core.inventory_monitor import InventoryMonitor
from agents.forecasting.pricing_agent import PricingAgent
from agents.interfaces.advisor_agent import AdvisorAgent
from agents.forecasting.sales_impact_agent import SalesImpactAgent
from agents.forecasting.forecasting_agent import ForecastingAgent
from agents.core.audit_agent import AuditAgent
from agents.orders.auto_reorder import AutoReorder
from agents.orders.stock_redistribution_agent import StockRedistribution

# Setup
st.set_page_config(page_title="Smart Inventory Monitoring System", layout="wide")
logging.basicConfig(filename='logs/main.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Utility Functions
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

def plot_demand_trend(df):
    if "Date" not in df.columns or "Sales Quantity" not in df.columns:
        return
    df["Date"] = pd.to_datetime(df["Date"])
    daily_demand = df.groupby("Date")["Sales Quantity"].sum().reset_index()

    plt.figure(figsize=(10, 4))
    sns.lineplot(data=daily_demand, x="Date", y="Sales Quantity")
    plt.title("Daily Sales Quantity Over Time")
    st.pyplot(plt.gcf())
    plt.clf()

def apply_filters(df, product_id, store_id):
    if not df.empty:
        if product_id:
            df = df[df["Product ID"].astype(str).str.contains(product_id)]
        if store_id:
            df = df[df["Store ID"].astype(str).str.contains(store_id)]
    return df

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

# Run All Agents
if run_now or auto_refresh:
    try:
        forecast_agent = ForecastingAgent(demand_data_path="data/demand_forecasting.csv")
        forecast_result = forecast_agent.run()
        forecast_df = forecast_result["forecast_df"] if isinstance(forecast_result, dict) else None

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
forecast_data = data_store["forecast_data"]
reorder_data = data_store["reorder_data"]
inventory_data = data_store["inventory_data"]
overpriced = data_store["overpriced"]
underpriced = data_store["underpriced"]
sales_impact = data_store["sales_impact"]
audit_logs = data_store["audit_logs"]
redistribution = data_store["redistribution"]

# Merge Pricing Data
if not overpriced.empty or not underpriced.empty:
    overpriced["Status"] = "Overpriced"
    underpriced["Status"] = "Underpriced"
    pricing_data = pd.concat([overpriced, underpriced], ignore_index=True)
else:
    pricing_data = pd.DataFrame()

# UI Tabs
st.title("Smart Inventory Monitoring Dashboard")
tab1, tab2 = st.tabs(["📊 Dashboard", "💬 Advisor"])

# Dashboard Tab
with tab1:
    st.header("📉 Key Visuals and Reports")

    col1, col2, col3 = st.columns(3)
    col1.metric("Stockout Risk", "1050 items")
    col2.metric("Expiring Soon", "10000 items")
    col3.metric("Low Capacity Items", "0")

    # Graphs
    st.subheader("📌 Visual Insights")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("#### Stock Levels vs Reorder Point")
        st.image("output/stock_vs_reorder.png", use_column_width=True)
    with col2:
        st.markdown("#### Expiring Products")
        st.image("output/expiry_pie.png", use_column_width=True)

    st.markdown("#### Warehouse Capacity Distribution")
    st.image("output/capacity_histogram.png", width=600)

    # Data Tables
    def display_table(title, df, key):
        with st.expander(f"📁 {title}", expanded=False):
            filtered = apply_filters(df, product_id, store_id)
            st.dataframe(filtered, use_container_width=True)
            if not filtered.empty:
                csv = filtered.to_csv(index=False).encode('utf-8')
                st.download_button(f"⬇️ Export {title}", data=csv, file_name=f"{key}.csv", mime='text/csv')

    display_table("Forecasted Demand", forecast_data, "forecasted_demand")
    display_table("Reorder Suggestions", reorder_data, "reorder_suggestions")
    display_table("Inventory Monitoring", inventory_data, "inventory_data")
    display_table("Pricing Optimization", pricing_data, "pricing_data")
    display_table("Sales Impact Analysis", sales_impact, "sales_impact")
    display_table("Audit Logs", audit_logs, "audit_logs")
    display_table("Stock Redistribution", redistribution, "redistribution")

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
        if operation == "Sell Stock":
            df.loc[mask, "Stock Levels"] -= amount
        else:
            df.loc[mask, "Stock Levels"] += amount
        df.to_csv("data/inventory_monitoring.csv", index=False)
        st.sidebar.success("Inventory updated successfully.")
