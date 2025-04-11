# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
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


# --- Utility Functions ---
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


# --- Sidebar ---
st.set_page_config(page_title="Smart Inventory Dashboard", layout="wide")
st.sidebar.title("Controls")
run_once = st.sidebar.button("Run All Agents Once")
refresh_data = st.sidebar.button("🔄 Refresh Data")

# Filters
product_id = st.sidebar.text_input("Filter by Product ID")
store_id = st.sidebar.text_input("Filter by Store ID")
selected_model = st.sidebar.selectbox("Advisor LLM Model", ["llama3", "phi3", "mistral"])


# --- Run All Agents ---
if run_once:
    forecast_agent = ForecastingAgent(demand_data_path="data/demand_forecasting.csv")
    forecast_result = forecast_agent.run()
    if not isinstance(forecast_result, dict) or "forecast_df" not in forecast_result:
        st.error("Forecasting failed - invalid return type or missing forecast_df")
        st.stop()
    forecast_df = forecast_result["forecast_df"]

    inventory_path = "data/inventory_monitoring.csv"
    if not os.path.exists(inventory_path):
        st.error(f"Inventory file not found at {inventory_path}")
        st.stop()

    inventory_agent = InventoryMonitor(inventory_path=inventory_path)
    inventory_agent.run()

    try:
        reorder_agent = ReorderAgent(forecast_df=forecast_df, inventory_path=inventory_path)
        reorder_agent.run()
    except Exception as e:
        st.error(f"Reorder agent failed: {str(e)}")
        st.stop()

    pricing_agent = PricingAgent(pricing_data_path="data/pricing_optimization.csv")
    pricing_agent.run()

    sales_agent = SalesImpactAgent(sales_data_path="data/pricing_optimization.csv")
    sales_agent.run()

    audit_agent = AuditAgent(inventory_data_path="data/inventory_monitoring.csv")
    audit_agent.run()

    advisor_agent = AdvisorAgent(model_name=selected_model)
    advisor_agent.run()

    auto_reorder = AutoReorder(
        reorder_path="output/reorder_suggestions.csv",
        inventory_path="data/inventory_monitoring.csv"
    )
    auto_reorder.process_reorders()

    try:
        inventory_df = pd.read_csv("data/inventory_monitoring.csv")
        redistributor = StockRedistribution(inventory_df=inventory_df)
        redistributor.redistribute_stock()
    except Exception as e:
        st.error(f"Stock redistribution failed: {e}")

    st.success("All agents executed successfully!")


# --- Load or Refresh Data ---
if "data_store" not in st.session_state or refresh_data:
    st.session_state.data_store = reload_all_data()
    st.sidebar.success(f"Data refreshed at {datetime.now().strftime('%H:%M:%S')}")

data_store = st.session_state.data_store
forecast_data = data_store["forecast_data"]
reorder_data = data_store["reorder_data"]
inventory_data = data_store["inventory_data"]
overpriced = data_store["overpriced"]
underpriced = data_store["underpriced"]
sales_impact_data = data_store["sales_impact"]
audit_logs = data_store["audit_logs"]
redistribution_data = data_store["redistribution"]

# Combine pricing data
if not overpriced.empty or not underpriced.empty:
    overpriced["Status"] = "Overpriced"
    underpriced["Status"] = "Underpriced"
    pricing_data = pd.concat([overpriced, underpriced], ignore_index=True)
else:
    pricing_data = pd.DataFrame()


# --- Filter Logic ---
def apply_filters(df):
    if not df.empty:
        if product_id:
            df = df[df["Product ID"].astype(str).str.contains(product_id)]
        if store_id:
            df = df[df["Store ID"].astype(str).str.contains(store_id)]
    return df


# --- Dashboard UI ---
st.title("Smart Inventory Monitoring Dashboard")
st.markdown("Monitor, analyze, and optimize your retail inventory operations.")

# Forecast Section
st.subheader("Forecasted Demand")
filtered_forecast = apply_filters(forecast_data)
st.dataframe(filtered_forecast, use_container_width=True)
plot_demand_trend(filtered_forecast)

# Reorder Suggestions
st.subheader("Reorder Suggestions")
filtered_reorder = apply_filters(reorder_data)
st.dataframe(filtered_reorder, use_container_width=True)

# Inventory Monitoring
st.subheader("Inventory Monitoring")
filtered_inventory = apply_filters(inventory_data)
st.dataframe(filtered_inventory, use_container_width=True)

# Pricing Optimization
st.subheader("Pricing Optimization")
filtered_pricing = apply_filters(pricing_data)
st.dataframe(filtered_pricing, use_container_width=True)

# Sales Impact Analysis
st.subheader("Sales Impact Analysis")
filtered_sales_impact = apply_filters(sales_impact_data)
st.dataframe(filtered_sales_impact, use_container_width=True)

# Audit Logs
st.subheader("Inventory Audit Logs")
filtered_audit = apply_filters(audit_logs)
st.dataframe(filtered_audit, use_container_width=True)

# Stock Redistribution
st.subheader("Stock Redistribution Recommendations")
filtered_redistribution = apply_filters(redistribution_data)
st.dataframe(filtered_redistribution, use_container_width=True)


# Advisor Chat Agent
st.subheader("Advisor Agent (LLM)")
if "advisor_agent" not in st.session_state or st.session_state.get("selected_model") != selected_model:
    st.session_state.advisor_agent = AdvisorAgent(model_name=selected_model)
    st.session_state.selected_model = selected_model

query = st.text_input("Ask your inventory-related question:")
if query:
    response = st.session_state.advisor_agent.ask(query)
    st.write("Advisor Response:", response)


# Manual Inventory Update
st.sidebar.markdown("---")
st.sidebar.subheader("Manual Inventory Update")
operation = st.sidebar.radio("Choose Operation", ["Sell Stock", "Restock Inventory"])
selected_pid = st.sidebar.text_input("Product ID")
selected_sid = st.sidebar.text_input("Store ID")
amount = st.sidebar.number_input("Amount", min_value=1, step=1)

if st.sidebar.button("Update Inventory"):
    df = pd.read_csv("data/inventory_monitoring.csv")
    mask = (df["Product ID"] == selected_pid) & (df["Store ID"] == selected_sid)
    if df[mask].empty:
        st.sidebar.error("No matching product-store record found.")
    else:
        if operation == "Sell Stock":
            df.loc[mask, "Stock Levels"] -= amount
        else:
            df.loc[mask, "Stock Levels"] += amount
        df.to_csv("data/inventory_monitoring.csv", index=False)
        st.sidebar.success("Inventory updated successfully!")
