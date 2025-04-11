import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

def export_button(df, filename, label):
    if not df.empty:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=label,
            data=csv,
            file_name=filename,
            mime="text/csv"
        )

def render_interactive_line_chart(df, x_col, y_col, title):
    fig = px.line(df, x=x_col, y=y_col, title=title)
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

def render_inventory_dashboard(data_store, product_id, store_id):
    st.title("📊 Inventory Monitoring Dashboard")

    inventory_data = data_store["inventory_data"]
    reorder_data = data_store["reorder_data"]

    st.markdown("### Summary")
    st.markdown(f"""
        - **Stockout Risk**: {inventory_data[inventory_data['Stock Levels'] <= inventory_data['Reorder Point']].shape[0]} items  
        - **Expiring Soon**: {inventory_data[inventory_data['Days to Expiry'] <= 30].shape[0]} items  
        - **Low Warehouse Capacity Items**: {inventory_data[inventory_data['Warehouse Capacity'] < 10].shape[0]} items  
    """)

    with st.expander("📉 Stock Levels vs Reorder Point", expanded=False):
        if not inventory_data.empty:
            fig = px.scatter(inventory_data, x="Product ID", y="Stock Levels",
                             color=(inventory_data["Stock Levels"] <= inventory_data["Reorder Point"]),
                             labels={"color": "Below Reorder?"}, title="Stock Levels vs Reorder Point")
            fig.add_scatter(x=inventory_data["Product ID"], y=inventory_data["Reorder Point"],
                            mode="lines", name="Reorder Point", line=dict(color="red"))
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("🥧 Expiry Distribution", expanded=False):
        if "Days to Expiry" in inventory_data.columns:
            inventory_data["Expiry Bucket"] = pd.cut(inventory_data["Days to Expiry"],
                                                    bins=[-1, 7, 30, 90, float('inf')],
                                                    labels=["<1 week", "<1 month", "<3 months", "Later"])
            fig = px.pie(inventory_data, names="Expiry Bucket", title="Products Expiry Distribution")
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("🏭 Warehouse Capacity Distribution", expanded=False):
        fig = px.histogram(inventory_data, x="Warehouse Capacity", nbins=20,
                           title="Warehouse Capacity Histogram")
        st.plotly_chart(fig, use_container_width=True)

def render_dashboard(data_store, product_id, store_id, selected_model):
    forecast_data = data_store["forecast_data"]
    reorder_data = data_store["reorder_data"]
    inventory_data = data_store["inventory_data"]
    overpriced = data_store["overpriced"]
    underpriced = data_store["underpriced"]
    sales_impact = data_store["sales_impact"]
    audit_logs = data_store["audit_logs"]
    redistribution = data_store["redistribution"]

    def apply_filters(df):
        if not df.empty:
            if product_id:
                df = df[df["Product ID"].astype(str).str.contains(product_id)]
            if store_id:
                df = df[df["Store ID"].astype(str).str.contains(store_id)]
        return df

    render_inventory_dashboard(data_store, product_id, store_id)

    st.subheader("📈 Forecasted Demand")
    filtered = apply_filters(forecast_data)
    if not filtered.empty:
        st.dataframe(filtered, use_container_width=True)
        filtered["Date"] = pd.to_datetime(filtered["Date"])
        daily = filtered.groupby("Date")["Sales Quantity"].sum().reset_index()
        render_interactive_line_chart(daily, "Date", "Sales Quantity", "Daily Forecasted Sales")
        export_button(filtered, "forecasted_demand.csv", "Export Forecasted Demand")
    else:
        st.info("No forecast data available.")

    st.subheader("🚚 Reorder Suggestions")
    filtered = apply_filters(reorder_data)
    st.dataframe(filtered, use_container_width=True)
    export_button(filtered, "reorder_suggestions.csv", "Export Reorder Suggestions")

    st.subheader("📦 Inventory Monitoring")
    filtered = apply_filters(inventory_data)
    st.dataframe(filtered, use_container_width=True)
    export_button(filtered, "inventory_data.csv", "Export Inventory")

    st.subheader("💰 Pricing Optimization")
    if not overpriced.empty or not underpriced.empty:
        overpriced["Status"] = "Overpriced"
        underpriced["Status"] = "Underpriced"
        pricing_data = pd.concat([overpriced, underpriced], ignore_index=True)
    else:
        pricing_data = pd.DataFrame()
    filtered = apply_filters(pricing_data)
    st.dataframe(filtered, use_container_width=True)
    export_button(filtered, "pricing_data.csv", "Export Pricing Report")

    st.subheader("📊 Sales Impact Analysis")
    filtered = apply_filters(sales_impact)
    st.dataframe(filtered, use_container_width=True)
    export_button(filtered, "sales_impact.csv", "Export Sales Impact")

    st.subheader("🧾 Audit Logs")
    filtered = apply_filters(audit_logs)
    st.dataframe(filtered, use_container_width=True)
    export_button(filtered, "audit_logs.csv", "Export Audit Logs")

    st.subheader("♻️ Stock Redistribution")
    filtered = apply_filters(redistribution)
    st.dataframe(filtered, use_container_width=True)
    export_button(filtered, "redistribution.csv", "Export Redistribution Report")
