import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def render_main_dashboard(data_store, product_id, store_id):
    st.title("📊 Smart Inventory Monitoring System")

    # Check for required columns before filtering
    required_filter_cols = {'Product ID', 'Store ID'}
    if not required_filter_cols.issubset(data_store.columns):
        st.warning(f"Missing required columns for filtering: {required_filter_cols - set(data_store.columns)}")
        st.stop()

    # Filter inventory data
    inventory_data = data_store[
        (data_store['Product ID'] == product_id) &
        (data_store['Store ID'] == store_id)
    ]

    # Key Metrics
    st.subheader("📌 Key Metrics")

    col1, col2, col3 = st.columns(3)

    # Metric 1: Stockout Risk
    if 'Stock Levels' in inventory_data.columns and 'Reorder Point' in inventory_data.columns:
        stockout_risk = (inventory_data['Stock Levels'] <= inventory_data['Reorder Point']).sum()
        col1.metric("Stockout Risk", f"{stockout_risk} items")
    else:
        col1.metric("Stockout Risk", "N/A")

    # Metric 2: Expiring Soon
    if 'Days to Expiry' in inventory_data.columns:
        expiring_soon = (inventory_data['Days to Expiry'] <= 30).sum()
        col2.metric("Expiring Soon", f"{expiring_soon} items")
    else:
        col2.metric("Expiring Soon", "N/A")

    # Metric 3: Low Capacity Items
    if 'Warehouse Capacity' in inventory_data.columns:
        low_capacity = (inventory_data['Warehouse Capacity'] < 10).sum()
        col3.metric("Low Capacity Items", f"{low_capacity} items")
    else:
        col3.metric("Low Capacity Items", "N/A")

    # Visual Insights
    st.subheader("📌 Visual Insights")

    # Stock vs Reorder Point
    if {'Product Name', 'Stock Levels', 'Reorder Point'}.issubset(inventory_data.columns):
        st.markdown("**📉 Stock Levels vs Reorder Point**")
        fig1 = px.bar(inventory_data, x="Product Name", y=["Stock Levels", "Reorder Point"],
                      barmode="group", title="Stock vs Reorder Point")
        st.plotly_chart(fig1)

    # Expiry Pie Chart
    if 'Expiry Date' in inventory_data.columns and 'Product Name' in inventory_data.columns:
        st.markdown("**🥫 Expiry Overview**")
        fig2 = px.pie(inventory_data, names='Product Name', title="Expiry Distribution")
        st.plotly_chart(fig2)

    # Warehouse Capacity Histogram
    if 'Warehouse Capacity' in inventory_data.columns:
        st.markdown("**🏭 Warehouse Capacity**")
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(x=inventory_data['Warehouse Capacity']))
        fig3.update_layout(title='Warehouse Capacity Histogram', xaxis_title='Capacity', yaxis_title='Count')
        st.plotly_chart(fig3)
