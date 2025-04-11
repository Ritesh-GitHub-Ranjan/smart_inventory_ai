import plotly.express as px
import pandas as pd

# Inventory Heatmap
def create_inventory_heatmap(inventory_df):
    """
    Create a heatmap to visualize inventory levels across stores and products.
    """
    heatmap = px.imshow(
        inventory_df.pivot('Product ID', 'Store ID', 'Stock Levels'),
        labels=dict(x="Store ID", y="Product ID", color="Stock Levels"),
        title="Inventory Heatmap"
    )
    return heatmap

# Reorder Trends
def create_reorder_trends_plot(reorder_df):
    """
    Create a plot to visualize how inventory levels have changed due to reorder actions.
    """
    reorder_trends = px.line(reorder_df, x='Date', y='Stock Levels', color='Product ID', title="Reorder Trends")
    return reorder_trends

# Under/Over-stock Pie Chart
def create_stock_pie_chart(inventory_df):
    """
    Create a pie chart showing understocked and overstocked products based on reorder points.
    """
    understocked = inventory_df[inventory_df['Stock Levels'] < inventory_df['Reorder Point']]
    overstocked = inventory_df[inventory_df['Stock Levels'] > inventory_df['Reorder Point']]
    
    data = {
        'Understocked': len(understocked),
        'Overstocked': len(overstocked)
    }
    
    pie_chart = px.pie(values=data.values(), names=data.keys(), title="Understocked vs Overstocked Products")
    return pie_chart

# Before-After Impact of Pricing
def create_pricing_impact_plot(pricing_df, before_price_col='Price', after_price_col='Adjusted Price', sales_col='Sales Volume'):
    """
    Create a plot comparing sales and inventory levels before and after price adjustments.
    """
    pricing_impact = px.scatter(
        pricing_df,
        x=before_price_col,
        y=sales_col,
        color=after_price_col,
        title="Before-After Pricing Impact"
    )
    return pricing_impact
