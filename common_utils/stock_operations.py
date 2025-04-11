# utils/stock_operations.py

from agents.orders.stock_updater import StockUpdater
from agents.orders.reorder_agent import ReorderAgent
from datetime import datetime
import csv
import os
import logging
import pandas as pd

# Configure logging
logger = logging.getLogger("StockOperations")
logger.setLevel(logging.INFO)

def validate_store_id(store_id: str) -> bool:
    """Validate Store ID format and existence.
    
    Args:
        store_id: The Store ID to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(store_id, str) or not store_id.strip():
        logger.error(f"Invalid Store ID: {store_id}")
        return False
    return True

def get_valid_store_ids() -> list:
    """Get list of valid Store IDs from inventory data.
    
    Returns:
        list: Unique Store IDs from inventory
    """
    try:
        inv_df = pd.read_csv("data/inventory_monitoring.csv")
        return inv_df['Store ID'].astype(str).unique().tolist()
    except Exception as e:
        logger.error(f"Failed to load Store IDs: {str(e)}")
        return []

def log_transaction(product_id, store_id, qty, tx_type):
    """Log inventory transaction with validation."""
    if not validate_store_id(store_id):
        raise ValueError(f"Invalid Store ID: {store_id}")
        
    os.makedirs("logs", exist_ok=True)
    with open("logs/stock_updates.csv", mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), product_id, store_id, qty, tx_type])
    logger.info(f"Logged {tx_type} for {product_id} at store {store_id}")

def process_sale(product_id, store_id, sold_qty):
    """Process sale transaction with validation."""
    if not validate_store_id(store_id):
        raise ValueError(f"Invalid Store ID: {store_id}")
        
    updater = StockUpdater()
    msg = updater.update_sales(product_id, store_id, sold_qty)
    log_transaction(product_id, store_id, sold_qty, "SALE")
    logger.info(f"Processed sale of {sold_qty} {product_id} at store {store_id}")
    
    # Trigger reorder check
    reorder = ReorderAgent(
        inventory_path="data/inventory_monitoring.csv",
        forecast_path="output/predicted_demand.csv"
    )
    reorder.run()
    return msg

def process_stock_in(product_id, store_id, added_qty):
    """Process stock restock with validation."""
    if not validate_store_id(store_id):
        raise ValueError(f"Invalid Store ID: {store_id}")
        
    updater = StockUpdater()
    msg = updater.update_incoming_stock(product_id, store_id, added_qty)
    log_transaction(product_id, store_id, added_qty, "RESTOCK")
    logger.info(f"Processed restock of {added_qty} {product_id} at store {store_id}")
    
    # Trigger reorder check
    reorder = ReorderAgent(
        inventory_path="data/inventory_monitoring.csv",
        forecast_path="output/predicted_demand.csv"
    )
    reorder.run()
    return msg
