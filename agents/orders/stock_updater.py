import pandas as pd
import os

class StockUpdater:
    def __init__(self, inventory_path="data/inventory_monitoring.csv"):
        self.path = inventory_path
        self.df = pd.read_csv(self.path)

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.df.to_csv(self.path, index=False)

    def update_sales(self, product_id, store_id, sold_qty):
        match = (self.df["Product ID"] == product_id) & (self.df["Store ID"] == store_id)

        if not match.any():
            return f"Product {product_id} not found in Store {store_id}"

        current_stock = self.df.loc[match, "Stock Levels"].values[0]
        if sold_qty > current_stock:
            return f"Not enough stock to sell {sold_qty} units. Current stock: {current_stock}"

        self.df.loc[match, "Stock Levels"] -= sold_qty
        self.save()
        return f"Sale recorded: -{sold_qty} units for Product {product_id} at Store {store_id}"

    def update_incoming_stock(self, product_id, store_id, added_qty):
        match = (self.df["Product ID"] == product_id) & (self.df["Store ID"] == store_id)

        if not match.any():
            return f"Product {product_id} not found in Store {store_id}"

        self.df.loc[match, "Stock Levels"] += added_qty
        self.save()
        return f"New stock received: +{added_qty} units for Product {product_id} at Store {store_id}"
