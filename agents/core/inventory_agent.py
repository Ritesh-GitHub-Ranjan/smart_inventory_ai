# inventory_agent.py
import pandas as pd
import sqlite3

class InventoryAgent:
    def __init__(self, db_path = None, low_stock_threshold=10, overstock_threshold=300):
        self.db_path = db_path
        self.low_stock_threshold = low_stock_threshold
        self.overstock_threshold = overstock_threshold

    def load_data(self):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM inventory", conn)
        conn.close()
        return df

    def analyze_stock(self, df):
        out_of_stock = df[df['Stock Levels'] == 0]
        low_stock = df[(df['Stock Levels'] > 0) & (df['Stock Levels'] < self.low_stock_threshold)]
        overstock = df[df['Stock Levels'] > self.overstock_threshold]

        return {
            "out_of_stock": out_of_stock,
            "low_stock": low_stock,
            "overstock": overstock
        }
