# check_inventory_schema.py

import sqlite3
import pandas as pd

# Connect to the SQLite database
conn = sqlite3.connect('db/inventory.db')

# Load the inventory table
query = "SELECT * FROM inventory"
df = pd.read_sql_query(query, conn)

# Print column names and a sample row
print("📌 Columns in 'inventory' table:", df.columns.tolist())
print("\n📊 Sample data:\n", df.head())

conn.close()
