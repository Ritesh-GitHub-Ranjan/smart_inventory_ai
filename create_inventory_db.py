# create_inventory_db.py

import sqlite3
import pandas as pd
import os

# Paths
csv_path = 'data/inventory_monitoring.csv'
db_path = 'db/inventory.db'

# Create DB folder if not exists
os.makedirs('db', exist_ok=True)

# Load CSV
df = pd.read_csv(csv_path)

# Optional: Display schema preview
print("🧾 Inventory CSV Columns:", df.columns)

# Save to SQLite
conn = sqlite3.connect(db_path)
df.to_sql('inventory', conn, if_exists='replace', index=False)
conn.close()

print("✅ inventory table created successfully in inventory.db")
