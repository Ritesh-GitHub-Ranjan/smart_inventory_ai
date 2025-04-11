# inventory_monitor.py
import pandas as pd
import os
import matplotlib.pyplot as plt
import webbrowser
import logging
from agents.orders.reorder_agent import ReorderAgent

class InventoryMonitor:
    def __init__(self, inventory_path="data/inventory_monitoring.csv"):
        self.inventory_path = inventory_path
        self.logger = logging.getLogger("InventoryMonitor")
        handler = logging.FileHandler("logs/inventory_monitor.log")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def run(self):
        self.logger.info("Running InventoryMonitor...")
        df = pd.read_csv(self.inventory_path)
        os.makedirs("output", exist_ok=True)

        df['Expiry Date'] = pd.to_datetime(df['Expiry Date'], errors='coerce')

        stockout_risk = df[df["Stock Levels"] < df["Reorder Point"]]
        expiring_soon = df[df["Expiry Date"] <= pd.Timestamp.now() + pd.Timedelta(days=30)]
        low_capacity = df[df["Warehouse Capacity"] < 20]

        report = [
            f"Stockout risk: {len(stockout_risk)} items",
            f"Expiring soon: {len(expiring_soon)} items",
            f"Low warehouse capacity items: {len(low_capacity)}"
        ]
        with open("output/inventory_report.csv", "w", encoding="utf-8") as f:
            for line in report:
                f.write(line + "\n")

        self.logger.info("Inventory report generated")

        if not stockout_risk.empty:
            self.logger.info(f"Detected {len(stockout_risk)} stockout risks — triggering ReorderAgent")
            print("Triggering ReorderAgent due to stockout risks...")
            try:
                forecast_df = pd.read_csv("output/forecasted_demand.csv")
                
                # Validate and prepare forecast data
                if 'Store ID' not in forecast_df.columns:
                    # Use Store IDs from inventory data if available
                    inventory_df = pd.read_csv(self.inventory_path)
                    if len(inventory_df['Store ID'].unique()) == 1:
                        forecast_df['Store ID'] = inventory_df['Store ID'].iloc[0]
                    else:
                        forecast_df['Store ID'] = 'DEFAULT_STORE'
                
                # Ensure consistent data types
                forecast_df['Store ID'] = forecast_df['Store ID'].astype(str)
                forecast_df['Product ID'] = forecast_df['Product ID'].astype(int)
                
                reorder_agent = ReorderAgent(
                    inventory_path=self.inventory_path,
                    forecast_df=forecast_df,
                    output_path="output/reorder_suggestions.csv"
                )
                reorder_agent.run()
                self.logger.info("ReorderAgent triggered successfully from InventoryMonitor")
            except Exception as e:
                self.logger.error(f"Failed to trigger ReorderAgent: {e}")
                print(f"Error triggering ReorderAgent: {e}")

        # Chart 1: Stock vs Reorder Point
        try:
            df_sample = df.head(20)
            plt.figure(figsize=(10, 5))
            plt.bar(df_sample["Product ID"].astype(str), df_sample["Stock Levels"], label="Stock Levels")
            plt.bar(df_sample["Product ID"].astype(str), df_sample["Reorder Point"], alpha=0.5, label="Reorder Point")
            plt.xlabel("Product ID")
            plt.ylabel("Quantity")
            plt.title("Stock vs Reorder Point")
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig("output/stock_vs_reorder.png")
            plt.close()
            self.logger.info("Chart saved: stock_vs_reorder.png")
        except Exception as e:
            self.logger.error(f"Error generating Stock vs Reorder chart: {e}")

        # Chart 2: Expiring Products
        try:
            pie_labels = ["Expiring Soon", "Not Expiring Soon"]
            pie_sizes = [len(expiring_soon), len(df) - len(expiring_soon)]
            plt.figure(figsize=(5, 5))
            plt.pie(pie_sizes, labels=pie_labels, autopct="%1.1f%%", colors=["red", "green"])
            plt.title("Product Expiry (Next 30 Days)")
            plt.savefig("output/expiry_pie.png")
            plt.close()
            self.logger.info("Chart saved: expiry_pie.png")
        except Exception as e:
            self.logger.error(f"Error generating expiry pie chart: {e}")

        # Chart 3: Warehouse Capacity
        try:
            plt.figure(figsize=(7, 5))
            plt.hist(df["Warehouse Capacity"], bins=10, color="skyblue", edgecolor="black")
            plt.xlabel("Warehouse Capacity")
            plt.ylabel("Frequency")
            plt.title("Warehouse Capacity Distribution")
            plt.savefig("output/capacity_histogram.png")
            plt.close()
            self.logger.info("Chart saved: capacity_histogram.png")
        except Exception as e:
            self.logger.error(f"Error generating warehouse capacity chart: {e}")

        print("Charts saved to output/ folder")

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Inventory Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; padding: 30px; background: #f4f4f4; }}
                h1 {{ color: #2d3436; }}
                ul {{ font-size: 18px; }}
                img {{ border: 1px solid #ccc; border-radius: 8px; margin: 20px 0; }}
                .chart-container {{ margin-bottom: 40px; }}
            </style>
        </head>
        <body>
            <h1>Inventory Monitoring Dashboard</h1>
            <ul>
                <li>{report[0]}</li>
                <li>{report[1]}</li>
                <li>{report[2]}</li>
            </ul>

            <div class="chart-container">
                <h2>Stock Levels vs Reorder Point</h2>
                <img src="stock_vs_reorder.png" width="600">
            </div>

            <div class="chart-container">
                <h2>Expiring Products</h2>
                <img src="expiry_pie.png" width="400">
            </div>

            <div class="chart-container">
                <h2>Warehouse Capacity Distribution</h2>
                <img src="capacity_histogram.png" width="500">
            </div>
        </body>
        </html>
        """

        dashboard_path = "output/inventory_dashboard.html"
        with open(dashboard_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        self.logger.info("Inventory dashboard HTML created")
        print("Dashboard created at output/inventory_dashboard.html")
        webbrowser.open(f"file://{os.path.abspath(dashboard_path)}")

        self.logger.info("Finished InventoryMonitor")
