import pandas as pd
import os
import logging

class AuditAgent:
    def __init__(self, inventory_path="data/inventory_monitoring.csv", forecast_path="output/predicted_demand.csv"):
        self.inventory_path = inventory_path
        self.forecast_path = forecast_path
        self.output_path = "output/inventory_audit.csv"
        self.threshold = 20  # Customize as needed

        self.logger = logging.getLogger('AuditAgent')
        handler = logging.FileHandler('logs/audit_agent.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def run(self):
        self.logger.info("Running AuditAgent...")

        if not os.path.exists(self.inventory_path) or not os.path.exists(self.forecast_path):
            self.logger.error("Required files not found.")
            print("Required files not found.")
            return

        inventory_df = pd.read_csv(self.inventory_path)
        forecast_df = pd.read_csv(self.forecast_path)

        self.logger.info("Files loaded successfully.")

        # Merge on Product ID and Store ID
        merged = pd.merge(
            inventory_df, forecast_df,
            on=["Product ID", "Store ID"],
            how="inner"
        )

        # Calculate difference
        merged["Stock Gap"] = (merged["Stock Levels"] - merged["Predicted Sales Quantity"]).abs()


        # Flag mismatches
        audit_df = merged[merged["Stock Gap"] > self.threshold].copy()
        audit_df["Flag"] = "Mismatch"

        # Save results
        audit_df.to_csv(self.output_path, index=False)

        self.logger.info(f"Audit completed: {len(audit_df)} mismatches found.")
        print(f"Audit completed: {len(audit_df)} mismatches found.")

        self.logger.info("Finished AuditAgent")
